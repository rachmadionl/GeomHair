import errno
import os
import sys
from collections import namedtuple
import random
sys.path.append('..')
import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import cv2 as cv
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import DataLoader
from src.models.dataset import Dataset, MonocularDataset
from src.models.dataset_torch import NPHMHaircutDataset
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.hair_networks.sdf import HairSDFNetwork
from src.hair_networks.deepsdf import DeepSDFDecoder
from src.hair_networks.capudf import CAPUDFNetwork
from src.hair_networks.levelsetudf import LevelSetUDFNetwork
from src.hair_networks.nsh import NSH
from src.strands_trainer import StrandsTrainer
import yaml
from copy import deepcopy

from src.utils.util import set_seed, scale_mat

import warnings
warnings.filterwarnings("ignore")
    
class Runner:
    def __init__(
            self,
            conf_path,
            case='CASE_NAME',
            scene_type='DATASET_TYPE',
            shrink=True,
            use_3do_2do=True,
            use_old_dif_prior=True,
            is_blobs=False,
            diffuse_iter=10000,
            n_diffuse_steps=4,
            checkpoint_name=None,
            hair_conf_path=None,
            exp_name=None,
            exp_basedir=None,
            data_root='DATA_ROOT',
            max_iter=None,
        ):
        
        self.device = torch.device('cuda')
        
        # Configuration of geometry      
        self.conf_path = conf_path 
        with open(conf_path, 'r') as f:
            replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', case).replace('DATA_ROOT', data_root)
            if case == '478':
                print('Detected sample 478, use expression 001...')
                replaced_conf = replaced_conf.replace('/000/', '/001/')
            self.conf = yaml.load(replaced_conf, Loader=yaml.Loader)

        # Configuration of hair strands
        self.hair_conf_path = hair_conf_path
        with open(hair_conf_path) as f:
            replaced_conf = str(yaml.load(f, Loader=yaml.Loader)).replace('CASE_NAME', case).replace('DATA_ROOT', data_root)
            if case == '478':
                replaced_conf = replaced_conf.replace('/000/', '/001/')
            self.hair_conf = yaml.load(replaced_conf, Loader=yaml.Loader)
        
        if not shrink:
            self.hair_conf['textured_strands']['path_to_mesh'] = (
                self.hair_conf['textured_strands']['path_to_mesh']
                .replace('_scaled', '')
            )
        print(f"Scalp used is: {self.hair_conf['textured_strands']['path_to_mesh']}")
        shrink_scalp = True if 'scaled' in self.hair_conf['textured_strands']['path_to_mesh'] else False
        print(f'Setting IS_SHRINK to: {shrink_scalp}')

        if use_3do_2do:
            self.hair_conf['render']['use_render'] = True
        else:
            self.hair_conf['render']['use_render'] = False
        print(f'Setting USE_3DO_2DO to: {self.hair_conf["render"]["use_render"]}')

        if not use_old_dif_prior:
            self.hair_conf['diffusion_prior']['model']['dif_path'] = (
                self.hair_conf['diffusion_prior']['model']['dif_path']
                .replace('dif_ckpt.pth', 'wo_bug_blender_uv_00130000.pth')
            )
        old_dif_prior = (
            False if 'blender' in self.hair_conf['diffusion_prior']['model']['dif_path'] else True
        )
        print(f"Setting OLD_DIF_PRIOR to: {old_dif_prior}")
        print(self.hair_conf['diffusion_prior']['model']['dif_path'])

        self.hair_conf['diffusion_prior']['diffuse_iter'] = diffuse_iter
        print(f"Setting DIFFUSE_ITER to: {self.hair_conf['diffusion_prior']['diffuse_iter']}")

        self.hair_conf['diffusion_prior']['n_diffuse_steps'] = n_diffuse_steps
        print(f"Setting N_DIFFUSE_STEPS to: {self.hair_conf['diffusion_prior']['n_diffuse_steps']}")

        if not is_blobs:
            self.hair_conf['sdf_chamfer']['blob_faces_idx'] = None
            self.hair_conf['general']['orientation_mask'] = None
        
        print(f"Setting blobs_faces_idx to: {self.hair_conf['sdf_chamfer']['blob_faces_idx']}")
        print(f"Setting orientation_mask to: {self.hair_conf['general']['orientation_mask']}")

        train_conf = self.conf['train']
        self.end_iter = train_conf['end_iter']
        if max_iter is not None:
            self.end_iter = min(self.end_iter, max_iter)
            print(f'Capping end_iter to: {self.end_iter}')
        self.report_freq = train_conf['report_freq']
        self.batch_size = train_conf['batch_size']
        
        if exp_name is not None:
            date, time = str(datetime.today()).split('.')[0].split(' ')
            exps_dir = Path(exp_basedir) / exp_name / case / Path(conf_path).stem
            try:
                prev_exps = sorted(exps_dir.iterdir())
                cur_dir = prev_exps[-1].name
                is_continue = True
            except FileNotFoundError:
                print("No previous experiment in directory. Will initialize a new training instead...")
                cur_dir = date + '_' + time
                is_continue = False
            self.base_exp_dir =  exps_dir / cur_dir        
        else:
            self.base_exp_dir = self.conf['general']['base_exp_dir']
                 
        self.img_size = self.hair_conf['render']['image_size']
        
        os.makedirs(self.base_exp_dir, exist_ok=True)                    
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'orients'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'hair_primitives'), exist_ok=True)
        orients2D_dir = os.path.join(self.base_exp_dir, '2d_orients')
        os.makedirs(orients2D_dir, exist_ok=True)
        
        self.dataset = NPHMHaircutDataset(
            os.path.join(self.conf["dataset"]["data_dir"], "camera_params.json"), self.conf["dataset"])
        # if scene_type == 'h3ds':
        #     self.dataset = H3DSDatasetTorch(self.batch_size, self.conf['dataset'])
        # else:
        #     self.dataset = MonocularDatasetTorch(self.batch_size, self.conf['dataset'])
        
        self.iter_step = 0
        self.writer = None
        set_seed(42)
        self.hair_primitives_trainer = StrandsTrainer(
            self.hair_conf, run_model= lambda model, x: model(x), device=self.device, save_dir=self.base_exp_dir
        )
        if '040000' in self.hair_conf['general']['sdf_path']:
            print('Using LevelSetUDF for the volume loss')
            self.hair_network = self.get_levelset_udf()
        elif '009999' in self.hair_conf['general']['sdf_path']:
            print('Using NSH network for the volume loss')
            self.hair_network = self.get_nsh()
        else:
            raise ValueError('Unknown hair network')

        # Upload strand-based geometry         
        if train_conf['pretrain_strands_path']: 
            print('Upload strands!')
            self.hair_primitives_trainer.load_weights(train_conf['pretrain_hair_path'])
        
        # Load checkpoint (resume) only if the existing experiment dir actually has one.
        latest_model_name = None
        if is_continue:
            model_list = sorted(
                f for f in os.listdir(os.path.join(self.base_exp_dir, 'hair_primitives'))
                if f.endswith('.pth'))
            mesh_list = sorted(os.listdir(os.path.join(self.base_exp_dir, 'meshes')))
            if model_list and mesh_list:
                latest_model_name = model_list[-1]
                latest_iter = int(mesh_list[-1][0:8])
                assert latest_iter <= self.end_iter
            else:
                print('No saved checkpoint in the existing experiment dir; starting fresh.')
            # if checkpoint_name is not None and checkpoint_name + ".pth" in model_list:
            #     latest_model_name = checkpoint_name + ".pth"
            # else:
            #     latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info(f'Find checkpoint: {latest_model_name} with latest iter {latest_iter}')
            self.hair_primitives_trainer.load_weights(os.path.join(self.base_exp_dir, 'hair_primitives', latest_model_name))
            self.iter_step = latest_iter

        # Backup codes and configs for debug
        self.file_backup()

        # Orientation loss visualization
        self.cmap = mpl.colormaps['viridis']

        # torch profiler
        default_profiler_args = {
            "activities": [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "schedule": torch.profiler.schedule(
                wait=5, warmup=5, active=5, repeat=1
            ),
            "record_shapes": True,  # Record Shapes
            "profile_memory": True,  # Track memory usage
            "with_stack": True,  # Capture stack traces
        }
        self.profiler = torch.profiler.profile(**default_profiler_args)
    
    def get_hair_deepsdf(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = DeepSDFDecoder(256)
        model.load_state_dict(torch.load('./implicit-hair-data/data/nphm/039/model_best.ckpt', map_location='cpu'))
        model.eval()
        model.to(self.device)
        return model
    
    def get_cap_udf(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = CAPUDFNetwork(**self.hair_conf['udf_network'])
        state_dict = torch.load('./implicit-hair-data/data/nphm/039/ckpt_080000.pth', map_location='cpu')
        model.load_state_dict(state_dict['udf_network_fine'])
        model.eval()
        model.to(self.device)
        return model
    
    def get_nsh(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = NSH(**self.hair_conf['nsh_network'])
        state_dict = torch.load(self.hair_conf['general']['sdf_path'], map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model.decoder
    
    def get_levelset_udf(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = LevelSetUDFNetwork(**self.hair_conf['udf_network'])
        state_dict = torch.load(self.hair_conf['general']['sdf_path'], map_location='cpu')
        model.load_state_dict(state_dict['udf_network_fine'])
        model.eval()
        model.to(self.device)
        return model
    
    def train(self):
        res_step = self.end_iter - self.iter_step
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        image_perm = self.get_image_perm()
        losses = {}
        # self.profiler.start()
        
        # bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        # bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        
        scaler = torch.cuda.amp.GradScaler()
        for _ in tqdm(range(res_step)):            
            # cur_img = image_perm[self.iter_step % len(image_perm)]
            # raster_dict = self.dataset[cur_img]
            # # _, cam_intr, cam_pose = self.dataset.gen_random_rays_at(cur_img, self.batch_size)
            # return_dict = self.dataset[cur_img]
            # cam_extr = torch.linalg.inv(return_dict['pose'].detach())

            # orig_data_size = return_dict['H'] # H shape, suppose square images
            
            # raster_dict['iter'] = self.iter_step
            # raster_dict['cam_extr'] = cam_extr
            
            # if orig_data_size is None:
            #     # raster_dict['cam_intr'] = cam_intr
            #     # raster_dict['gt_silh'] = self.dataset.hair_masks[cur_img].permute(2, 0, 1).cuda()
            #     # raster_dict['gt_rgb'] = self.dataset.images[cur_img].permute(2, 0, 1).cuda()
            #     raster_dict['cam_intr'] = return_dict['intrinsic']
            #     raster_dict['gt_silh'] = return_dict['hair_mask']
            #     raster_dict['gt_rgb'] = return_dict['image']
            # else:
            #     # need to change cameras intrinsic as we render in resolution 512x512
            #     scale_factor = orig_data_size / self.img_size
            #     # raster_dict['cam_intr'] = scale_mat(deepcopy(cam_intr), scale_factor)
            #     # raster_dict['gt_silh'] = F.interpolate(self.dataset.hair_masks[cur_img].permute(2, 0, 1)[None], size=self.img_size, mode='bilinear')[0].cuda()
            #     # raster_dict['gt_rgb'] = F.interpolate(self.dataset.images[cur_img].permute(2, 0, 1)[None], size=self.img_size, mode='bilinear')[0].cuda()
            #     raster_dict['cam_intr'] = scale_mat(deepcopy(return_dict['intrinsic']), scale_factor)
            #     raster_dict['gt_silh'] = F.interpolate(return_dict['hair_mask'][None], size=self.img_size, mode='bilinear')[0].cuda()
            #     raster_dict['gt_rgb'] = F.interpolate(return_dict['image'][None], size=self.img_size, mode='bilinear')[0].cuda()


            # # raster_dict['visual_gt_orients'] = self.dataset.orient_at(cur_img, resolution_level=1)
            # raster_dict['visual_gt_orients'] = return_dict['orient_image']
            
            # with self.profiler:
            losses.update({
                'hair_' + str(key): val for key, val in self.hair_primitives_trainer.train_step(
                        model=self.hair_network,
                        it=self.iter_step,
                        raster_dict=None,
                        scaler=scaler,
                    ).items()
                })

            self.iter_step += 1
            # self.profiler.step()

            if losses is not None:
                for k, v in losses.items():
                    self.writer.add_scalar(f'Loss/{k}', v, self.iter_step)

            if self.iter_step % self.report_freq == 0 or self.iter_step == 1:
                self.save_strands_pointcloud()
            
            # if self.iter_step % len(image_perm) == 0:
            #     image_perm = self.get_image_perm()
            
            if self.iter_step % self.report_freq == 0:
                self.hair_primitives_trainer.save_weights(os.path.join(self.base_exp_dir, 'hair_primitives', 'ckpt_latest.pth'))
        # self.profiler.stop()
        # self.save_profiler(os.path.join(self.base_exp_dir, 'traces.json'))
    
    def save_strands_pointcloud(self):
        if self.hair_primitives_trainer:
            strands_origins = self.hair_primitives_trainer.strands_origins.reshape(-1, 100, 3)
            

            cols = torch.cat((torch.rand(strands_origins.shape[0], 3).unsqueeze(1).repeat(1, 100, 1), torch.ones(strands_origins.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()
            trimesh.PointCloud(strands_origins.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_strands_points.ply'.format(self.iter_step)))

            if self.iter_step > self.hair_conf["general"]["starting_rendering_iter"]:
                if self.hair_primitives_trainer.loss_orient is not None:
                    orient_colors = np.reshape(np.concatenate([self.cmap(self.hair_primitives_trainer.loss_orient), np.ones([1900, 1, 4])], axis=1), (-1, 4))
                    trimesh.PointCloud(strands_origins.reshape(-1, 3).detach().cpu(), colors=orient_colors).export(os.path.join(self.base_exp_dir, 'orients', '{:0>8d}_strands_points.ply'.format(self.iter_step)))

    def save_profiler(self, export_path: str) -> None:
        """
           Use 'chrome://tracing/' for small traces < 50MB or 'ui.perfetto.dev' for > 50 MB.
           Log the 10 most expensive cuda operations.
        Args:
            export_path (str): Save the profiler to the specified path if it is active.
        """
        """Save the profiler to the specified patg if it is active.
        """
        print(
            self.profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            )
        )
        self.profiler.export_chrome_trace(export_path)
    
    def get_image_perm(self):
        return torch.randperm(len(self.dataset))

    def file_backup(self):
        dir_lis = self.conf['general']['recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.yaml'))
        copyfile(self.hair_conf_path, os.path.join(self.base_exp_dir, 'recording', 'hair_config.yaml'))

if __name__ == '__main__':
    print('Hello Wooden')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--scene_type', type=str, default='')
    parser.add_argument('--hair_conf', type=str, default=None, help='Use hair primitives config')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to continue training')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--shrink', type=str, required=True)
    parser.add_argument('--use_3do_2do', type=str, required=True)
    parser.add_argument('--use_old_dif_prior', type=str, required=True)
    parser.add_argument('--is_blobs', type=str, required=True, default="False")
    parser.add_argument('--diffuse_iter', type=int, default=10000)
    parser.add_argument('--n_diffuse_steps', type=int, default=4)
    parser.add_argument('--exp_basedir', type=str, default='./exps_second_stage')
    parser.add_argument('--data_root', type=str, default='DATA_ROOT',
                        help='Replaces the DATA_ROOT token in the configs (root of the preprocessed scans)')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Optional cap on training iterations (e.g. for a smoke run)')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    shrink = True if args.shrink.lower() in ['yes', 'true', 't', 'y', '1'] else False
    use_3do_2do = True if args.use_3do_2do.lower() in ['yes', 'true', 't', 'y', '1'] else False
    use_old_dif_prior = True if args.use_old_dif_prior.lower() in ['yes', 'true', 't', 'y', '1'] else False
    is_blobs = True if args.is_blobs.lower() in ['yes', 'true', 't', 'y', '1'] else False

    runner = Runner(
        args.conf,
        args.case,
        args.scene_type,
        shrink=shrink,
        use_3do_2do=use_3do_2do,
        use_old_dif_prior=use_old_dif_prior,
        is_blobs=is_blobs,
        diffuse_iter=args.diffuse_iter,
        n_diffuse_steps=args.n_diffuse_steps,
        hair_conf_path=args.hair_conf,
        checkpoint_name=args.checkpoint,
        exp_name=args.exp_name,
        exp_basedir=args.exp_basedir,
        data_root=args.data_root,
        max_iter=args.max_iter,
    )

    runner.train()
