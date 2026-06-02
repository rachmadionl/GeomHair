import os

import cv2
import mahotas as mh
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.io import IO
from skimage.draw import polygon

from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
from src.hair_networks.strands_renderer import Renderer
from src.hair_networks.point_strands_renderer import HairStrandProcessor
from src.losses.sdf_chamfer import SdfChamfer
from src.losses.one_way_chamfer import chamfer_distance
from src.utils.util import freeze_gradients


class UniformPointSampler(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, points, normals=None):
        """
        Uniformly sample points and optionally normals from input tensors.
        
        :param points: (B, N, 3) tensor of input points
        :param normals: (B, N, 3) tensor of input normals (optional)
        :return: Tuple of sampled points and normals (if provided)
        """
        B, N, _ = points.shape
        
        # Generate random indices for each item in the batch
        indices = torch.argsort(torch.rand(B, N, device=points.device), dim=1)[:, :self.num_samples]
        
        # Gather sampled points
        sampled_points = torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        
        if normals is not None:
            # Gather sampled normals
            sampled_normals = torch.gather(normals, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
            return sampled_points, sampled_normals
        
        return sampled_points


class StrandsTrainer:
    def __init__(self, config, run_model=None, device=None, save_dir=None) -> None:
        
        self.device = device
        self.config = config

        params_to_train = []
        self.strands = OptimizableTexturedStrands(**config['textured_strands'], diffusion_cfg=config['diffusion_prior']).to(self.device)
        params_to_train += list(self.strands.parameters())

        # self.strands_render = config['render']['use_render']
        self.logging_freq = config['render']['logging_freq']
        self.orients2D = os.path.join(save_dir, '2d_orients')
        if config['render']['use_render']:
            print('Create rasterizer!')
            # self.strands_render = Renderer(config['render'], save_dir=save_dir).to(self.device)
            # params_to_train += list(self.strands_render.parameters())
            self.strands_render = HairStrandProcessor()
        else:
            self.strands_render = None
            print('No Rendering Used!')

        self.starting_rendering_iter = config['general']['starting_rendering_iter']  

        self.optimizer = torch.optim.Adam(params_to_train, config['general']['lr'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['general']['milestones'], gamma=config['general']['gamma'])
        self.loss_factors = config['loss_factors']

        self.sdfchamfer = SdfChamfer(**config['sdf_chamfer'])
        self.run_model = run_model
        self.strands_origins = None
        self.orientation = IO().load_pointcloud(config['general']['orientation_path'], device=self.device)
        self.lifted_orientation = IO().load_pointcloud(config['general']['lifted_orientation_path'], device=self.device)
        self.blob_idx = config['general'].get('orientation_mask', None)
        if self.blob_idx:
            self.blob_idx = torch.tensor(np.load(self.blob_idx)).to(self.device)
        self.loss_orient = None
        self.filtered_idx = None
        print(f"alpha for orientation: {self.config['render']['alpha']}")
    
    def get_latent_codes(self):
        """
        :return: latent codes which were optimized during training
        """
        latent_codes = torch.nn.Embedding.from_pretrained(
            torch.load('./implicit-hair-data/data/nphm/039/latent_best.ckpt', map_location='cpu')['weight']
        )
        latent_codes.to(self.device)
        return latent_codes

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.strands.load_state_dict(state_dict['strands'])
        if isinstance(self.strands_render, Renderer):
            self.strands_render.load_state_dict(state_dict['strands_render'])


    def save_weights(self, path):
        state_dict = {
            'strands': self.strands.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if isinstance(self.strands_render, Renderer):
            state_dict['strands_render'] = self.strands_render.state_dict()
        torch.save(state_dict, path)

    def sort_by_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(tensor, p=2, dim=1)
        idx = torch.argsort(norm, stable=True)
        return tensor[idx]

    def train_step(self, model=None, it=0, raster_dict=None, scaler=None):
        losses = {}

        with torch.cuda.amp.autocast():
            self.strands_origins, z_geom, z_app, uv_idx, dif_dict = self.strands(it=it)
            num_strands, strand_len = self.strands_origins.shape[0], self.strands_origins.shape[1]
            
            with freeze_gradients(model):
                out = self.run_model(model, self.strands_origins.view(-1, 3))

            # Calculate origin loss
            sdf = out.view(-1, strand_len)
            sdf_inside = torch.relu(sdf[:, 1:])
            losses['volume'] = F.mse_loss(sdf_inside, torch.zeros_like(sdf_inside))

            # Calculate orientation loss
            prim_dir = self.strands_origins[:, 1:] - self.strands_origins[:, :-1] # [N_strands, strand_len-1, 3]

            # Calculate orientations only near the visible outer surface
            if (self.filtered_idx is None) or (it % 1000 == 0):
                dist, closest_face_idx, _ = self.sdfchamfer.points2face(
                    self.sdfchamfer.mesh_outer_hair_remeshed, self.strands_origins[:, :-1, :].reshape(-1, 3)
                )
                filtered_idx = torch.nonzero(dist <= 1e-4).reshape(-1)
                self.filtered_idx = filtered_idx

                if self.sdfchamfer.blob_face_idx is not None:
                    blobs_mask = torch.isin(closest_face_idx, self.sdfchamfer.blob_face_idx)
                    filtered_mask = torch.zeros_like(blobs_mask, dtype=torch.bool, device=self.device)
                    filtered_mask[filtered_idx] = True
                    blobs_mask_near_surface = (filtered_mask * blobs_mask).reshape(num_strands, strand_len - 1)
                    blobs_mask_near_surface = torch.max(blobs_mask_near_surface, dim=-1)[0].numpy(force=True)
                    blobs_mask_near_surface = self.create_texture_mask(
                        blobs_mask_near_surface, uv_idx, it
                    )
                    self.blobs_mask_near_surface = blobs_mask_near_surface

            _, loss_orient = chamfer_distance(
                x = self.strands_origins[:, :-1, :].reshape(-1, 3)[self.filtered_idx].unsqueeze(0),
                x_normals = prim_dir.reshape(-1, 3)[self.filtered_idx].unsqueeze(0),
                y = self.orientation.points_packed().to(prim_dir.dtype).unsqueeze(0), 
                y_normals = self.orientation.normals_packed().to(prim_dir.dtype).unsqueeze(0),
                batch_reduction=None,
                orient_mask=self.blob_idx,
            )
            losses['orient'] = loss_orient.mean(dim=1)

            # Calculate chamfer on visible outer surface
            if self.sdfchamfer is not None:
                # losses['chamfer'] = loss_chamfer
                losses['chamfer'] = self.sdfchamfer.calc_chamfer(self.strands_origins[:, 1:, :].reshape(-1, 3)[None])

            # Calculate photometric losses
            if self.strands_render and it > self.starting_rendering_iter:
                _, lifted_orient_loss = chamfer_distance(
                    x = self.strands_origins[:, :-1, :].reshape(-1, 3).unsqueeze(0),
                    x_normals = prim_dir.reshape(-1, 3).unsqueeze(0),
                    y = self.lifted_orientation.points_packed().to(prim_dir.dtype).unsqueeze(0), 
                    y_normals = self.lifted_orientation.normals_packed().to(prim_dir.dtype).unsqueeze(0),
                    batch_reduction=None,
                    orient_mask=self.blob_idx,
                )
                alpha = self.config["render"]["alpha"]
                losses['orient'] = (1 - alpha) * losses['orient'] + alpha * lifted_orient_loss.mean(dim=1)
            
            # Calculate diffusion loss
            if len(dif_dict) > 0:
                if self.sdfchamfer.blob_face_idx is not None:
                    losses['L_diff'] = (dif_dict['L_diff'] * self.blobs_mask_near_surface).mean()
                else:
                    losses['L_diff'] = dif_dict['L_diff'].mean()

        total_loss = sum(loss * float(self.loss_factors[name]) for name, loss in losses.items())
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # for param in self.optimizer.param_groups[0]['params']:
        #     if param.grad is not None and param.grad.isnan().any():
        #         self.optimizer.zero_grad()
        #         print('NaN during backprop was found, skipping iteration...')
        #         return losses
        
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        return losses

    def create_texture_mask(self, strand_mask, uv_idx,  it):
        """
        Create texture-space mask using UV coordinates of problematic strands
        
        Args:
            strand_mask: (N_strands,) boolean tensor marking problematic strands
            uvs: (N_strands, 2) UV coordinates for strands
            blur_sigma: float, standard deviation for Gaussian blur
        """
        mask = np.zeros(
            (self.strands.texture_size, self.strands.texture_size), 
        )
        
        # Convert UVs to pixel coordinates.
        uv_blobs = self.strands.uvs[uv_idx][strand_mask].numpy(force=True)
        uv_pixels = ((uv_blobs + 1) * ((self.strands.texture_size - 1) / 2)).astype(np.uint8)
        rr, cc = polygon(uv_pixels[:,0], uv_pixels[:,1], mask.shape)
        mask[rr, cc] = 255

        # dilation
        mask = mask.astype(np.uint8)
        os.makedirs('./strands_diff_mask', exist_ok=True)
        cv2.imwrite(os.path.join('strands_diff_mask', f'strands_mask_{it}_before.png'), mask)
        mask = mh.dilate(mask, np.ones((7, 7)))
        cv2.imwrite(os.path.join('strands_diff_mask', f'strands_mask_{it}_after.png'), mask)
        mask = mask / mask.max()

        # interpolate to the denoised texture shape.
        mask = torch.tensor(mask[None, None], device=self.device, dtype=torch.float32)
        mask = F.interpolate(
            mask, 
            size=(self.strands.diffusion_input, self.strands.diffusion_input),
            mode='bilinear',
            align_corners=False
        )

        # range [1, 2], where 2 is the blobs.
        mask = mask + 1
        return mask
