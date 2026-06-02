import argparse
import torch
import os
import sys
import json
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import math
from pytorch3d.io import IO
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex,
    Materials,
)

sys.path.append(os.getcwd())

sys.path.append(os.path.join(os.getcwd(), './submodules/LLaVA'))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from src.utils.text_utils import QUESTIONS

# Newer transformers (4.33) + LLaVA's LoRA loader leave the freshly-added mm_projector
# params on `meta` (low_cpu_mem_usage / device_map="auto"), so the projector weights never
# actually load -> NaN logits at generation. Force a plain (non-meta) load here; we move the
# merged model to the GPU after load_pretrained_model returns.
_orig_llava_from_pretrained = LlavaLlamaForCausalLM.from_pretrained.__func__


def _llava_from_pretrained_no_meta(cls, *args, **kwargs):
    kwargs['low_cpu_mem_usage'] = False
    kwargs.pop('device_map', None)
    return _orig_llava_from_pretrained(cls, *args, **kwargs)


LlavaLlamaForCausalLM.from_pretrained = classmethod(_llava_from_pretrained_no_meta)

HEIGHT = 1024
WIDTH = 1024

def calculate_optimal_distance(mesh, diagonal_size_ref: float = 1.4356) -> float:
    bbox = mesh.get_bounding_boxes()
    bbox_size = bbox[0, :, 1] - bbox[0, :, 0]
    diagonal_size = torch.norm(bbox_size)
    factor = diagonal_size / diagonal_size_ref
    return factor * (1.0 if factor >= 1.6 else 1.1)

def make_renderer(R, T, device='cuda:0'):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)
    raster_settings = RasterizationSettings(
        image_size=(HEIGHT, WIDTH),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    lights = PointLights(device=device, location=cameras.get_camera_center())
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    materials = Materials(shininess=0).to(cameras.device)
    renderer = MeshRenderer(
        rasterizer,
        shader=HardFlatShader(device=device, cameras=cameras, lights=lights, materials=materials)
    )
    return renderer

def render_views(mesh_path, save_dir, device='cuda'):
    mesh = IO().load_mesh(mesh_path, device=device)
    verts_rgb = torch.ones_like(mesh.verts_packed())[None]
    mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))
    
    template_center = mesh.get_bounding_boxes().mean(dim=-1)
    optimal_dist = calculate_optimal_distance(mesh)
    
    # Create renderers for frontal and back views
    view_params = [
        (optimal_dist, 30, 40),    # Frontal view
        (optimal_dist, 30, 240),   # Back view
    ]
    
    os.makedirs(save_dir, exist_ok=True)
    rendered_images = {}
    
    for idx, (dist, elev, azim) in enumerate(view_params):
        R, T = look_at_view_transform(dist, elev, azim, at=template_center)
        renderer = make_renderer(R, T, device)
        
        image_torch = renderer(mesh)[0, ..., :3].cpu().data.numpy()
        image_cv2 = (image_torch * 255).astype(np.uint8)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
        
        output_path = os.path.join(save_dir, f'{idx:02d}.png')
        cv2.imwrite(output_path, image_cv2)
        rendered_images[idx] = output_path
    
    return rendered_images

def llava_eval_model(
    tokenizer,
    model,
    image_processor,
    context_len,
    questions,
    answers_file,
    rendered_images,
    conv_mode='llava_v1',
    temperature=0.1,
    pc_idx=0
):
    os.makedirs(answers_file, exist_ok=True)
    ans_file = open(os.path.join(answers_file, f'{pc_idx:05d}.txt'), "w")
    
    for idx, qs in enumerate(questions):
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        output_views = []
        for view_idx, view_name in enumerate(['frontal', 'back']):
            image = Image.open(rendered_images[view_idx])
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            
            outputs = ''.join(c for c in outputs.strip() if c != '"')
            outputs = f'From {view_name} view ' + outputs[0].lower() + outputs[1:]
            output_views.append(outputs)
        
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text_frontal": output_views[0],
            "text_back": output_views[1],
            "metadata": {}
        }) + "\n")
        ans_file.flush()
    
    ans_file.close()

def main(args):
    if args.case != 478:
        scan_folder = os.path.join(args.dir, f'{args.case:03d}/000')
    else:
        scan_folder = os.path.join(args.dir, f'{args.case:03d}/001')
    
    mesh_path = os.path.join(scan_folder, 'scan.obj')
    answers_file_path = os.path.join(scan_folder, 'dataset/answers')
    render_output_dir = os.path.join(scan_folder, 'dataset/renders')

    rendered_images = render_views(
        mesh_path=mesh_path,
        save_dir=render_output_dir,
        device=args.device
    )
    
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    model_base = 'lmsys/vicuna-7b-v1.5'
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_base, model_name)
    model = model.to(args.device)

    llava_eval_model(
        tokenizer,
        model,
        image_processor,
        context_len,
        questions=QUESTIONS,
        pc_idx=0,
        answers_file=answers_file_path,
        rendered_images=rendered_images,
        temperature=args.temperature
    )
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', required=True, type=str)
    parser.add_argument('--case', required=True, type=int)
    parser.add_argument('--model_path', default='liuhaotian/llava-v1.5-7b-lora', type=str)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    
    args = parser.parse_args()
    main(args)