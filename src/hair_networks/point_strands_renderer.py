import torch
import torch.nn.functional as F
import math
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizer,
    PointsRasterizationSettings
)
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from src.utils.geometry import soft_interpolate, hard_interpolate

def bilinear_interpolate(valid_pixels, x):
    """
    Bilinear interpolation using grid_sample for 1D indices.
    
    Args:
        valid_pixels: (N,) tensor of indices
        x: (M, C) tensor of features to sample from
            M: number of points
            C: number of features
    
    Returns:
        interpolated: (N, C) tensor of interpolated features
    """
    # Reshape x to (1, C, M, 1) for grid_sample
    x = x.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # (1, C, M, 1)
    
    # Normalize indices to [-1, 1] range
    norm_pixels = (valid_pixels.float() / (x.shape[2] - 1)) * 2 - 1  # (N,)
    
    # Create grid for sampling
    grid = torch.stack([
        torch.zeros_like(norm_pixels),  # x coordinates (all zeros)
        norm_pixels                     # y coordinates
    ], dim=-1)  # (N, 2)
    
    # Reshape grid to (B, H, W, 2) format expected by grid_sample
    grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
    
    # Perform interpolation
    sampled = F.grid_sample(
        x,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # (1, C, N, 1)
    
    # Reshape to match expected output format
    sampled = sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (N, C)
    
    return sampled

class HairStrandProcessor:
    """
    Combined pipeline for processing hair strands with PyTorch3D camera support and bilinear interpolation.
    """
    def __init__(self, image_size=512, points_per_pixel=8, radius=0.01):
        self.image_size = image_size
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=points_per_pixel
        )
        self.rasterizer = PointsRasterizer(raster_settings=raster_settings)
        
    def process_strands(
            self, 
            org_3d, 
            cam_intr, 
            cam_extr, 
            gt_orient_map=None, 
            orient_mask=None, 
            orient_conf=None
        ):
        # Get the 3D Orient
        orient_3d = torch.zeros_like(org_3d)
        orient_3d[:, :orient_3d.shape[1] - 1] = (org_3d[:, 1:] - org_3d[:, :-1])
        orient_3d[:, orient_3d.shape[1] - 1: ] = orient_3d[:, orient_3d.shape[1] - 2: orient_3d.shape[1] - 1]
        orient_3d = orient_3d.reshape(-1, 3)

        # Create PyTorch3D cameras from OpenCV parameters
        cameras = cameras_from_opencv_projection(
            camera_matrix=cam_intr.unsqueeze(0),
            R=cam_extr[:3, :3].unsqueeze(0),
            tvec=cam_extr[:3, 3].unsqueeze(0),
            image_size=torch.tensor([self.image_size, self.image_size]).unsqueeze(0)
        ).to(org_3d.device)
        
        self.rasterizer.cameras = cameras
        
        # Rasterize points
        point_cloud = Pointclouds(points=[org_3d.view(-1, 3)])
        fragments = self.rasterizer(point_cloud)
        
        # Get rasterization indices and create valid mask
        raster_idxs = fragments.idx[0, :, :, 0]  # (H, W, K)
        raster_idxs[raster_idxs == -1] = 0  # Replace -1 with 0 temporarily
        valid_pixels = raster_idxs[raster_idxs != 0]
        
        # Perform bilinear interpolation for positions and orientations
        strands_origins = soft_interpolate(
            valid_pixels.cuda(), 
            org_3d.view(-1, 3)
        )
        
        # orient_interp = hard_interpolate(
        #     valid_pixels.cuda(),
        #     orient_3d
        # )

        orient_interp = bilinear_interpolate(
            valid_pixels.cuda(),
            orient_3d
        )

        # Project interpolated orientations to camera space
        ret_value = self.project_orient_to_camera(
            orient_interp.unsqueeze(1),
            strands_origins,
            cam_intr,
            cam_extr
        )
        orient_angle, org_2d_cam = ret_value
        
        # Create output orientation map
        plane_orients = torch.zeros(
            self.image_size, 
            self.image_size, 
            1, 
            device=orient_3d.device
        )
        valid_mask = raster_idxs != 0
        plane_orients[valid_mask, :] = orient_angle
        
        # Sample ground truth if provided
        if gt_orient_map is not None and orient_mask is not None and orient_conf is not None:
            gt_results = self.sample_gt_orientations(
                org_2d_cam,
                gt_orient_map, 
                orient_mask, 
                orient_conf
            )
            gt_orient_angle, orient_mask_sampled, orient_conf_sampled = gt_results
        else:
            gt_orient_angle = None
            orient_mask_sampled = None
            orient_conf_sampled = None
            
        return {
            'fragments': fragments,
            'pred_orients': plane_orients.permute(2, 0, 1),
            'raster_indices': raster_idxs,
            'valid_mask': valid_mask,
            'orient_angle': orient_angle,
            'projected_points': org_2d_cam,
            'gt_orient_angle': gt_orient_angle,
            'orient_mask_sampled': orient_mask_sampled,
            'orient_conf_sampled': orient_conf_sampled,
            'interpolated_origins': strands_origins,
            'interpolated_orients': orient_interp
        }
    
    @staticmethod
    def project_orient_to_camera(orient_3d, org_3d, cam_intr, cam_extr):
        """Project 3D orientations to camera space."""
        reshape_out = len(orient_3d.shape) == 3
        if reshape_out:
            b, n, _ = orient_3d.shape
        org_3d = org_3d.view(-1, 3)
        orient_3d = orient_3d.view(-1, 3)

        dst_3d = org_3d + orient_3d

        dummy_ones = torch.ones(dst_3d.shape[0], 1, device=dst_3d.device, dtype=dst_3d.dtype)
        org_3d = torch.cat([org_3d, dummy_ones], dim=1)[..., None]
        dst_3d = torch.cat([dst_3d, dummy_ones], dim=1)[..., None]

        if cam_extr.dim() == 3:
            org_3d_cam = torch.matmul(cam_intr, torch.matmul(cam_extr, org_3d))[:, :3, 0]
            dst_3d_cam = torch.matmul(cam_intr, torch.matmul(cam_extr, dst_3d))[:, :3, 0]
        else:
            org_3d_cam = torch.matmul(cam_intr[None], torch.matmul(cam_extr[None], org_3d))[:, :3, 0]
            dst_3d_cam = torch.matmul(cam_intr[None], torch.matmul(cam_extr[None], dst_3d))[:, :3, 0]

        org_2d_cam = org_3d_cam[:, :2] / (org_3d_cam[:, [2]] + 1e-5)
        dst_2d_cam = dst_3d_cam[:, :2] / (dst_3d_cam[:, [2]] + 1e-5)

        orient_2d_cam = dst_2d_cam - org_2d_cam
        orient_2d_cam = orient_2d_cam / (orient_2d_cam.norm(dim=-1, keepdim=True) + 1e-5)
        if reshape_out:
            orient_2d_cam = orient_2d_cam.view(b, n, 2)

        orient_sin = orient_2d_cam[..., 0]
        to_mirror = torch.ones_like(orient_sin)
        to_mirror[orient_sin < 0] *= -1
        orient_cos = orient_2d_cam[..., 1] * to_mirror

        sampled_orient_angle = torch.acos(orient_cos.clamp(-1 + 1e-5, 1 - 1e-5))
        sampled_orient_angle = (math.pi / 2 - sampled_orient_angle) % (2 * math.pi)
        if reshape_out:
            orient_angle = sampled_orient_angle[:, 0]
        else:
            orient_angle = sampled_orient_angle
        orient_angle = orient_angle[:, None]

        return orient_angle, org_2d_cam

    @staticmethod
    def sample_gt_orientations(orient_2d_cam, gt_orient_map, orient_mask, orient_conf):
        """Sample ground truth orientations at projected 2D positions."""
        height, width = gt_orient_map.shape[2:]

        norm_coords = orient_2d_cam.clone()
        norm_coords[:, 0] = (norm_coords[:, 0] / (width - 1)) * 2 - 1
        norm_coords[:, 1] = (norm_coords[:, 1] / (height - 1)) * 2 - 1
        norm_coords = torch.clamp(norm_coords, -1.0, 1.0)
        
        grid = norm_coords.view(1, 1, -1, 2).to(gt_orient_map.dtype).to(gt_orient_map.device)
        sampled_orientations = F.grid_sample(gt_orient_map, grid, align_corners=True, mode='bilinear')
        sampled_mask = F.grid_sample(orient_mask, grid, align_corners=True, mode='bilinear')
        sampled_conf = F.grid_sample(orient_conf, grid, align_corners=True, mode='bilinear')
        
        return (
            sampled_orientations.view(-1, 1),
            sampled_mask.view(-1, 1),
            sampled_conf.view(-1, 1)
        )