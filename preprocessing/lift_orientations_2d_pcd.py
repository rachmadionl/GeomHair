"""Lift 2D hair orientations to a 3D oriented point cloud.

For each view, renders the hair mesh's surface normals + depth (PyTorch3D), back-projects
the 2D line pixels to 3D, and projects each pixel's 2D orientation onto the local surface
tangent plane. Writes ``lifted_orients.ply`` (points + per-point orientation normals), plus
an optional interactive ``orients_plotter.html``.
"""

import argparse
import os
import sys
sys.path.append('./')

import numpy as np
import torch
import torch.nn as nn
import yaml
try:
    import pyvista as pv
except Exception:  # pyvista is only used for optional debug visualization
    pv = None
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    HardFlatShader,
    DirectionalLights,
    TexturesVertex,
)
from pytorch3d.io import IO
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

from src.models.dataset_torch import NPHMHaircutDataset

# Optional offscreen backend for the pyvista HTML preview; not needed for lifting.
if pv is not None and hasattr(pv, "start_xvfb"):
    try:
        pv.start_xvfb()
    except Exception:
        pass


class MeshRendererWithDepth(nn.Module):
    """Mesh renderer that also returns the rasterizer's z-buffer (depth)."""

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world: Meshes, **kwargs):
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class MultiViewLineOrientationLifter:
    def __init__(self, mesh: Meshes, image_size: int = 256):
        self.mesh = mesh
        self.device = mesh.device
        self.vertex_normals = mesh.verts_normals_packed()
        self.image_size = image_size

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            clip_barycentric_coords=True,
        )

        # Render surface normals as vertex colors (mapped from [-1,1] to [0,1]).
        normal_colors = (self.vertex_normals + 1) / 2
        self.mesh.textures = TexturesVertex(normal_colors.unsqueeze(0))

        lights = DirectionalLights(
            device=self.device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
        )
        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=HardFlatShader(device=self.device, lights=lights, blend_params=None),
        )

    def create_camera(self, extrinsic: torch.Tensor, intrinsic: torch.Tensor):
        return cameras_from_opencv_projection(
            camera_matrix=intrinsic.unsqueeze(0),
            R=extrinsic[:3, :3].unsqueeze(0),
            tvec=extrinsic[:3, 3].unsqueeze(0),
            image_size=torch.tensor([self.image_size, self.image_size]).unsqueeze(0),
        ).to(self.device)

    def lift_to_point_cloud(self, dataset: torch.utils.data.Dataset):
        """Return a Pointclouds with 3D line points and tangent-projected orientations."""
        all_points = []
        all_orientations = []
        for view_idx in range(len(dataset)):
            data = dataset[view_idx]
            lines_2d = (data['strands_line'] > 0).to(torch.float32)
            camera = self.create_camera(data['cam_extr'], data['cam_intr'])

            image, zbuf = self.renderer(meshes_world=self.mesh, cameras=camera)
            normal_map = image[0, ..., :3]  # (H, W, 3)
            depth_map = zbuf[0, :, :, 0]     # (H, W)

            world_points = get_rgbd_point_cloud(
                camera,
                image_rgb=torch.ones_like(depth_map.unsqueeze(0).unsqueeze(0)).expand(-1, 3, -1, -1),
                depth_map=depth_map.unsqueeze(0).unsqueeze(0),
                mask=lines_2d,
                mask_thr=0.1,
            ).points_packed()

            valid_mask = lines_2d[0, 0]
            orient = data['orient'][0, 0]  # already in radians
            if not valid_mask.any():
                continue

            normal_map = normal_map * 2 - 1  # back to [-1, 1]
            y_coords, x_coords = torch.where(valid_mask)

            # 2D orientation angle -> screen-space unit vector.
            vectors = torch.stack([torch.cos(orient), torch.sin(orient)], dim=-1)  # (H, W, 2)
            screen_vectors = vectors[y_coords, x_coords]      # (N, 2)
            surface_normals = normal_map[y_coords, x_coords]  # (N, 3)
            depths = depth_map[y_coords, x_coords]

            # Lift screen vectors to 3D (z=1) and transform view -> world.
            view_vectors_3d = torch.cat(
                [screen_vectors, torch.ones_like(screen_vectors[:, :1])], dim=1)  # (N, 3)
            view_to_world = camera.get_world_to_view_transform().inverse()
            world_vectors = view_to_world.transform_points(view_vectors_3d.float())  # (N, 3)

            # Project orientations onto the surface tangent plane and normalize.
            dot_products = torch.sum(world_vectors * surface_normals, dim=1, keepdim=True)
            projections = world_vectors - dot_products * surface_normals
            norms = torch.norm(projections, dim=1, keepdim=True)
            projections = torch.where(norms > 1e-6, projections / norms, torch.zeros_like(projections))

            all_points.append(world_points)
            all_orientations.append(projections[depths > 0])

        world_points = torch.concatenate(all_points)
        orientations = torch.cat(all_orientations, dim=0)
        return Pointclouds(points=[world_points], normals=[orientations])


def process_lines_and_orientations(mesh: Meshes, dataset: torch.utils.data.Dataset,
                                   image_size: int = 256):
    """Build the oriented point cloud for a mesh + multi-view dataset."""
    lifter = MultiViewLineOrientationLifter(mesh, image_size)
    return lifter.lift_to_point_cloud(dataset)


def visualize_point_cloud_pyvista(points, vectors, subsample_size=None, vector_scale=0.1,
                                  point_size=5, background_color='white', window_size=(1024, 768)):
    """Build a PyVista plotter showing the point cloud and its orientation arrows."""
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(vectors):
        vectors = vectors.cpu().numpy()

    if subsample_size is not None and len(points) > subsample_size:
        indices = np.random.choice(len(points), subsample_size, replace=False)
        points = points[indices]
        vectors = vectors[indices]

    vectors = vectors / (np.linalg.norm(vectors, axis=1)[:, np.newaxis] + 1e-8) * vector_scale

    cloud = pv.PolyData(points)
    cloud['vectors'] = vectors
    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background_color)
    plotter.add_mesh(cloud, render_points_as_spheres=False, point_size=point_size, color='blue')
    plotter.add_arrows(cloud.points, cloud['vectors'], color='red', mag=vector_scale)
    plotter.add_axes()
    return plotter


def main():
    parser = argparse.ArgumentParser(description='Process hair mesh and visualize orientations')
    parser.add_argument('--case', type=int, required=True, help='Case number for processing (e.g., 17)')
    parser.add_argument('--config', type=str, default='configs/monocular/neural_strands.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dir', type=str, default='./implicit-hair-data/data/nphm',
                        help='Base folder containing scan data')
    parser.add_argument('--image-size', type=int, default=512, help='Image size for processing')
    parser.add_argument('--subsample-size', type=int, default=15000,
                        help='Number of points to subsample for visualization')
    parser.add_argument('--vector-scale', type=float, default=0.12,
                        help='Scale factor for vector visualization')
    parser.add_argument('--point-size', type=int, default=5, help='Point size for visualization')
    args = parser.parse_args()

    sub = '001' if args.case == 478 else '000'
    scan_folder = os.path.join(args.dir, f'{args.case:03d}', sub)
    mesh_path = os.path.join(scan_folder, 'mesh.obj')
    save_pcd_path = os.path.join(scan_folder, 'lifted_orients.ply')

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    conf['dataset']['data_dir'] = scan_folder

    dataset = NPHMHaircutDataset(os.path.join(scan_folder, 'camera_params.json'), conf['dataset'])
    mesh = IO().load_mesh(mesh_path, device='cuda')
    pointcloud = process_lines_and_orientations(mesh, dataset, image_size=args.image_size)
    IO().save_pointcloud(pointcloud, save_pcd_path)

    points = pointcloud.points_packed().numpy(force=True)
    orients = pointcloud.normals_packed().numpy(force=True)

    # Optional interactive HTML preview; requires pyvista + a rendering backend (trame).
    # Not needed for the lifted_orients.ply output, so failures here are non-fatal.
    if pv is not None:
        try:
            print(f"Visualizing {len(points)} points...")
            plotter = visualize_point_cloud_pyvista(
                points, orients,
                subsample_size=args.subsample_size,
                vector_scale=args.vector_scale,
                point_size=args.point_size,
            )
            plotter.background_color = [180, 220, 180]
            plotter.export_html(save_pcd_path.replace('lifted_orients.ply', 'orients_plotter.html'))
        except Exception as e:
            print(f"[lift] skipping optional HTML visualization: {e}")


if __name__ == '__main__':
    main()
