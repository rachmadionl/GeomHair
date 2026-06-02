"""Render the hair mesh from multiple viewpoints.

Loads ``<dir>/<case:03d>/000/mesh.obj``, renders it from a set of (elevation, azimuth)
viewpoints with PyTorch3D, and writes the views to ``rendered_mesh/rendered_mesh_*.png``
plus the per-view camera parameters to ``camera_params.json``.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pyvista as pv
import torch
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex,
)

HEIGHT = 512
WIDTH = 512

# Vertex scale applied before rendering (also saved as registration_shrunked.ply).
SHRINK_SCALE = 0.93
# Camera distance = optimal distance * this margin, so the mesh fits in frame.
DISTANCE_MARGIN = 1.2
# Small meshes (relative diagonal below this) are pushed slightly further back.
SMALL_MESH_THRESHOLD = 1.6
SMALL_MESH_BOOST = 1.1
# Elevation (deg) that uses a doubled azimuth step (sparser ring).
SPARSE_ELEVATION = 60


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def to_dict(self):
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


def calculate_optimal_distance(mesh, diagonal_size_ref: float = 1.4356) -> float:
    """Camera distance factor so the whole mesh stays visible, from its bbox diagonal."""
    bbox = mesh.get_bounding_boxes()
    bbox_size = bbox[0, :, 1] - bbox[0, :, 0]  # max corner - min corner
    factor = torch.norm(bbox_size) / diagonal_size_ref
    if factor < SMALL_MESH_THRESHOLD:
        factor *= SMALL_MESH_BOOST
    return factor


def make_renderer(R, T, device='cuda:0', lights_location=None):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)
    raster_settings = RasterizationSettings(
        image_size=(HEIGHT, WIDTH),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    if lights_location is None:
        lights_location = cameras.get_camera_center()
    lights = PointLights(device=device, location=lights_location)

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    materials = Materials(shininess=0).to(cameras.device)
    renderer = MeshRenderer(
        rasterizer,
        shader=HardFlatShader(device=device, cameras=cameras, lights=lights, materials=materials),
    )
    R_inv = torch.linalg.inv(R)
    extrinsics_3x4 = torch.hstack([R_inv.squeeze(0), T.permute(1, 0)])
    extrinsics = torch.vstack([extrinsics_3x4, torch.Tensor([0, 0, 0, 1])])
    return renderer, extrinsics.numpy(force=True), rasterizer


def save_camera_params(filename: str, extrinsics: List, intrinsics: Intrinsics) -> None:
    data = {"extrinsics": extrinsics, "intrinsics": intrinsics.to_dict()}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Render 2D views of a 3D mesh with camera parameters')
    parser.add_argument('--case', type=int, required=True, help='Case number to process')
    parser.add_argument('--dir', type=str,
                        default='/cluster/himring/asevastopolsky/NPHMHaircut/nphm/scan',
                        help='Base folder containing scan data')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device (cuda/cpu)')
    parser.add_argument('--elevation-angles', type=float, nargs='+', default=[-15, 15, 30],
                        help='List of elevation angles for rendering')
    parser.add_argument('--azimuth-step', type=float, default=20,
                        help='Step size for azimuth angles (degrees)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512],
                        help='Output image resolution (height width)')
    parser.add_argument('--fov', type=float, default=60, help='Field of view in degrees')
    return parser.parse_args()


def main():
    args = parse_args()

    sub = '001' if args.case == 478 else '000'
    scan_folder = os.path.join(args.dir, f'{args.case:03d}', sub)
    mesh_path = os.path.join(scan_folder, 'mesh.obj')
    save_dir = os.path.join(scan_folder, 'rendered_mesh')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(args.device)
    mesh = IO().load_mesh(mesh_path, device=device)
    pv.read(mesh_path)

    scaled_vertices = mesh.verts_packed() * SHRINK_SCALE
    new_mesh = Meshes(verts=[scaled_vertices], faces=mesh.faces_list())
    IO().save_mesh(new_mesh, path=mesh_path.replace('mesh.obj', 'registration_shrunked.ply'))

    verts_rgb = torch.ones_like(mesh.verts_packed())[None]
    mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))
    template_center = mesh.get_bounding_boxes().mean(dim=-1)

    optimal_dist = calculate_optimal_distance(mesh)
    print(f"Calculated optimal camera distance: {optimal_dist:.2f}")

    # Build one renderer per (elevation, azimuth) viewpoint.
    renderers = []
    for elev in args.elevation_angles:
        azim_step = int(args.azimuth_step) * (2 if elev == SPARSE_ELEVATION else 1)
        for azim in range(0, 360, azim_step):
            R, T = look_at_view_transform(optimal_dist * DISTANCE_MARGIN, elev, azim, at=template_center)
            renderers.append(make_renderer(R, T))

    fov_rad = math.radians(args.fov)
    fx = (args.image_size[0] / 2) / math.tan(fov_rad / 2)
    intrinsics = Intrinsics(fx, fx, args.image_size[1] / 2, args.image_size[0] / 2)

    extrinsics_data = []
    for i, (renderer, extrinsics, rasterizer) in enumerate(renderers):
        image_torch = renderer(mesh)[0, ..., :3].cpu().data.numpy()

        # Convert PyTorch3D world->cam pose to an OpenCV world->cam pose for the JSON.
        pose = Pose(
            extrinsics,
            pose_type=PoseType.WORLD_2_CAM,
            camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D,
        )
        pose = pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
        pose = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)
        pose = pose.change_pose_type(PoseType.WORLD_2_CAM, inplace=False)

        image_cv2 = cv2.cvtColor((image_torch * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f'rendered_mesh_{i:02d}.png'), image_cv2)
        extrinsics_data.append(np.array(pose))

    save_camera_params(
        os.path.join(scan_folder, 'camera_params.json'),
        np.stack(extrinsics_data).tolist(),
        intrinsics,
    )


if __name__ == '__main__':
    main()
