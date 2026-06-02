"""Extract the hair mesh from a segmented head scan.

Refines the hair segmentation (dilate/erode), keeps scan vertices close to the hair
region (chamfer), removes the ears (using the FLAME registration), and finally keeps the
largest connected component. Writes ``segmented_refined_hair.obj`` and ``mesh.obj``.

The only dataset-specific differences are the FLAME registration filename and the
scan-chamfer threshold, selected by ``--dataset``.
"""

import argparse
import os
import pickle as pk

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_gather, knn_points

# Per-label RGBA colors of the NPHMv2 segmentation mesh; index 13 is hair.
COLORS = [
    (0.15196265161417932, 0.9960074647996185, 0.056455206420483184),
    (1.0, 0.0, 1.0),
    (0.0, 0.5, 1.0),
    (1.0, 0.5, 0.0),
    (0.5, 0.25, 0.5),
    (0.8287688577502536, 0.9974111526990148, 0.25534135453919227),
    (0.19204822368146024, 0.9674099082930834, 0.7089872150114194),
    (0.18974610373767486, 0.0113867394908016, 0.987720480177032),
    (1.0, 0.0, 0.0),
    (0.024164823996170814, 0.504017555447308, 0.15957672065835793),
    (1.0, 0.5, 1.0),
    (0.0, 0.0, 0.5),
    (0.5753721833879069, 0.6613309331010218, 0.7724912645477964),
    (0.5456606360848165, 0.647436791896486, 0.1666277922028755),
    (0.9937081099784579, 0.31184249142315956, 0.530789732625259),
    (0.5417979844742928, 0.0753907901970361, 0.018094454076140742),
    (0.07458428906151104, 0.6272489808983358, 0.5629589765679636),
    (0.5933799287307657, 0.1877949395380566, 0.9634759480913743),
    (0.890054617436647, 0.6806840500601011, 0.4585165367063032),
]
COLORS = (np.array(COLORS) * 255).astype(np.uint8)
COLORS = np.concatenate([COLORS, (np.ones((COLORS.shape[0], 1)) * 255).astype(np.uint8)], axis=1)

HAIR_LABEL = 13
HAIR_DILATE_ITERS = 30       # grow the hair mask...
HAIR_ERODE_ITERS = 85        # ...then erode more, shrinking onto the hair region
EAR_CHAMFER_THRESHOLD = 7e-5  # ear proximity (to FLAME ears) for removal

# Dataset-specific knobs (everything else is shared).
DATASET_CONFIG = {
    'meshy':    {'flame_file': 'flame.ply',        'scan_chamfer_threshold': 5e-5},
    'geomhair': {'flame_file': 'registration.ply', 'scan_chamfer_threshold': 1e-4},
}


def chamfer_distance_no_reduction(x, y, x_lengths=None, y_lengths=None,
                                  x_normals=None, y_normals=None, weights=None):
    """One-directional (x->y) chamfer distances and normal-cosine distances, no reduction.

    Adapted from ``pytorch3d.loss.chamfer`` with the y->x direction omitted; returns the
    per-point distances ``(N, P1)`` instead of a reduced scalar.
    """
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]  # [N, P1]

    cham_norm_x = x.new_zeros(())
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        cham_norm_x = 1 - torch.abs(F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6))
        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    return cham_x, cham_norm_x


def pruned_chamfer_loss(x, y, x_normals=None, y_normals=None, dist_thr=None,
                        normals_thr=None, mask_x=None, mask_y=None, device='cuda'):
    """Per-point x->y chamfer distances, optionally masking the input point sets."""
    x_masked = x if mask_x is None else x[mask_x]
    y_masked = y if mask_y is None else y[mask_y]
    x_normals_masked = None if x_normals is None else \
        (x_normals if mask_x is None else x_normals[mask_x]).unsqueeze(0)
    y_normals_masked = None if y_normals is None else \
        (y_normals if mask_y is None else y_normals[mask_y]).unsqueeze(0)

    cham_x, cham_norm_x = chamfer_distance_no_reduction(
        x_masked.unsqueeze(0), y_masked.unsqueeze(0),
        x_normals=x_normals_masked, y_normals=y_normals_masked)

    if x_normals is not None and y_normals is not None:
        return cham_x[0], cham_norm_x[0]
    return cham_x[0], cham_norm_x


def dilate_vertex_mask(edges, mask):
    """Grow a boolean vertex mask by one ring across mesh edges."""
    mask_dilated = mask.copy().astype(bool)
    crossing = mask_dilated[edges[:, 0]] != mask_dilated[edges[:, 1]]
    mask_dilated[edges[crossing, 0]] = True
    mask_dilated[edges[crossing, 1]] = True
    return mask_dilated


def remove_vertices_and_corresponding_faces(ver, faces, mask, ver_n=None):
    """Keep masked vertices and reindex the faces that survive entirely."""
    if ver_n is not None:
        assert len(ver) == len(ver_n)
    ver_to_keep = torch.arange(ver.shape[0]).to(ver.device)[mask]
    ver_new = ver[ver_to_keep]
    old2new = torch.full((ver.shape[0],), -1, dtype=torch.long).to(ver.device)
    old2new[ver_to_keep] = torch.arange(ver_to_keep.shape[0]).to(ver.device)
    faces_new = old2new[faces[torch.all(mask[faces], dim=1)]]
    if ver_n is not None:
        return ver_new, ver_n[ver_to_keep], faces_new
    return ver_new, faces_new


def load_mesh(file_path, device='cuda'):
    """Load a mesh with Open3D and return (PyTorch3D mesh, verts, faces, normals)."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = torch.tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
    faces = torch.tensor(np.asarray(mesh.triangles, dtype=np.int64), device=device)
    normals = torch.tensor(np.asarray(mesh.vertex_normals, dtype=np.float32), device=device)
    pytorch3d_mesh = Meshes(verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0))
    return pytorch3d_mesh, vertices, faces, normals


def get_hair_mask(mesh):
    """Boolean per-vertex mask of hair vertices (by segmentation color)."""
    hair_color = COLORS[HAIR_LABEL]
    return np.all(mesh.visual.vertex_colors[:, :3] == hair_color[:3], axis=1)


def main(args, number):
    print(f'Extracting hair on subject number {number} ...')
    cfg = DATASET_CONFIG[args.dataset]

    sub = '001' if args.case == 478 else '000'
    base_folder = os.path.join(args.dir, f'{number:03d}', sub)
    filename_scan = os.path.join(base_folder, 'scan.ply')
    filename_segm = os.path.join(base_folder, 'segmented_model/segmented.obj')
    filename_flame = os.path.join(base_folder, cfg['flame_file'])

    _, verts_scan, faces_scan, _ = load_mesh(filename_scan)
    mesh_segm = trimesh.load(filename_segm)
    verts_segm = torch.from_numpy(np.array(mesh_segm.vertices)).to('cuda')
    faces_segm = torch.from_numpy(np.array(mesh_segm.faces)).to('cuda')
    verts_flame, _ = load_ply(filename_flame)
    verts_flame = verts_flame.to('cuda')

    # Refine the hair segmentation: morphological close (dilate then erode further).
    hair_mask = get_hair_mask(mesh_segm)
    edges = np.array([(v1, v2) for v1, v2 in mesh_segm.edges_unique])
    for _ in range(HAIR_DILATE_ITERS):
        hair_mask = dilate_vertex_mask(edges, hair_mask)
    for _ in range(HAIR_ERODE_ITERS):
        hair_mask = ~dilate_vertex_mask(edges, ~hair_mask)
    colors = mesh_segm.visual.vertex_colors.view(np.ndarray)
    colors[hair_mask] = COLORS[HAIR_LABEL]
    mesh_segm.visual.vertex_colors = colors
    mesh_segm.export(filename_segm.replace('segmented.obj', 'segmented_refined_hair.obj'))

    # Keep scan vertices close to the refined hair region.
    hair_color = COLORS[HAIR_LABEL]
    label_segm = mesh_segm.visual.vertex_colors.view(np.ndarray)
    hair_idx = (label_segm == hair_color).all(axis=1) == 1
    verts_orig, faces_orig = remove_vertices_and_corresponding_faces(
        verts_segm, faces_segm.long(), torch.from_numpy(hair_idx).to('cuda'))

    chamf_scan, _ = pruned_chamfer_loss(
        verts_scan.to(torch.float32), verts_orig.to(device='cuda', dtype=torch.float32))
    chamf_mask = chamf_scan <= cfg['scan_chamfer_threshold']
    verts_orig, faces_orig = remove_vertices_and_corresponding_faces(
        verts_scan, faces_scan.long(), chamf_mask)

    # Remove vertices near the FLAME ears.
    with open('./data/flame_masks.pkl', 'rb') as fin:
        flame_region_masks = pk.load(fin)
    ears_flame_idx = np.concatenate(
        [flame_region_masks['left_ear'], flame_region_masks['right_ear']])
    chamf_ear, _ = pruned_chamfer_loss(verts_orig.to('cuda'), verts_flame[ears_flame_idx])
    chamf_ear_mask = chamf_ear <= EAR_CHAMFER_THRESHOLD
    verts_final, faces_final = remove_vertices_and_corresponding_faces(
        verts_orig, faces_orig.long(), ~chamf_ear_mask)

    # Keep only the largest connected component.
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_final.numpy(force=True)),
        triangles=o3d.utility.Vector3iVector(faces_final.numpy(force=True)))
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, _ = mesh_o3d.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    mesh_o3d.remove_triangles_by_mask(triangle_clusters != cluster_n_triangles.argmax())

    o3d.io.write_triangle_mesh(os.path.join(base_folder, "mesh.obj"), mesh_o3d)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--case', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='geomhair', choices=list(DATASET_CONFIG),
                        help='Selects the FLAME registration filename and chamfer threshold')
    args = parser.parse_args()

    if not args.case:
        folders = sorted(os.listdir(args.dir))
        print(f'There are {len(folders)} subjects.\n')
        for folder in folders:
            main(args, int(folder))
    else:
        main(args, args.case)
