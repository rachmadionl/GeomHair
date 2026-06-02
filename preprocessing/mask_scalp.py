"""Cut the scalp region and build the diffusion mask.

Transfers the segmentation labels onto the FLAME registration, dilates the hair
region, cuts the scalp sub-mesh, and rasterizes it into a UV-space diffusion mask.
Writes ``labeled_flame.ply``, ``cutted_scalp.obj``, ``cut_scalp_verts.pickle`` and
``dif_mask.png`` (plus ``registration_scaled.ply`` when ``--shrink`` is set).
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from skimage.draw import polygon
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj

# Per-label RGBA colors of the segmentation mesh (label index -> color).
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

# Semantic labels kept when transferring onto the FLAME mesh (others are zeroed).
KEEP_LABELS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]
HAIR_LABEL = 13
SCALP_MASK_SIZE = 256        # UV-space diffusion-mask resolution
NN_K = 3                     # neighbors for label transfer (weighted vote)
DILATE_ITERS = 5             # hair-mask dilation passes
CLOSE_ITERS = 5              # extra erosion passes for morphological close
SHRINK_SCALE = 0.90          # FLAME scale for registration_scaled.ply


def create_scalp_mask(scalp_mesh, scalp_uvs):
    """Rasterize the scalp faces into a binary mask in UV space."""
    img = np.zeros((SCALP_MASK_SIZE, SCALP_MASK_SIZE, 1), 'uint8')
    for i in range(scalp_mesh.faces_packed().shape[0]):
        text = scalp_uvs[0][scalp_mesh.faces_packed()[i]].reshape(-1, 2).cpu().numpy()
        poly = (SCALP_MASK_SIZE - 1) / 2 * (text + 1)  # UV [-1, 1] -> pixels
        rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc, :] = 255
    return np.flip(img.transpose(1, 0, 2), axis=0)


def dilate_vertex_mask(edges, mask):
    """Grow a boolean vertex mask by one ring across mesh edges."""
    mask_dilated = mask.copy().astype(bool)
    crossing = mask_dilated[edges[:, 0]] != mask_dilated[edges[:, 1]]
    mask_dilated[edges[crossing, 0]] = True
    mask_dilated[edges[crossing, 1]] = True
    return mask_dilated


def get_label_from_color(vertex_color, colors_map):
    """Map per-vertex RGBA colors to semantic label indices."""
    labels = np.zeros(len(vertex_color), dtype=np.int32)
    for label_idx, color in enumerate(colors_map):
        labels[np.all(vertex_color == color, axis=1)] = label_idx
    return labels


def transfer_labels(source_vertices, source_labels, target_vertices, k=1):
    """Transfer labels from source to target vertices via (weighted) nearest neighbors."""
    kdtree = cKDTree(source_vertices)
    distances, indices = kdtree.query(target_vertices, k=k)

    if k == 1:
        return source_labels[indices]

    weights = 1.0 / (distances + 1e-10)
    transferred_labels = np.zeros(len(target_vertices), dtype=np.int32)
    for i in range(len(target_vertices)):
        label_votes = source_labels[indices[i]]
        unique_labels, _ = np.unique(label_votes, return_counts=True)
        weighted_counts = np.array(
            [np.sum(weights[i][label_votes == label]) for label in unique_labels])
        transferred_labels[i] = unique_labels[np.argmax(weighted_counts)]
    return transferred_labels


def process_meshes(flame_mesh_path, segmented_mesh_path, save_dir, scalp_data_path,
                   is_shrink, is_close, device='cuda'):
    flame_mesh = trimesh.load(flame_mesh_path)
    segm_mesh = trimesh.load(segmented_mesh_path)

    scalp_vert_idx = torch.load(os.path.join(scalp_data_path, 'new_scalp_vertex_idx.pth')).long().to(device)
    scalp_faces = torch.load(os.path.join(scalp_data_path, 'new_scalp_faces.pth'))[None].to(device)
    scalp_uvs = torch.load(os.path.join(scalp_data_path, 'new_scalp_uvcoords.pth'))[None].to(device)

    # Segmentation labels -> transfer onto the FLAME registration.
    segm_labels = get_label_from_color(segm_mesh.visual.vertex_colors, COLORS)
    flame_vertices = np.array(flame_mesh.vertices)
    flame_labels = transfer_labels(np.array(segm_mesh.vertices), segm_labels, flame_vertices, k=NN_K)
    flame_labels[~np.isin(flame_labels, KEEP_LABELS)] = 0

    # Dilate (and optionally close) the hair region across the mesh graph.
    edges = np.array([(v1, v2) for v1, v2 in flame_mesh.edges_unique])
    hair_mask = flame_labels == HAIR_LABEL
    for _ in range(DILATE_ITERS):
        hair_mask = dilate_vertex_mask(edges, hair_mask)
    if is_close:
        for _ in range(CLOSE_ITERS):
            hair_mask = ~dilate_vertex_mask(edges, ~hair_mask)
    flame_labels[hair_mask] = HAIR_LABEL

    flame_colors = np.zeros((len(flame_vertices), 4), dtype=np.uint8)
    for label in np.unique(flame_labels):
        flame_colors[flame_labels == label] = COLORS[label]
    flame_mesh.visual.vertex_colors = flame_colors
    flame_mesh.export(os.path.join(save_dir, 'labeled_flame.ply'))

    # Build the scalp sub-mesh, keeping only vertices that are not the (label-1) face.
    scalp_verts = torch.tensor(flame_vertices).to(device)[None, scalp_vert_idx]
    scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces).to(device)
    scalp_keep_mask = torch.tensor(flame_labels != 1).to(device)
    sorted_idx = torch.where(scalp_keep_mask[scalp_vert_idx])[0]

    full_scalp_list = sorted(sorted_idx.cpu().numpy())
    idx_map = {old: new for new, old in enumerate(full_scalp_list)}
    faces_masked = [[idx_map[int(v)] for v in face]
                    for face in scalp_mesh.faces_packed().cpu().numpy()
                    if all(v in full_scalp_list for v in face)]

    cut_scalp_verts = scalp_mesh.verts_packed()[sorted_idx]
    cut_scalp_faces = torch.tensor(faces_masked).to(device)
    save_obj(os.path.join(save_dir, 'cutted_scalp.obj'), verts=cut_scalp_verts, faces=cut_scalp_faces)

    with open(os.path.join(save_dir, 'cut_scalp_verts.pickle'), 'wb') as f:
        pickle.dump(list(sorted_idx.cpu().numpy()), f)

    # Rasterize the cut scalp into the UV-space diffusion mask.
    scalp_uvs = scalp_uvs[:, sorted_idx]
    scalp_mesh = Meshes(verts=cut_scalp_verts.unsqueeze(0), faces=cut_scalp_faces.unsqueeze(0)).to(device)
    scalp_mask = create_scalp_mask(scalp_mesh, scalp_uvs)
    cv2.imwrite(os.path.join(save_dir, 'dif_mask.png'), scalp_mask)

    if is_shrink:
        flame_mesh.vertices = flame_mesh.vertices * SHRINK_SCALE
        flame_mesh.export(os.path.join(save_dir, 'registration_scaled.ply'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Data directory to process')
    parser.add_argument('--case', type=int, required=True, help='Which sample to process')
    parser.add_argument('--scalp_data_path', type=str, required=True, help='Path to scalp data directory')
    parser.add_argument('--shrink', type=bool, default=True, help='Whether to also write a shrunk FLAME mesh')
    parser.add_argument('--close', type=bool, default=True,
                        help='If True, perform morphological close (dilate then erode); otherwise dilate only')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    sub = '001' if args.case == 478 else '000'
    data_dir = os.path.join(args.dir, f'{args.case:03d}', sub)
    path_to_mesh = os.path.join(data_dir, 'registration.ply')
    path_to_segm = os.path.join(data_dir, 'segmented_model', 'segmented.obj')

    process_meshes(path_to_mesh, path_to_segm, data_dir, args.scalp_data_path,
                   args.shrink, args.close, args.device)
    print(f"Processing complete. Results saved in {data_dir}")


if __name__ == "__main__":
    main()
