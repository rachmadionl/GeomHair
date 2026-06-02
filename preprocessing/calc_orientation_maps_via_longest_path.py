"""2D hair orientation maps from TEED edge maps.

For each TEED edge image: threshold and clean it, skeletonize, build a graph over the
skeleton, trace longest paths, and assign a per-pixel orientation (top-to-bottom). Writes
``orientation_maps/``, ``quiver_plot/`` and ``longest_path/`` for every view, in parallel.
"""

import argparse
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import matplotlib
matplotlib.use('Agg')

import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import PIL.ImageOps
from skimage import measure
from skimage.segmentation import flood
from skimage.morphology import skeletonize
from tqdm import tqdm

MIN_COMPONENT_AREA = 10          # skip skeleton components smaller than this
ADJACENCY_DIST_SQ = 4            # connect skeleton pixels within this squared distance
CLOSE_KERNEL = np.ones((40, 40))
BOUNDARY_KERNEL = np.ones((7, 7))


def make_graph(coords):
    """Build an upper-triangular adjacency matrix over nearby skeleton pixels."""
    v = coords.astype(float)
    norm = np.linalg.norm(v, axis=1, keepdims=True) ** 2
    pdist = norm + norm.T - 2 * v @ v.T
    adj_mx = (pdist <= ADJACENCY_DIST_SQ) & ~(np.eye(len(v)).astype(bool))

    ii, jj = np.meshgrid(np.arange(len(v)), np.arange(len(v)), indexing='ij')
    adj_mx[ii > jj] = 0  # keep the upper triangle only (undirected adjacency)
    return v, adj_mx


def get_complete_paths(coords, graph):
    """Cover the skeleton graph with shortest paths between endpoints/branch points."""
    undirected_graph = nx.Graph(graph)
    endpoints = [n for n in undirected_graph.nodes() if undirected_graph.degree(n) == 1]
    branch_points = [n for n in undirected_graph.nodes() if undirected_graph.degree(n) > 2]
    important_points = endpoints + branch_points

    paths = []
    edges_covered = set()
    for i in range(len(important_points)):
        for j in range(i + 1, len(important_points)):
            try:
                path = nx.shortest_path(undirected_graph, important_points[i], important_points[j])
            except nx.NetworkXNoPath:
                continue
            path_edges = {tuple(sorted([path[k], path[k + 1]])) for k in range(len(path) - 1)}
            if path_edges - edges_covered:
                paths.append(path)
                edges_covered.update(path_edges)

    # Cover any edges not on a path above (e.g. isolated loops).
    remaining_edges = set(map(tuple, map(sorted, undirected_graph.edges()))) - edges_covered
    if remaining_edges:
        remaining_graph = nx.Graph()
        remaining_graph.add_edges_from(remaining_edges)
        for component in nx.connected_components(remaining_graph):
            component_graph = remaining_graph.subgraph(component)
            start = list(component)[0]
            path_lengths = nx.single_source_shortest_path_length(component_graph, start)
            furthest_node = max(path_lengths.items(), key=lambda x: x[1])[0]
            path = nx.shortest_path(component_graph, start, furthest_node)
            paths.append(path)
            for k in range(len(path) - 1):
                edges_covered.add(tuple(sorted([path[k], path[k + 1]])))

    return paths


def calculate_edge_orientations(coords, graph):
    """Per-pixel edge orientations along each path, oriented top-to-bottom."""
    paths = get_complete_paths(coords, nx.Graph(graph))

    orientations = {}
    for path in paths:
        path_coords = coords[path]
        if path_coords[0][0] > path_coords[-1][0]:  # force first y <= last y (top to bottom)
            path_coords = path_coords[::-1]

        for i in range(len(path_coords) - 1):
            current, next_point = path_coords[i], path_coords[i + 1]
            angle = np.arctan2(next_point[0] - current[0], next_point[1] - current[1])
            for point in (current, next_point):
                orientations.setdefault((point[0], point[1]), []).append(angle)

    return orientations


def find_orient_via_longest_path(skeleton):
    img_labeled = measure.label(skeleton)
    skeleton_longest_paths = np.zeros_like(skeleton, dtype=np.uint8)
    orientation_longest_paths = np.zeros_like(skeleton, dtype=float)

    for prop in measure.regionprops(img_labeled):
        if prop.area < MIN_COMPONENT_AREA:
            continue
        _, graph = make_graph(prop.coords)
        orientations = calculate_edge_orientations(prop.coords, graph)
        for coord, angles in orientations.items():
            # Circular mean of the per-edge angles at this pixel.
            mean_angle = np.angle(np.mean(np.exp(1j * np.array(angles))))
            skeleton_longest_paths[coord[0], coord[1]] = 255
            orientation_longest_paths[coord[0], coord[1]] = mean_angle

    return orientation_longest_paths, skeleton_longest_paths


def quiver_plot_artem(img: np.ndarray, sample_rate: int,
                      vis_img: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None) -> None:
    """Quiver visualization of an orientation image (angles in degrees)."""
    valid_mask = img != 0
    y_coords, x_coords = np.where(valid_mask)
    orientations = img[valid_mask]
    orientations_rad = np.deg2rad(orientations)

    x_coords = x_coords[::sample_rate]
    y_coords = y_coords[::sample_rate]
    orientations_rad = orientations_rad[::sample_rate]
    u = np.cos(orientations_rad)
    v = np.sin(orientations_rad)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.imshow(img if vis_img is None else vis_img, cmap='gray')
    ax.quiver(x_coords, y_coords, u, v, orientations[::sample_rate],
              angles='xy', scale_units='xy', scale=0.1)
    ax.set_title('Orientation of Longest Path')

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


def extract_orientation(image_path, orientation_path, quiver_path, longest_path):
    img = np.array(PIL.ImageOps.invert(Image.open(image_path)))

    T_otsu = mh.otsu(img)
    img_otsu = np.where(img > T_otsu, 255, 0)

    # Drop the closed outer boundary so it isn't skeletonized as hair.
    img_fgd = flood(mh.close(img_otsu, Bc=CLOSE_KERNEL) > 0, (0, 0))
    boundary = img_fgd ^ mh.erode(img_fgd)
    thick_boundary = mh.dilate(boundary, BOUNDARY_KERNEL)
    img_otsu[thick_boundary == 1] = 0

    skeleton = skeletonize(img_otsu)
    orient, skeleton_longest_path = find_orient_via_longest_path(skeleton)
    orient_deg = orient * 180 / np.pi

    quiver_plot_artem(orient_deg, sample_rate=5, vis_img=skeleton_longest_path,
                      save_path=quiver_path.replace('.png', '_quiver.png'))

    Image.fromarray(orient_deg.astype(np.uint8)).convert('RGB').save(
        orientation_path.replace('rendered_mesh', 'orient'))
    Image.fromarray(skeleton_longest_path.astype(np.uint8)).convert('RGB').save(
        longest_path.replace('rendered_mesh', 'longest_path'))


def process_single_image(teed_image, base_dirs):
    return extract_orientation(
        os.path.join(base_dirs['teed_dir'], teed_image),
        orientation_path=os.path.join(base_dirs['output_dir'], teed_image),
        quiver_path=os.path.join(base_dirs['quiver_dir'], teed_image),
        longest_path=os.path.join(base_dirs['longest_path_dir'], teed_image),
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate orientation maps from TEED edges')
    parser.add_argument('--case', type=int, required=True, help='Case number to process (e.g., 17)')
    parser.add_argument('--dir', type=str,
                        default='/cluster/himring/asevastopolsky/NPHMHaircut/nphm/scan',
                        help='Base directory containing scan data')
    return parser.parse_args()


def main():
    args = parse_args()
    sub = '001' if args.case == 478 else '000'
    teed_dir = os.path.join(args.dir, f'{args.case:03d}', sub, 'teed')
    scan_folder = os.path.dirname(teed_dir)

    base_dirs = {
        'teed_dir': teed_dir,
        'output_dir': os.path.join(scan_folder, 'orientation_maps'),
        'quiver_dir': os.path.join(scan_folder, 'quiver_plot'),
        'longest_path_dir': os.path.join(scan_folder, 'longest_path'),
    }
    for directory in ('output_dir', 'quiver_dir', 'longest_path_dir'):
        os.makedirs(base_dirs[directory], exist_ok=True)

    teed_images = sorted(os.listdir(teed_dir))
    process_fn = functools.partial(process_single_image, base_dirs=base_dirs)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_fn, teed_images),
                  total=len(teed_images), desc="Processing images"))


if __name__ == '__main__':
    main()
