import argparse
import os
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import cKDTree


LINE_THRESHOLD = 3
STD_THRESHOLD = 0.6

# Function to find the tip and the toe of the line
def find_tip_and_toe(edges):
    from collections import defaultdict, deque

    # Create an adjacency list
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Find all endpoints (vertices with only one connection)
    endpoints = [v for v in adj_list if len(adj_list[v]) == 1]

    if len(endpoints) > 2:
        endpoints = endpoints[:2]
    
    if len(endpoints) == 0:
        endpoints = [edges[0][0]]

    return endpoints


# Function to trace the path from tip to toe
def trace_path(edges, start):
    # Create an adjacency list
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    path = []
    visited = set()

    # BFS
    # queue = deque([start])
    # while queue:
    #     node = queue.popleft()
    #     if node not in visited:
    #         visited.add(node)
    #         path.append(node)
    #         for neighbor in adj_list[node]:
    #             if neighbor not in visited:
    #                 queue.append(neighbor)
    
    # DFS
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            path.append(node)
            # Add neighbors to the stack in reverse order to maintain the correct path order
            for neighbor in reversed(adj_list[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return path


def get_crest_info(loaded_file, file_format):
    # Loop through the header.
    if file_format.lower() == 'ply':
        for _ in range(10):
            line = loaded_file.readline()
            if "element vertex" in line:
                n_vertices = int(line.split(' ')[-1])
            
            if "element edge" in line:
                n_edges = int(line.split(' ')[-1])

        return n_vertices, n_edges
    
    elif file_format.lower() == 'txt':
        n_vertices = int(loaded_file.readline())
        n_edges = int(loaded_file.readline())
        n_strands = int(loaded_file.readline())
        return n_vertices, n_edges, n_strands

    else:
        raise ValueError('File format is not supported! Only .ply and .txt')


def get_strands_and_edges(loaded_file, n_vertices, n_edges, n_strands):
    # Assuming file format .txt is loaded.
    strands = {}
    vertices = []
    edges = []
    for vertex_id in range(n_vertices):
        line = loaded_file.readline()
        vertex = [float(position) for position in line.split(' ')[:-1]]
        vertices.append(vertex)
        strand_id = int(line.split(' ')[-1])
        assert len(vertex) == 3

        strands[strand_id] = strands.get(strand_id, []) + [vertex_id]

        # if strand_id not in strands:
        #     strands[strand_id] = np.array(vertex)
        # else:
        #     strands[strand_id] = np.vstack([strands[strand_id], np.array(vertex)])

    for _ in range(n_strands):
        loaded_file.readline()

    for _ in range(n_edges):
        line = loaded_file.readline()
        edge = tuple(int(node) for node in line.split(' ')[:-1])
        edges.append(edge)

    vertices = np.array(vertices)
    return vertices, strands, edges


def calculate_orient(crest_lines, window_size=3):
    # Calculate orientations
    orient = np.array(crest_lines[1:] - crest_lines[:-1])
    
    # Smooth orientations
    orient_smooth = np.array([np.mean(orient[max(0, i-window_size//2):min(len(orient), i+window_size//2+1)], axis=0) 
                              for i in range(len(orient))])
    
    # Add first orientation to the beginning to match length of crest_lines
    orient_smooth = np.vstack([orient_smooth[0], orient_smooth])
    
    # Normalize
    norms = np.linalg.norm(orient_smooth, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    orient_normalized = orient_smooth / norms
    
    return orient_normalized


def calculate_hair_orient(crest_lines, scalp_points=None, min_window=3, max_window=7):
    def compute_local_frame(points):
        """Compute local coordinate frame using PCA."""
        mean = np.mean(points, axis=0)
        centered = points - mean
        _, _, vh = np.linalg.svd(centered)
        return vh

    def adaptive_window(i, curvature):
        """Determine window size based on local curvature."""
        return max(min_window, min(max_window, int(max_window * (1 - curvature[i]))))

    # Estimate local curvature
    curvature = np.zeros(len(crest_lines))
    for i in range(1, len(crest_lines) - 1):
        v1 = crest_lines[i] - crest_lines[i-1]
        v2 = crest_lines[i+1] - crest_lines[i]
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        if norm_v1 > eps and norm_v2 > eps:
            dot_product = v1 @ v2
            # Clamp the value to [-1, 1] to prevent numerical errors
            cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
            curvature[i] = 1 - cos_angle
        else:
            # If either vector is too small, use the previous curvature value
            # or 0 if it's the first valid calculation
            curvature[i] = curvature[i-1] if i > 1 else 0.0

    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    # Calculate orientations with adaptive window size
    orientations = []
    for i in range(len(crest_lines)):
        window = adaptive_window(i, curvature)
        start = max(0, i - window // 2)
        end = min(len(crest_lines), i + window // 2 + 1)
        local_points = crest_lines[start:end]
        local_frame = compute_local_frame(local_points)
        orientations.append(local_frame[0])  # First principal component

    orientations = np.array(orientations)

    # Ensure root-to-tip directionality if scalp points are provided
    if scalp_points is not None:
        tree = cKDTree(scalp_points)
        _, root_index = tree.query(crest_lines[0])
        root_to_tip = crest_lines[-1] - scalp_points[root_index]
        if root_to_tip @ orientations[0] < 0:
            orientations = -orientations

    # Smooth orientations
    smoothed = np.zeros_like(orientations)
    for i in range(len(orientations)):
        window = adaptive_window(i, curvature)
        start = max(0, i - window // 2)
        end = min(len(orientations), i + window // 2 + 1)
        smoothed[i] = np.mean(orientations[start:end], axis=0)

    # Normalize
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    orient_normalized = smoothed / norms

    return orient_normalized


# Loop through the strands
def write_output(vertices, strands, edges, output_file):
    vertices_counter = 0
    for strand_id in strands.keys():
        vertices_idx = strands[strand_id]
        filtered_edges = [edge for edge in edges if edge[0] in vertices_idx or edge[1] in vertices_idx]

        # Find the tip and the toe
        endpoints = find_tip_and_toe(filtered_edges)
        if len(endpoints) == 2:
            tip, toe = endpoints[0], endpoints[1]
        elif len(endpoints) == 1:
            tip = endpoints[0]
        else:
            continue

        # Trace the path from tip to toe
        path = trace_path(filtered_edges, tip)

        # Print the sorted vertices from tip to toe
        sorted_vertices = vertices[path]
        orients = calculate_hair_orient(sorted_vertices)
        angles = np.array([orients[i] @ orients[i - 1] for i in range(1, len(orients) - 1)])
        colors = (angles <= -0.8).reshape(-1, 1) * np.array([255, 255, 255]).reshape(1, -1)            
        colors = np.vstack([colors, [[0, 0, 0]] * 2])
        assert len(sorted_vertices) == len(orients)
        if len(sorted_vertices) >= LINE_THRESHOLD and np.std(angles) < STD_THRESHOLD:
            vertices_counter += len(sorted_vertices)
            for vertex, orient, color in zip(sorted_vertices, orients, colors):
                if (color == np.array([255, 255, 255])).all():
                    orient = orient * (-1)
                    color  = np.array([0, 0, 0])
                vertex_to_write = np.array2string(vertex, precision=7, separator =' ', suppress_small=False)[1:].replace(']', ' ')
                orient_to_write = np.array2string(orient, precision=7, separator =' ', suppress_small=False)[1:].replace(']', ' ')
                colors_to_write = np.array2string(color, precision=7, separator =' ', suppress_small=False)[1:].replace(']', '\n')
                output_file.write(vertex_to_write + orient_to_write + colors_to_write)
                    
    
    return vertices_counter


def trace_crest_lines(args):
    ravines_file = open('ravines.txt', 'r')
    ridges_file = open('ridges.txt', 'r')
    file_format = ravines_file.name.split('.')[-1]

    n_vertices_ravines, n_edges_ravines, n_strands_ravines = get_crest_info(ravines_file, file_format)
    vertices_ravines, strands_ravines, edges_ravines = get_strands_and_edges(
        ravines_file, n_vertices_ravines, n_edges_ravines, n_strands_ravines
    )

    n_vertices_ridges, n_edges_ridges, n_strands_ridges = get_crest_info(ridges_file, file_format)
    vertices_ridges, strands_ridges, edges_ridges = get_strands_and_edges(
        ridges_file, n_vertices_ridges, n_edges_ridges, n_strands_ridges
    )

    # Write the header of ply
    ply_header = [
        "ply",
        "format ascii 1.0",
        "element vertex",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    # output_filename = args.filename.replace('mesh', 'orients').replace('.txt', f'_line_{LINE_THRESHOLD}_std_{STD_THRESHOLD}.ply')

    # Get the directory and basename separately
    dir_path = os.path.dirname(args.filename)
    base_name = os.path.basename(args.filename)

    # Modify only the basename
    new_base_name = base_name.replace('mesh', 'orients').replace('.txt', f'_line_{LINE_THRESHOLD}_std_{STD_THRESHOLD}.ply')

    # Rejoin with the directory path
    output_filename = os.path.join(dir_path, new_base_name)

    output_file = open(output_filename, 'w')
    for ply in ply_header:
        if "element vertex" in ply:
            output_file.write(ply + ' ' + str(n_vertices_ravines + n_vertices_ridges) + '\n')
        else:
            output_file.write(ply + '\n')

    filtered_ravines_count = write_output(vertices=vertices_ravines, strands=strands_ravines, edges=edges_ravines, output_file=output_file)
    filtered_ridges_count = write_output(vertices=vertices_ridges, strands=strands_ridges, edges=edges_ridges, output_file=output_file)

    ravines_file.close()
    ridges_file.close()
    output_file.close()

    with open(output_filename, 'r') as output_file:
        text = output_file.read().replace(f'element vertex {str(n_vertices_ravines + n_vertices_ridges)}',
                                        f'element vertex {str(filtered_ravines_count + filtered_ridges_count)}')


    with open(output_filename, "w") as output_file:
        output_file.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--filename', default='./implicit-hair-data/data/nphm/019/19_mesh_sharpened_subdiv.txt', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    trace_crest_lines(args)
