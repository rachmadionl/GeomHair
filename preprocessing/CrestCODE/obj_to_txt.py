import argparse
import os

import open3d as o3d
from pytorch3d.io import IO


def convert_obj_to_txt(args):
    if args.case != 478:
        filename = os.path.join(args.dir, f'{args.case:03d}', '000', f'mesh.obj')
    else:
        filename = os.path.join(args.dir, f'{args.case:03d}', '001', f'mesh.obj')
    if 'ply' in filename:
        mesh = IO().load_mesh(filename)
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()

        mesh_o3d = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts.numpy(force=True)),
            triangles=o3d.utility.Vector3iVector(faces.numpy(force=True))
        )
        o3d.io.write_triangle_mesh(filename.replace('ply', 'obj'), mesh_o3d)

        file = open(filename.replace('ply', 'obj'), "r")
    else:
        file = open(filename, "r")

    # Loop through the header.
    for i in range(4):
        line = file.readline()
        if i == 2:
            n_vertices = int(line.split(sep=' ')[-1])
        if i == 3:
            n_faces = int(line.split(sep=' ')[-1])

    if 'ply' in filename:
        output_file = open(filename.replace('ply', 'txt'), "w")
    else:
        output_file = open(filename.replace('obj', 'txt'), "w")
    # Write header.
    output_file.write(str(n_vertices) + '\n')
    output_file.write(str(n_faces) + '\n')
    output_file.write(str(1) + '\n') # Number of k-link.
    output_file.write(str(1) + '\n') # With or without line tracing.

    # Write vertices.
    for _ in range(n_vertices):
        line = file.readline()
        line_to_write = " ".join(line.split(' ')[1:])
        output_file.write(line_to_write)

    # Write faces.
    for _ in range(n_faces):
        line = file.readline()
        vertices_obj = line.split(' ')[1:]
        vertices_ply2 = [str(int(vert) - 1) for vert in vertices_obj]
        line_to_write = " ".join(vertices_ply2) + "\n"
        output_file.write(line_to_write)

    file.close()
    output_file.close()
    return filename.replace('obj', 'txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--filename', default='./implicit-hair-data/data/nphm/019/19_mesh_sharpened_subdiv.obj', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    convert_obj_to_txt(args)