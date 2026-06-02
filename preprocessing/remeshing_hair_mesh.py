"""Isotropic remeshing of the extracted hair mesh.

Reads ``<dir>/<case:03d>/000/mesh.obj`` and writes ``mesh_remeshed.obj`` next to it,
using pymeshlab's isotropic explicit remeshing (default parameters).
"""

import argparse
import os

import pymeshlab


def main(args):
    sub = '001' if args.case == 478 else '000'
    mesh_file = os.path.join(args.dir, f'{args.case:03d}', sub, 'mesh.obj')

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)
    ms.apply_filter('meshing_isotropic_explicit_remeshing')
    ms.save_current_mesh(mesh_file.replace('.obj', '_remeshed.obj'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', default='./implicit-hair-data/data/nphm', type=str)
    parser.add_argument('--case', type=int, default=39)
    args = parser.parse_args()
    main(args)
