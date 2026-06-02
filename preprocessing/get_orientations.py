"""3D hair orientations via crest lines.

Thin wrapper around the vendored CrestCODE tool: converts the remeshed hair mesh to
CrestCODE's text format, runs the ``setCurvature`` binary, traces crest lines into the
3D orientation point cloud (``orients_line_*.ply``), and cleans up intermediate files.
"""

import argparse
import os
import sys

sys.path.append('./preprocessing/CrestCODE')
from CrestCODE.obj_to_txt import convert_obj_to_txt
from CrestCODE.trace_path import trace_crest_lines

CREST_BIN = './preprocessing/CrestCODE/setCurvature'


def main(args):
    txt_file = convert_obj_to_txt(args)
    txt_output_file = txt_file.replace('.txt', '_output.txt')
    os.system(f'{CREST_BIN} {txt_file} {txt_output_file}')

    args.filename = txt_file
    trace_crest_lines(args)

    for tmp in ('ridges.txt', 'ravines.txt', txt_file, txt_output_file):
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', default='./implicit-hair-data/data/nphm', type=str)
    parser.add_argument('--case', type=int, default=39)
    args = parser.parse_args()
    main(args)
