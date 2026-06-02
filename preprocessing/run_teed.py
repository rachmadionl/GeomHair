"""Run TEED edge detection over the multi-view renders of the hair mesh.

Reads ``<dir>/<case:03d>/000/rendered_mesh/*.png`` and writes the fused TEED
edge map for each view to ``<dir>/<case:03d>/000/teed/<same_name>.png``.

This is a thin, self-contained wrapper around the vendored TEED model
(``submodules/TEED``); it reproduces TEED's CLASSIC test-time transform and the
fused-output post-processing (sigmoid -> normalize -> invert) so the result
matches the upstream ``main.py`` output, while writing flat files (no ``fused/``
subdir) as the rest of the pipeline expects.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

TEED_DIR = os.path.join(os.getcwd(), 'submodules', 'TEED')
sys.path.insert(0, TEED_DIR)

from ted import TED  # noqa: E402
from utils.img_processing import image_normalization  # noqa: E402

# TEED CLASSIC test config (dataset.py): BGR mean, checkpoint trained on BIPED.
MEAN_BGR = [104.007, 116.669, 122.679]
CHECKPOINT = os.path.join(TEED_DIR, 'checkpoints', 'BIPED', '7', '7_model.pth')
IMG_EXTS = ('.png', '.jpg', '.jpeg')


def preprocess(img):
    img = np.array(img, dtype=np.float32)
    img -= MEAN_BGR
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img.copy()).float()


def main(args):
    sub = '001' if args.case == 478 else '000'
    scan_folder = os.path.join(args.dir, f'{args.case:03d}', sub)
    in_dir = os.path.join(scan_folder, 'rendered_mesh')
    out_dir = os.path.join(scan_folder, 'teed')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TED().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(IMG_EXTS))
    with torch.no_grad():
        for fn in files:
            img = cv2.imread(os.path.join(in_dir, fn), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            x = preprocess(img).unsqueeze(0).to(device)
            preds = model(x)
            fused = torch.sigmoid(preds[-1])[0, 0].cpu().numpy()
            edge = np.uint8(image_normalization(fused))
            edge = cv2.bitwise_not(edge)
            if edge.shape[0] != h or edge.shape[1] != w:
                edge = cv2.resize(edge, (w, h))
            cv2.imwrite(os.path.join(out_dir, fn), edge)
            print(f'TEED: {fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', required=True, type=str)
    parser.add_argument('--case', required=True, type=int)
    # accepted for compatibility with the original TEED invocation; unused
    parser.add_argument('--choose_test_data', type=int, default=-1)
    args, _ = parser.parse_known_args()
    main(args)
