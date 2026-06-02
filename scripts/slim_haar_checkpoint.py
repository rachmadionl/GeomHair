"""Slim the HAAR diffusion checkpoint to just what training needs.

The released haar_diffusion.pth (~14 GB) bundles optimizer/scheduler state that
torch.load deserializes in full on every training start. The strand optimizer only
uses ckpt['model_ema'] (+ 'config'), so we re-save just those into a small file that
loads in seconds. Uses mmap so the 14 GB source is not fully read into RAM.

Usage: python scripts/slim_haar_checkpoint.py <in.pth> <out.pth>
"""

import sys
import torch


def main(src, dst):
    ckpt = torch.load(src, map_location='cpu', mmap=True)
    slim = {'model_ema': ckpt['model_ema']}
    if 'config' in ckpt:
        slim['config'] = ckpt['config']
    torch.save(slim, dst)
    print(f'Wrote slim checkpoint: {dst}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
