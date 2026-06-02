#!/bin/bash
#
# Download the pretrained priors needed for strand optimization (training) into
# ./pretrained_models/ :
#   - strand prior  -> pretrained_models/strand_prior/strand_ckpt.pth   (~69 MB)
#   - HAAR diffusion prior -> pretrained_models/haar_prior/haar_diffusion.pth  (~14 GB)
#
# Requires gdown >= 5 (older versions fail on Google Drive's confirmation flow):
#   pip install -U gdown
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/pretrained_models"
mkdir -p "$DEST/strand_prior" "$DEST/haar_prior"

# Strand prior (from the Neural Haircut model release).
if [ ! -f "$DEST/strand_prior/strand_ckpt.pth" ]; then
    echo "Downloading strand_ckpt.pth ..."
    gdown 1DESwUb-nsmi38VCDvnBwpd9kjcWONNT6 -O "$DEST/strand_prior/strand_ckpt.pth"
else
    echo "strand_ckpt.pth already present, skipping."
fi

# HAAR diffusion prior. The released checkpoint is ~14 GB (it bundles optimizer
# state); we slim it to model_ema + config (~3.4 GB) so training startup is fast and
# fits in modest RAM. base_meshy.yaml points at the slim file.
if [ ! -f "$DEST/haar_prior/haar_diffusion_slim.pth" ]; then
    echo "Downloading haar_diffusion.pth (~14 GB) ..."
    gdown 1vCQ7vX3v6GWMQUv9gUkqJOutvAwb3uvF -O "$DEST/haar_prior/haar_diffusion.pth"
    echo "Slimming to haar_diffusion_slim.pth ..."
    python "$REPO_ROOT/scripts/slim_haar_checkpoint.py" \
        "$DEST/haar_prior/haar_diffusion.pth" "$DEST/haar_prior/haar_diffusion_slim.pth"
    rm -f "$DEST/haar_prior/haar_diffusion.pth"   # keep only the slim checkpoint
else
    echo "haar_diffusion_slim.pth already present, skipping."
fi

echo "Done. Pretrained models are in $DEST"
