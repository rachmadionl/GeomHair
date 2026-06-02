#!/bin/bash
#
# GeomHair end-to-end: preprocessing followed by strand optimization.
#
# Usage:
#   scripts/run_all.sh <DATA_DIR> <CASE> [--dataset meshy|geomhair]
#       [--is_shrink True/False]
#       [--exp_dir DIR] [--use_3do_2do True/False] [--use_old_dif_prior True/False]
#
# Training-only flags (--exp_dir, --use_3do_2do, --use_old_dif_prior) are routed to
# the training stage; shared flags (--dataset, --is_shrink) go to both.
#
set -e
set -o pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <DATA_DIR> <CASE> [--dataset meshy|geomhair] [--is_shrink True/False] [--exp_dir DIR] [--use_3do_2do True/False] [--use_old_dif_prior True/False]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
folder="$1"
number="$2"
shift 2

DATASET=meshy
IS_SHRINK=False
EXP_DIR="exps"
USE_3DO_2DO=False
USE_OLD_DIF_PRIOR=False

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)           DATASET="$2"; shift 2 ;;
        --is_shrink)         IS_SHRINK="$2"; shift 2 ;;
        --exp_dir)           EXP_DIR="$2"; shift 2 ;;
        --use_3do_2do)       USE_3DO_2DO="$2"; shift 2 ;;
        --use_old_dif_prior) USE_OLD_DIF_PRIOR="$2"; shift 2 ;;
        *) echo "Invalid argument: $1"; exit 1 ;;
    esac
done

"$SCRIPT_DIR/run_preprocessing.sh" "$folder" "$number" --dataset "$DATASET" --is_shrink "$IS_SHRINK"

"$SCRIPT_DIR/run_training.sh" "$folder" "$number" \
    --dataset "$DATASET" \
    --is_shrink "$IS_SHRINK" \
    --exp_dir "$EXP_DIR" \
    --use_3do_2do "$USE_3DO_2DO" \
    --use_old_dif_prior "$USE_OLD_DIF_PRIOR"
