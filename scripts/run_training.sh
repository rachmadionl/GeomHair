#!/bin/bash
#
# GeomHair strand optimization (training).
#
# Runs after preprocessing, on the per-case artifacts produced by
# scripts/run_preprocessing.sh. The configs use DATA_ROOT/CASE_NAME tokens that are
# substituted at runtime (DATA_ROOT <- <DATA_ROOT>, CASE_NAME <- <CASE>).
#
# Usage:
#   scripts/run_training.sh <DATA_ROOT> <CASE> [--dataset meshy|geomhair]
#       [--exp_dir DIR] [--is_shrink True/False] [--use_3do_2do True/False]
#       [--use_old_dif_prior True/False] [--max_iter N]
#
# Example (the meshy reference, full run):
#   scripts/run_training.sh ~/geomhair_data/wv 000 --dataset meshy
#
set -e
set -o pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <DATA_ROOT> <CASE> [--dataset meshy|geomhair] [--exp_dir DIR] [--is_shrink True/False] [--use_3do_2do True/False] [--use_old_dif_prior True/False] [--max_iter N]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# vendored submodules importable from the repo root (NeuS, and LAVIS for the diffusion prior)
export PYTHONPATH="$REPO_ROOT/submodules:$REPO_ROOT/submodules/LAVIS:${PYTHONPATH}"

data_root="$1"
number="$2"
shift 2

DATASET=meshy
EXP_DIR="./exps_meshy"
IS_SHRINK=False              # meshy default: strips '_scaled' -> uses registration.ply
USE_3DO_2DO=False
USE_OLD_DIF_PRIOR=False
MAX_ITER=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)           DATASET="$2"; shift 2 ;;
        --exp_dir)           EXP_DIR="$2"; shift 2 ;;
        --is_shrink)         IS_SHRINK="$2"; shift 2 ;;
        --use_3do_2do)       USE_3DO_2DO="$2"; shift 2 ;;
        --use_old_dif_prior) USE_OLD_DIF_PRIOR="$2"; shift 2 ;;
        --max_iter)          MAX_ITER="$2"; shift 2 ;;
        *) echo "Invalid argument: $1"; exit 1 ;;
    esac
done

HAIR_CONF="./configs/example_config/base_${DATASET}.yaml"
echo "Dataset: $DATASET | data_root: $data_root | case: $number | hair_conf: $HAIR_CONF"

EXTRA=()
if [ -n "$MAX_ITER" ]; then
    EXTRA+=(--max_iter "$MAX_ITER")
fi

python run_strands_optimization.py \
    --case "$number" \
    --data_root "$data_root" \
    --scene_type monocular \
    --conf ./configs/monocular/neural_strands.yaml \
    --hair_conf "$HAIR_CONF" \
    --exp_name "${DATASET}_${number}" \
    --exp_basedir "$EXP_DIR" \
    --shrink "$IS_SHRINK" \
    --use_3do_2do "$USE_3DO_2DO" \
    --use_old_dif_prior "$USE_OLD_DIF_PRIOR" \
    --is_blobs False \
    --diffuse_iter 10000 \
    --n_diffuse_steps 2 \
    "${EXTRA[@]}"
