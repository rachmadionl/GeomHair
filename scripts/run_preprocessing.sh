#!/bin/bash
#
# GeomHair preprocessing pipeline (data-agnostic).
#
# Turns a raw scan into the geometry + orientation cues needed for strand
# optimization. The only dataset-dependent step is hair-mesh extraction
# (step 0): "meshy" data already ships mesh.obj, while "geomhair" data is
# extracted from the scan. Everything else is shared.
#
# Usage:
#   scripts/run_preprocessing.sh <DATA_DIR> <CASE> [--dataset meshy|geomhair]
#                                [--is_shrink True/False]
#
set -e
set -o pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <DATA_DIR> <CASE> [--dataset meshy|geomhair] [--is_shrink True/False]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# make the vendored submodules (e.g. NeuS) importable from the repo root
export PYTHONPATH="$REPO_ROOT/submodules:${PYTHONPATH}"

folder="$1"
number="$2"
shift 2

DATASET=meshy
IS_SHRINK=False

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --is_shrink)
            if [[ "${2,,}" == "true" ]]; then IS_SHRINK=True; fi; shift 2 ;;
        *)
            echo "Invalid argument: $1"; exit 1 ;;
    esac
done

echo "Dataset: $DATASET | case: $number | shrink: $IS_SHRINK"

# 0. Extract hair mesh -> mesh.obj  (geomhair only; meshy already provides mesh.obj)
if [ "$DATASET" = "geomhair" ]; then
    python preprocessing/extract_hair_mesh.py --dir "$folder" --case "$number" --dataset geomhair
    echo "Extract Hair Mesh: DONE"
fi

# 1. Remesh hair mesh -> mesh_remeshed.obj
python preprocessing/remeshing_hair_mesh.py --dir "$folder" --case "$number"
echo "Remesh Hair Mesh: DONE"

# 2. 3D orientations via CrestCODE -> orients_line_3_std_0.6.ply
python preprocessing/get_orientations.py --dir "$folder" --case "$number"
echo "Get 3D orientations: DONE"

# 3. Cut the scalp -> cutted_scalp.obj, dif_mask.png, cut_scalp_verts.pickle, labeled_flame.ply
if [ "$IS_SHRINK" = "True" ]; then
    python preprocessing/mask_scalp.py --dir "$folder" --case "$number" --scalp_data_path ./data --shrink True
else
    python preprocessing/mask_scalp.py --dir "$folder" --case "$number" --scalp_data_path ./data
fi
echo "Cutting Scalp: DONE"

# 4. Render hair mesh into 2D multi-view images -> rendered_mesh/, camera_params.json
python preprocessing/render_mesh_2d.py --case "$number" --dir "$folder"
echo "Render mesh into 2D: DONE"

# 5. TEED edge detection -> teed/
python preprocessing/run_teed.py --choose_test_data=-1 --dir "$folder" --case "$number"
echo "Running TEED: DONE"

# 6. 2D orientation maps -> orientation_maps/, longest_path/, quiver_plot/
python preprocessing/calc_orientation_maps_via_longest_path.py --case "$number" --dir "$folder"
echo "Calculate 2D orientations: DONE"

# 7. Lift 2D orientations into 3D -> lifted_orients.ply
python preprocessing/lift_orientations_2d_pcd.py --case "$number" --dir "$folder"
echo "Lift 2D Orientations into 3D: DONE"

# 8. Hairstyle descriptions via LLaVA VQA (render scan.obj frontal/back, ask QUESTIONS) -> dataset/answers
python preprocessing/obtain_hairstyle_descriptions.py --dir "$folder" --case "$number"
echo "Obtain hairstyle descriptions (LLaVA VQA): DONE"

# 9. BLIP text features (from dataset/answers) -> dataset/features/<idx>/{frontal,back}.pt
python preprocessing/calc_text_emb.py --dir "$folder" --case "$number"
echo "Obtain BLIP features: DONE"

# 10. Fit the surface (LevelSetUDF) -> ckpt_040000.pth, meshes/, mesh.npz
python submodules/LevelSetUDF/run.py \
    --gpu 0 \
    --conf submodules/LevelSetUDF/confs/object.conf \
    --dir "$folder" \
    --case "$number"
echo "Extract SDF: DONE"

echo "PREPROCESSING: DONE"
