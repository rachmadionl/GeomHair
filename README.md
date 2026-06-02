# GeomHair: Reconstruction of Hair Strands from Colorless 3D Scans

[**Paper**](https://arxiv.org/abs/2505.05376) | [**Project Page**](https://seva100.github.io/GeomHair/)

GeomHair reconstructs strand-based hairstyles from colorless 3D head scans. This repository contains the
full pipeline: a **preprocessing** stage that turns a raw scan into the geometry and orientation cues
needed for strand fitting, and a **strand-optimization** (training) stage that fits the strands using a
learned strand prior and a diffusion prior.

## Repository layout

```
preprocessing/         # the scan preprocessing step scripts (the pipeline)
  CrestCODE/            #   crest-line / curvature extraction (ships a setCurvature binary)
scripts/               # top-level entry points
  run_preprocessing.sh  #   data-agnostic preprocessing pipeline
  run_training.sh       #   strand optimization (training)
  run_all.sh            #   preprocessing followed by training
  download_pretrained.sh#   fetch the pretrained priors
submodules/            # external dependencies as git submodules
  LevelSetUDF/          #   surface (UDF/SDF) fitting
  LLaVA/                #   VQA hairstyle descriptions
  LAVIS/                #   BLIP-2 text-feature extraction
  TEED/                 #   edge detection
  NeuS/                 #   camera utilities (lift) + training
  k-diffusion/          #   diffusion prior (training)
src/                   # core library (models, datasets, hair networks, losses, utils)
configs/               # YAML configs
data/                  # shared priors (FLAME masks, scalp geometry, head priors)
```

## Getting started

```bash
git clone <repo-url> GeomHair
cd GeomHair
conda env create -n geomhair -f environment.yaml
conda activate geomhair
```

Initialize the submodules:

```bash
git submodule update --init submodules/LevelSetUDF submodules/LAVIS submodules/TEED submodules/NeuS submodules/LLaVA
```

Build the LevelSetUDF CUDA extension:

```bash
cd submodules/LevelSetUDF && python setup.py build_ext --inplace && cd ../..
```

The `preprocessing/CrestCODE/setCurvature` binary is shipped prebuilt; if it does not run on your
platform, rebuild it from `preprocessing/CrestCODE/CCode`.

## Pretrained models

The strand-optimization stage needs two pretrained priors. Download them into `pretrained_models/`:

```bash
pip install -U gdown
scripts/download_pretrained.sh
```

This fetches strand prior and the HAAR
diffusion prior, which we strip the latter only to the model weight so training startup
is fast and fits in modest RAM.

## Data layout

The pipeline operates on a dataset directory containing one folder per case:

```
<DATA_DIR>/<CASE>/000/
  scan.ply                      # raw scan geometry
  segmented_model/segmented.obj # semantic segmentation (geomhair only; meshy ships mesh.obj)
  flame.ply                     # FLAME fit to the scan
  registration.ply              # FLAME registration (head prior)
  camera_params.json            # camera intrinsics/extrinsics
```

Preprocessing writes its artifacts back into the same `000/` folder (`mesh.obj`, `mesh_remeshed.obj`,
`cutted_scalp.obj`, `dif_mask.png`, `rendered_mesh/`, `teed/`, `orientation_maps/`, `lifted_orients.ply`,
`orients_line_3_std_0.6.ply`, `dataset/features/00000/{frontal,back}.pt`, `ckpt_040000.pth`, ...).

## Running preprocessing

```bash
scripts/run_preprocessing.sh <DATA_DIR> <CASE> --dataset meshy
```

| Argument | Default | Description |
|---|---|---|
| `<DATA_DIR>` | — | root containing `<CASE>/000/` |
| `<CASE>` | — | case id (e.g. `0`) |
| `--dataset meshy\|geomhair` | `meshy` | `meshy` ships `mesh.obj` (extraction skipped); `geomhair` extracts the hair mesh from the scan |
| `--is_shrink True\|False` | `False` | also write a shrunk FLAME (`registration_scaled.ply`) |

The descriptions in `<CASE>/000/dataset/answers/` are produced by a LLaVA VQA step (it re-renders
`scan.obj` from the frontal/back views and asks a fixed set of questions); the BLIP step then embeds them.

The pipeline runs: hair-mesh extraction (geomhair only) → remeshing → 3D orientation (CrestCODE) → scalp
masking → 2D mesh rendering → TEED edge detection → 2D orientation maps → lifting to 3D → LLaVA hairstyle
descriptions → BLIP text features → LevelSetUDF surface fitting.

## Running training (strand optimization)

```bash
scripts/run_training.sh <DATA_ROOT> <CASE> --dataset meshy
```

| Argument | Default | Description |
|---|---|---|
| `<DATA_ROOT>` | — | root containing `<CASE>/000/` (substitutes `DATA_ROOT`/`CASE_NAME` in the configs) |
| `<CASE>` | — | case id (e.g. `000`) |
| `--dataset meshy\|geomhair` | `meshy` | selects `configs/example_config/base_<dataset>.yaml` |
| `--is_shrink True\|False` | `False` | `False` → `registration.ply`; `True` → `registration_scaled.ply` |
| `--use_3do_2do True\|False` | `False` | enable the rendering / 2D-orientation losses |
| `--use_old_dif_prior True\|False` | `False` | use the older diffusion prior |
| `--exp_dir DIR` | `./exps_<dataset>` | experiment output directory |
| `--max_iter N` | — | cap the number of iterations (e.g. for a quick smoke check) |

Outputs land in `./exps_<dataset>/<exp_name>/...` (`hair_primitives/ckpt_latest.pth`, strand point clouds
under `meshes/`).

## License

This project is released under the [MIT License](LICENSE.txt).

Note: the bundled 3D crest-line extractor (`preprocessing/CrestCODE`, by Yoshizawa et al.) is third-party
and may carry its own, non-MIT terms — see
https://www2.riken.jp/brict/Yoshizawa/Research/Crest.html.

## Links

GeomHair builds on and uses several great projects:

- [Neural Haircut](https://github.com/SamsungLabs/NeuralHaircut) — strand-based hair reconstruction;
- [NeuS](https://github.com/Totoro97/NeuS) — camera utilities / geometry;
- [LevelSetUDF](https://github.com/rachmadionl/LevelSetUDF) — surface fitting;
- [HAAR](https://github.com/Vanessik/HAAR) — strand-based hair diffusion prior;
- [k-diffusion](https://github.com/crowsonkb/k-diffusion) — diffusion sampling;
- [LLaVA](https://github.com/haotian-liu/LLaVA) — VQA hairstyle descriptions;
- [LAVIS](https://github.com/salesforce/LAVIS) — BLIP-2 text features;
- [TEED](https://github.com/xavysp/TEED) — edge detection;
- [CrestCODE](https://www2.riken.jp/brict/Yoshizawa/Research/Crest.html) (Yoshizawa et al.) — crest-line extraction.

## Citation

```
@misc{lazuardi2026geomhairreconstructionhairstrands,
      title={GeomHair: Reconstruction of Hair Strands from Colorless 3D Scans},
      author={Rachmadio Noval Lazuardi and Artem Sevastopolsky and Egor Zakharov and Matthias Niessner and Vanessa Sklyarova},
      year={2026},
      eprint={2505.05376},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05376},
}
```
