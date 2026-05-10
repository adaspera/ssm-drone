# Drone Detection using State Space Models

Experiments for drone detection with YOLO26, VMamba, and Mamba-based variants.

## Main entrypoints

- `python train.py` trains the YOLO26 + VMamba/Mamba models.
- `python rf_detr_train.py` runs the RF-DETR baseline utilities.

## Notes

- `mamba_ssm` lib in this workspace is expected to build against CUDA 12.8.
- The local Mamba3 code lives in `libs/mamba`. If the installed `mamba_ssm` package at the writing of this does not expose Mamba3 ops, symlink the `mamba3` and `tilelang` ops from `libs/mamba/mamba_ssm/ops/` into the active environment. Or install it as an editable dependancy (didn't get to work due to cuda mismatch)

## Layout

- `model-cfg/` model YAML files
- `data/` datasets
- `pretrained/` saved runs and weights
- `libs/` local modified copies of VMamba and Ultralytics and vanila Mamba (mamba_ssm)