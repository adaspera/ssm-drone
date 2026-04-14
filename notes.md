
### mamba-ssm compiles only on cuda 12.8

conda activate mamba-env

conda install cuda-toolkit=12.8 -c nvidia

which nvcc

pip install setuptools wheel

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

CUDA_HOME=/home/justas/miniconda3/envs/mamba-env MAX_JOBS=2 pip install --no-build-isolation -e ./libs/mamba

# Launch

cd /run/media/justas/Windows/Users/jusci/My\ Documents/Justo/VU/Bakis/

conda activate mamba-env

source "./venv/bin/activate"

python train.py


# Mamba3 support

The installed `mamba_ssm` (v2.3.1) doesn't include Mamba-3 ops. Symlink them from `libs/mamba`:

```
ln -s /run/media/justas/Storage/Bakis/libs/mamba/mamba_ssm/ops/triton/mamba3 \
      ./venv/lib/python3.14/site-packages/mamba_ssm/ops/triton/mamba3
```

This makes `from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined` work
without rebuilding the entire package (which requires CUDA at build time).

Or just install it as an editable code with pip -e .
idk its your life


# RF_DETR 

source "./venvrfdetr/bin/activate"

### rfdetr is written on older transformers lib:

pip install rfdetr

pip install "transformers<5.0" 

### validation bug:

/rfdetr/engine.py line 187

```
iou50_idx, area_idx, maxdet_idx = (
    int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)
 ^
 |
iou50_idx, area_idx, maxdet_idx = (
    int(np.argwhere(np.isclose(iou_thrs, 0.50))[0, 0]), 0, 2)
```

### TensorBoard

```
tensorboard --logdir ./runs/rfdetr
```


### File transfer

rsync -avz --progress /run/media/justas/Storage/Bakis/data/mergedv3/ bakis-vast:/root/workspace/ssm-drone/data/mergedv3/