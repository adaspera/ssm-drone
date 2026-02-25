
### mamba-ssm compiles only on cuda 12.8

conda activate mamba-env

conda install cuda-toolkit=12.8 -c nvidia

# Launch

cd /run/media/justas/Windows/Users/jusci/My\ Documents/Justo/VU/Bakis/

conda activate mamba-env

source "./venvlx/bin/activate"

python train.py


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


