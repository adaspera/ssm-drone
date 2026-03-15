# train.py
import mamba_registry  # This registers everything
from ultralytics import YOLO
import torch

MODEL_NAME = "yolo26n-v-mamba2-rope"
DATA = "merged3v"

def benchmark():
    model = YOLO(f"./runs/detect/{MODEL_NAME}-{DATA}3/weights/best.pt")
    # model = YOLO(f"yolo26-mamba.yaml")
    # model.info(verbose=True)
    # model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # pytorch
    model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # onnx



if __name__ == '__main__':
    # benchmark()
    # train_rf_detr()

    model = YOLO(f"model-cfg/{MODEL_NAME}.yaml")
    # model = YOLO(f"{MODEL_NAME}.yaml").load(f"./pretrained/{MODEL_NAME}/weights/best.pt")

    results = model.train(
        name=f"{MODEL_NAME}-{DATA}",
        data=f"{DATA}.yaml", 
        epochs=50, 
        batch=8,
        # optimizer='SGD',
        # lr0=0.01, # 0.01 was ok
        device=0)


# TODO:
# 2. Complex-valued state — can be approximated outside the kernel
# The kernel signature takes real tensors. However, complex multiplication (a + bi)(c + di) can be decomposed into real operations. You can double the state size (d_state *= 2) and implement the complex rotation manually in Python before/after the scan — the "RoPE-like" formulation the paper describes maps cleanly to this. No kernel change needed, just a wrapper.