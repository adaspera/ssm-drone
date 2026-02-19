# train.py
import mamba_registry  # This registers everything
from ultralytics import YOLO
import torch

MODEL_NAME = "yolo26-v-mamba"
DATA = "data"

def benchmark():
    model = YOLO(f"./runs/detect/yolo26-v-mamba-noC3k2-data/weights/best.pt")
    # model = YOLO(f"yolo26-mamba.yaml")
    # model.info(verbose=True)
    model.benchmark(format="-", data="data.yaml", device=0) # pytorch
    # model.benchmark(format="onnx", data="data.yaml", device=0) # onnx


if __name__ == '__main__':
    # benchmark()


    model = YOLO(f"{MODEL_NAME}.yaml")

    results = model.train(
        name=f"{MODEL_NAME}-{DATA}",
        data=f"{DATA}.yaml", 
        epochs=50, 
        batch=16,
        optimizer='SGD',
        lr0=0.01,
        device=0,
        amp=True)
