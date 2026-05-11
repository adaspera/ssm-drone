# train.py
import mamba_registry  # This registers everything
from ultralytics import YOLO, RTDETR
import torch

MODEL_NAME = "yolo26-v-mamba3-mimo"
DATA = "merged3v"

PRETRAINED_MODELS = [
    "yolo26-v-mamba2-attention-merged3v",
]


def benchmark():
    model = YOLO(f"./pretrained/yolo26-v-mamba2-merged3v2/weights/best.pt")
    # model = YOLO(f"yolo26-mamba.yaml")
    # model.info(verbose=True)
    # model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # pytorch
    model.benchmark(format="-", data=f"merged3v.yaml", device=0) # onnx

def benchmark_all():
    for name in PRETRAINED_MODELS:
        weights = f"./pretrained/{name}/weights/best.pt"
        try:
            model = YOLO(weights)
            model.benchmark(format="-", data="merged3v.yaml", device=0)
        except Exception as e:
            print(f"ERROR benchmarking {name}: {e}")


if __name__ == '__main__':
    # benchmark()
    # benchmark_all()
    # monitor()

    model = YOLO(f"model-cfg/{MODEL_NAME}.yaml")
    # model = YOLO(f"{MODEL_NAME}.yaml").load(f"./pretrained/{MODEL_NAME}/weights/best.pt")

    results = model.train(
        name=f"{MODEL_NAME}-{DATA}",
        data=f"{DATA}.yaml",
        epochs=2,
        batch=16,
        # optimizer='SGD',
        # lr0=0.01, # 0.01 was ok
        device=0)