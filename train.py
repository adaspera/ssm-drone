# train.py
import mamba_registry  # This registers everything
from ultralytics import YOLO, RTDETR
from gpu_monitor import GPUMonitor
import torch

MODEL_NAME = "yolo26-v-mamba-attention"
DATA = "merged3v"

PRETRAINED_MODELS = [
    "yolo26-v-mamba3-attention-coco",
]

PRETRAINED_BASE = "/run/media/justas/Storage/Bakis/pretrained"

def benchmark():
    model = YOLO(f"{PRETRAINED_BASE}/yolo26-v-mamba3-merged3v3/weights/best.pt")
    # model = YOLO(f"yolo26-mamba.yaml")
    # model.info(verbose=True)
    # model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # pytorch
    model.benchmark(format="-", data=f"merged3v.yaml", device=0) # onnx

def benchmark_all():
    for name in PRETRAINED_MODELS:
        weights = f"{PRETRAINED_BASE}/{name}/weights/best.pt"
        try:
            model = YOLO(weights)
            model.benchmark(format="-", data="merged3v.yaml", device=0)
        except Exception as e:
            print(f"ERROR benchmarking {name}: {e}")

def monitor():
    model = YOLO(f"model-cfg/{MODEL_NAME}.yaml")
    
    monitor = GPUMonitor(out_path=f"gpu_log_{MODEL_NAME}.csv", interval=0.5, gpu_idx=0)
    monitor.start()
    try:
        model.train(
            name=f"{MODEL_NAME}-{DATA}",
            data=f"{DATA}.yaml",
            epochs=1,
            batch=32,
            save=False,
            device=0)
    finally:
        monitor.stop()


if __name__ == '__main__':
    # benchmark()
    benchmark_all()
    # monitor()

    # model = YOLO(f"model-cfg/{MODEL_NAME}.yaml")
    # # model = YOLO(f"{MODEL_NAME}.yaml").load(f"./pretrained/{MODEL_NAME}/weights/best.pt")

    # results = model.train(
    #     name=f"{MODEL_NAME}-{DATA}",
    #     data=f"{DATA}.yaml",
    #     epochs=1,
    #     batch=32,
    #     # optimizer='SGD',
    #     # lr0=0.01, # 0.01 was ok
    #     device=0)
