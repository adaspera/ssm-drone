# train.py
import mamba_registry  # This registers everything
from ultralytics import YOLO
import torch

MODEL_NAME = "yolo26-v-mamba-pure"
DATA = "merged3v"

def benchmark():
    model = YOLO(f"./pretrained/{MODEL_NAME}-{DATA}/weights/best.pt")
    # model = YOLO(f"yolo26-mamba.yaml")
    # model.info(verbose=True)
    # model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # pytorch
    model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # onnx



if __name__ == '__main__':
    # benchmark()
    # train_rf_detr()

    # model = YOLO(f"model-cfg/{MODEL_NAME}.yaml")
    # model = YOLO(f"{MODEL_NAME}.yaml").load(f"./pretrained/{MODEL_NAME}/weights/best.pt")

    
    MODEL_NAME = "wang2024-Mamba-YOLO-T"
    DATA = "merged3v"

    model = YOLO(f"{MODEL_NAME}.yaml") 
    # model = YOLO(f"{MODEL_NAME}.yaml").load(f"./pretrained/{MODEL_NAME}/weights/best.pt")

    results = model.train(
        name=f"{MODEL_NAME}-{DATA}",
        data=f"{DATA}.yaml", 
        epochs=50, 
        batch=32,
        optimizer='SGD',
        lr0=0.005, # 0.01 was ok
        device=0)
