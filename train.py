# train.py
import mamba_registry  # This registers everything
from ultralytics import YOLO, RTDETR
import torch

MODEL_NAME = "yolo26-v-mamba-attention"
DATA = "merged3v"

def benchmark():
    model = YOLO(f"/run/media/justas/Storage/Bakis/pretrained/yolo26-v-mamba3-mimo-merged3v2/weights/best.pt")
    # model = YOLO(f"yolo26-mamba.yaml")
    # model.info(verbose=True)
    # model.benchmark(format="-", data=f"{DATA}.yaml", device=0) # pytorch
    model.benchmark(format="-", data=f"merged3v.yaml", device=0) # onnx



if __name__ == '__main__':
    benchmark()
    # train_rf_detr()

    # model = YOLO(f"model-cfg/{MODEL_NAME}.yaml")
    # # model = YOLO(f"{MODEL_NAME}.yaml").load(f"./pretrained/{MODEL_NAME}/weights/best.pt")

    # results = model.train(
    #     name=f"{MODEL_NAME}-{DATA}",
    #     data=f"{DATA}.yaml", 
    #     epochs=100, 
    #     batch=32,
    #     # optimizer='SGD',
    #     # lr0=0.01, # 0.01 was ok
    #     device=0)
