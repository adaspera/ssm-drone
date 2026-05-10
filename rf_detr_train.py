from rfdetr import RFDETRNano


def train_rf_detr():
    from roboflow import Roboflow
    
    # rf = Roboflow(api_key="YOUR_API_KEY")
    # project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
    # dataset = project.version(1).download("coco")

    model = RFDETRNano(pretrain_weights=None)
    model.train(
        dataset_dir='./data/mergedv3',
        epochs=50,
        batch_size=8,
        grad_accum_steps=2,
        # lr=1e-4,
        output_dir='./runs/rfdetr',
        num_workers=0,
        tensorboard=True
    )

def convert_to_coco():
    import supervision as sv

    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path="./data/mergedv3/train/images",
        annotations_directory_path="./data/mergedv3/train/labels",
        data_yaml_path="./data/mergedv3/data.yaml"
    )

    # Save as COCO
    dataset.as_coco(
        images_directory_path="./data/mergedv3-cocostyle/train",
        annotations_path="./data/mergedv3-cocostyle/train/_annotations.coco.json"
    )

def benchmark(weights_path: str, image_path: str, warmup: int = 50, runs: int = 1000):
    import time
    from PIL import Image

    model = RFDETRNano(pretrain_weights=weights_path)
    model.optimize_for_inference()
    image = Image.open(image_path).convert("RGB")

    for _ in range(warmup):
        model.predict(image)

    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(image)
    elapsed = (time.perf_counter() - t0) * 1000 / runs

    print(f"{elapsed:.2f} ms  |  {1000 / elapsed:.2f} FPS")


if __name__ == '__main__':

    # train_rf_detr()

    benchmark(
        weights_path='./pretrained/rfdetr-coco/checkpoint_best_total.pth',
        image_path='./gpu_log_yolo26n_plot.png')