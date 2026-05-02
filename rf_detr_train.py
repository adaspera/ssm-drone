from rfdetr import RFDETRNano


def train_rf_detr():
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

if __name__ == '__main__':

    train_rf_detr()