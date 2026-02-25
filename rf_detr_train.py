from rfdetr import RFDETRNano


def train_rf_detr():
    model = RFDETRNano(pretrain_weights=None)
    model.train(
        dataset_dir='./data/mergedv2-cocostyle',
        epochs=50,
        batch_size=8,
        grad_accum_steps=2,
        # lr=1e-4,
        output_dir='./runs/rfdetr',
        num_workers=0,
        tensorboard=True
    )

if __name__ == '__main__':

    train_rf_detr()