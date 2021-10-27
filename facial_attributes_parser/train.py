import argparse
from pathlib import Path

import yaml
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from facial_attributes_parser.dataset import CelebAMaskHQDataset, inference_transform, get_preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=19,
        activation="softmax2d"
    )
    loss = smp.utils.losses.CrossEntropyLoss()
    metrics = []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["lr"])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

    train_dataset = CelebAMaskHQDataset(
        Path(hparams["dataset_root_path"]),
        hparams["train_interval"],
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    valid_dataset = CelebAMaskHQDataset(
        Path(hparams["dataset_root_path"]),
        hparams["valid_interval"],
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=hparams["batch_size"],
                              shuffle=False,
                              num_workers=hparams["num_workers"])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=hparams["batch_size"],
                              shuffle=False,
                              num_workers=hparams["num_workers"])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    checkpoint_path = Path(hparams["checkpoint_path"])
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    # max_score = 0.
    for i in range(hparams["num_epochs"]):
        print(f"\nEpoch: {i}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # if max_score < valid_logs["iou_score"]:
        #     max_score = valid_logs["iou_score"]
    torch.save(model, checkpoint_path)
    print("Model saved!")


if __name__ == "__main__":
    main()

