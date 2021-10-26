from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataset import CelebAMaskHQDataset, inference_transform, get_preprocessing


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder_name = 'resnet34'
    encoder_weights = 'imagenet'

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=19,
        activation='softmax2d'
    )
    loss = smp.utils.losses.DiceLoss(eps=1e-7)
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

    train_dataset = CelebAMaskHQDataset(
        Path('../data/CelebAMask-HQ'),
        (.0, .0002),
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    valid_dataset = CelebAMaskHQDataset(
        Path('../data/CelebAMask-HQ'),
        (.1, .1002),
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

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

    max_score = 0
    for i in range(40):
        print(f'\nEpoch: {i}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, '../checkpoints/best_model.pth')
            print('Model saved!')


if __name__ == '__main__':
    main()
