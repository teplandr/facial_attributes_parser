from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from facial_attributes_parser.dataset import CelebAMaskHQDataset, inference_transform, get_preprocessing


def main():
    encoder_name = 'resnet34'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

    model = torch.load('../checkpoints/best_model.pth', map_location=torch.device('cpu'))

    dataset = CelebAMaskHQDataset(
        Path('../data/CelebAMask-HQ'),
        (.9, .9002),
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for x, y_true in loader:
        y_pred = model.forward(x)

        x = x.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        y_true = y_true.squeeze().cpu().detach().numpy()
        y_pred = y_pred.squeeze().cpu().detach().numpy()

        cv2.imshow('x', x)
        for i in range(19):
            cv2.imshow('y_true', np.where(y_true == i, 255, 0).astype(np.uint8))
            cv2.imshow('y_pred', y_pred[i])
            cv2.waitKey()


if __name__ == '__main__':
    main()
