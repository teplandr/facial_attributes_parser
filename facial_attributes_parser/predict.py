from typing import Tuple, List
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from facial_attributes_parser.dataset import CelebAMaskHQDataset, inference_transform, get_preprocessing, COLORS


def prepare_image(image: torch.Tensor, mean: List[float], std: List[float]) -> np.array:
    # RGB -> BGR for OpenCV
    channels_order = [2, 1, 0]
    image = image[channels_order, :, :]

    std = torch.tensor(std).reshape(3, 1, 1)
    mean = torch.tensor(mean).reshape(3, 1, 1)
    image.mul_(std)
    image.add_(mean)
    # CHW -> HWC
    image = image.permute(1, 2, 0)
    return image.cpu().detach().numpy()


def prepare_mask(mask: torch.Tensor) -> np.array:
    mask = mask.cpu().detach().numpy()

    shape = *mask.shape, 3
    result_mask = np.zeros(shape, dtype=np.uint8)
    for cls_index, color in enumerate(COLORS):
        result_mask[mask == cls_index] = color
    return result_mask


def main():
    encoder_name = 'resnet34'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    params = smp.encoders.get_preprocessing_params(encoder_name, encoder_weights)
    mean = params["mean"]
    std = params["std"]

    model = torch.load('../checkpoints/best_model.pth', map_location=torch.device('cpu'))

    dataset = CelebAMaskHQDataset(
        Path('../data/CelebAMask-HQ'),
        (.9, .9002),
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for x, y_true in loader:
        y_pred = model.predict(x)
        _, indices = y_pred.squeeze().max(dim=0)

        x = prepare_image(x.squeeze(), mean, std)
        y_true = prepare_mask(y_true.squeeze())
        y_pred = prepare_mask(indices)

        cv2.imshow('x', x)
        cv2.imshow('y_true', y_true)
        cv2.imshow('y_pred', y_pred)
        cv2.waitKey()


if __name__ == '__main__':
    main()
