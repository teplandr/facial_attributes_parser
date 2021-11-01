import argparse
from typing import List
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from facial_attributes_parser.dataset import CelebAMaskHQDataset
from facial_attributes_parser.transforms import inference_transform, get_preprocessing, get_valid_augmentations


def prepare_image(image: torch.Tensor, mean: List[float], std: List[float]) -> np.array:
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
    for cls_index, color in enumerate(CelebAMaskHQDataset.COLORS):
        result_mask[mask == cls_index] = color
    return result_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name", type=str, help="Name of the SMP encoder.", required=True)
    parser.add_argument("--encoder_weights", type=str, help="Name of the SMP encoder weights.", required=True)
    parser.add_argument("--model_path", type=Path, help="Path to the pretrained model.", required=True)
    parser.add_argument("--data_root_path", type=Path, help="Path to the CelebAMask-HQ root.", required=True)
    parser.add_argument("--subset_range", type=tuple, help="Range of samples to process.", default=(0.9999, 1.0))
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, args.encoder_weights)
    params = smp.encoders.get_preprocessing_params(args.encoder_name, args.encoder_weights)
    mean = params["mean"]
    std = params["std"]

    model = torch.load(args.model_path, map_location=device)

    dataset = CelebAMaskHQDataset(
        args.data_root_path,
        args.subset_range,
        transform=inference_transform,
        preprocessing=get_preprocessing(preprocessing_fn),
        augmentation=get_valid_augmentations()
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for x, y_true in loader:
        y_pred = model.predict(x)
        indices = y_pred.squeeze().argmax(dim=0)

        x = prepare_image(x.squeeze(), mean, std)
        y_true = prepare_mask(y_true.squeeze())
        y_pred = prepare_mask(indices)

        plt.figure(figsize=(16, 5))

        plt.subplot(131)
        plt.imshow(x)
        plt.axis('off')
        plt.title('Input Image')

        plt.subplot(132)
        plt.imshow(y_true)
        plt.axis('off')
        plt.title('Ground Truth')

        plt.subplot(133)
        plt.imshow(y_pred)
        plt.title('Prediction')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
