from pathlib import Path

import cv2

from facial_attributes_parser.dataset import CelebAMaskHQDataset, visualization_transform, inference_transform


def visualize_dataset():
    dataset = CelebAMaskHQDataset(Path('./data/CelebAMask-HQ'), (.01, .02), visualization_transform)
    print('Len:', len(dataset))
    for i in range(100):
        image, mask = dataset[i]
        cv2.imshow('Image', image)
        cv2.imshow('Mask', mask)
        cv2.waitKey()


def check_inference_transform():
    dataset = CelebAMaskHQDataset(Path('./data/CelebAMask-HQ'), (.01, .02), inference_transform)
    image, mask = dataset[0]
    print(image.shape, mask.shape)
    cv2.imshow('Test', mask[16])
    cv2.waitKey()


if __name__ == '__main__':
    visualize_dataset()
