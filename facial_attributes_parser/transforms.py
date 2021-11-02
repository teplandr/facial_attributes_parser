from typing import Dict, Tuple, Callable

import cv2
import numpy as np
import albumentations as albu

from facial_attributes_parser.dataset import CelebAMaskHQDataset


def visualization_transform(image: np.array, masks: Dict[int, np.array]) -> Tuple[np.array, np.array]:
    shape = 512, 512, 3
    result_mask = np.zeros(shape, dtype=np.uint8)
    for cls_index, mask in masks.items():
        result_mask[mask == 255] = CelebAMaskHQDataset.COLORS[cls_index]
    image = cv2.resize(image, result_mask.shape[:2])
    return image, result_mask


def inference_transform(image: np.array, masks: Dict[int, np.array]) -> Tuple[np.array, np.array]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # indices target (one-hotted in loss and metrics functions)
    result_mask = np.zeros((512, 512), dtype=np.long)
    for cls_index, mask in masks.items():
        result_mask[mask == 255] = cls_index

    return image, result_mask


def to_tensor(x: np.array, **kwargs) -> np.array:
    return x.transpose(2, 0, 1).astype(np.float32)


def get_preprocessing(preprocessing_fn: Callable) -> albu.Compose:
    transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
        albu.Lambda(mask=lambda x, **kwargs: x.astype(np.long)),
    ]
    return albu.Compose(transform)


def get_train_augmentations() -> albu.Compose:
    train_augs = [
        albu.Resize(height=512, width=512, p=1),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.6,
        ),
        albu.GaussNoise(p=0.2),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.6,
        ),
    ]
    return albu.Compose(train_augs)


def get_valid_augmentations() -> albu.Compose:
    valid_augs = [
        albu.Resize(height=512, width=512, p=1),
    ]
    return albu.Compose(valid_augs)
