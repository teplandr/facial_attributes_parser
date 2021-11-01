import os
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Callable

import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset


def visualization_transform(image: np.array, masks: Dict[int, np.array]) -> Tuple[np.array, np.array]:
    shape = 512, 512, 3
    result_mask = np.zeros(shape, dtype=np.uint8)
    for cls_index, mask in masks.items():
        result_mask[mask == 255] = CelebAMaskHQDataset.COLORS[cls_index]
    image = cv2.resize(image, result_mask.shape[:2])
    return image, result_mask


def inference_transform(image: np.array, masks: Dict[int, np.array]) -> Tuple[np.array, np.array]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # temp
    image = cv2.resize(image, (512, 512))

    # one-hot target
    # shape = 18, *masks[0].shape
    # result_mask = np.zeros(shape, dtype=np.float32)
    # for cls_index, mask in masks.items():
    #     result_mask[cls_index][mask == 255] = 1

    # indices target (one-hotted in loss and metrics functions)
    result_mask = np.zeros((512, 512), dtype=np.long)
    for cls_index, mask in masks.items():
        result_mask[mask == 255] = cls_index

    return image, result_mask


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype(np.float32)


def get_preprocessing(preprocessing_fn):
    transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(transform)


class CelebAMaskHQDataset(Dataset):

    COLORS = [(0, 0, 0), (204, 0, 0), (76, 153, 0), (204, 204, 0),
              (51, 51, 255), (204, 0, 204), (0, 255, 255),
              (255, 204, 204), (102, 51, 0), (255, 0, 0),
              (102, 204, 0), (255, 255, 0), (0, 0, 153),
              (0, 0, 204), (255, 51, 153), (0, 204, 204),
              (0, 51, 0), (255, 153, 51), (0, 204, 0)]
    CLASSES = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g',
               'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip',
               'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    SAMPLES_PER_DIR = 2000

    def __init__(
            self,
            root_path: Path,
            interval: Tuple[float, float],
            transform: Callable,
            augmentation=None,
            preprocessing=None
    ) -> None:
        self.transform = transform
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        assert os.path.exists(root_path)

        self.image_paths = glob(f'{root_path}/CelebA-HQ-img/*')
        # optional
        self.image_paths.sort(key=lambda x: int(Path(x).stem))

        start = int(interval[0] * len(self.image_paths))
        stop = int(interval[1] * len(self.image_paths))

        self.image_paths = self.image_paths[start:stop]

        self.mask_paths = []
        for image_path in self.image_paths:
            image_index = int(Path(image_path).stem)
            current_masks = dict()
            for cls_index, cls in enumerate(self.CLASSES, start=1):
                mask_path = f'{root_path}/CelebAMask-HQ-mask-anno/{image_index // self.SAMPLES_PER_DIR}/' \
                            f'{image_index:05}_{cls}.png'
                if os.path.exists(mask_path):
                    current_masks[cls_index] = mask_path
            self.mask_paths.append(current_masks)

        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        masks = dict()
        for cls_index, mask_path in self.mask_paths[index].items():
            masks[cls_index] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # visualization or inference
        image, mask = self.transform(image, masks)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
