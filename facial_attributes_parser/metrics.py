# adapted from pytorch_toolbelt/utils/catalyst/metrics.py

from typing import List

import torch
import numpy as np


def binary_iou_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    intersection = torch.sum(y_pred * y_true).item()
    union = (torch.sum(y_pred) + torch.sum(y_true)).item() - intersection + 1e-7
    return intersection / union


def multiclass_iou_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> List[float]:
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    for cls_index in range(num_classes):
        y_pred_i = (y_pred == cls_index).float()
        y_true_i = (y_true == cls_index).float()

        iou = binary_iou_score(y_pred_i, y_true_i)
        ious.append(iou)

    return ious


class IoUMetric:
    __name__ = 'iou_score'

    def __init__(self):
        pass

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        batch_size = y_true.size(0)
        score_per_image = []
        for image_index in range(batch_size):
            score_per_class = multiclass_iou_score(y_pred[image_index], y_true[image_index])
            score_per_class = np.array(score_per_class).reshape(-1)
            score_per_image.append(score_per_class)

        mean_score = np.mean(score_per_image)
        return float(mean_score)
