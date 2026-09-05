"""Intersection over Union em pixels."""

import torch

from src.metrics.semantic.confusion import Confusion


class IoU:
    """Intersection over Union: TP / (TP + FP + FN), em pixels."""

    def __init__(self):
        self.confusion = Confusion()

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        tp, fp, fn = self.confusion(pred, gt)
        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0
        return (tp / denominator).item()
