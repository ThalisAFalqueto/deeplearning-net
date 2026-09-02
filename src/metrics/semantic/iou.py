"""Intersection over Union em pixels."""

import torch
import torch.nn as nn

from src.metrics.semantic.confusion import Confusion


class IoU(nn.Module):
    """Intersection over Union: TP / (TP + FP + FN), em pixels."""

    def __init__(self):
        super().__init__()
        self.confusion = Confusion()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        tp, fp, fn = self.confusion(pred, gt)
        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0
        return (tp / denominator).item()

    def backward(self, *grad_outputs):
        raise NotImplementedError
