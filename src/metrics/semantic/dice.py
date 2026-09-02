"""Coeficiente de Dice em pixels."""

import torch
import torch.nn as nn

from src.metrics.semantic.confusion import Confusion


class Dice(nn.Module):
    """Coeficiente de Dice: 2·TP / (2·TP + FP + FN), em pixels."""

    def __init__(self):
        super().__init__()
        self.confusion = Confusion()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        tp, fp, fn = self.confusion(pred, gt)
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            return 1.0
        return (2 * tp / denominator).item()

    def backward(self, *grad_outputs):
        raise NotImplementedError
