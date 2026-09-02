"""Precisão média num único limiar: TP / (TP + FP + FN)."""

import torch
import torch.nn as nn

from src.metrics.instance.counts_at_threshold import CountsAtThreshold


class AveragePrecision(nn.Module):
    """Precisão média num único limiar: TP / (TP + FP + FN)."""

    def __init__(self, matcher: nn.Module = None):
        super().__init__()
        self.counts = CountsAtThreshold(matcher)

    def forward(self, iou: torch.Tensor, threshold: float) -> float:
        tp, fp, fn = self.counts(iou, threshold)
        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0
        return tp / denominator

    def backward(self, *grad_outputs):
        raise NotImplementedError
