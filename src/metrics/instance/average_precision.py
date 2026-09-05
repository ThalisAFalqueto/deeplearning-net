"""Precisão média num único limiar: TP / (TP + FP + FN)."""

import torch

from src.metrics.instance.counts_at_threshold import CountsAtThreshold


class AveragePrecision:
    """Precisão média num único limiar: TP / (TP + FP + FN)."""

    def __init__(self, matcher=None):
        self.counts = CountsAtThreshold(matcher)

    def __call__(self, iou: torch.Tensor, threshold: float) -> float:
        tp, fp, fn = self.counts(iou, threshold)
        denominator = tp + fp + fn
        if denominator == 0:
            return 1.0
        return tp / denominator
