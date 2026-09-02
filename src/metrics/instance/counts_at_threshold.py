"""Conta TP/FP/FN num limiar de IoU."""

import torch
import torch.nn as nn

from src.metrics.instance.match_greedy import MatchGreedy


class CountsAtThreshold(nn.Module):
    """Conta objetos verdadeiro-positivos, falso-positivos e falso-negativos."""

    def __init__(self, matcher: nn.Module = None):
        super().__init__()
        self.matcher = matcher if matcher is not None else MatchGreedy()

    def forward(self, iou: torch.Tensor, threshold: float) -> tuple:
        pares = self.matcher(iou, threshold)
        tp = len(pares)
        fp = iou.shape[0] - tp
        fn = iou.shape[1] - tp
        return (tp, fp, fn)

    def backward(self, *grad_outputs):
        raise NotImplementedError
