"""Conta TP/FP/FN num limiar de IoU."""

import torch

from src.metrics.instance.match_greedy import MatchGreedy


class CountsAtThreshold:
    """Conta objetos verdadeiro-positivos, falso-positivos e falso-negativos."""

    def __init__(self, matcher=None):
        self.matcher = matcher if matcher is not None else MatchGreedy()

    def __call__(self, iou: torch.Tensor, threshold: float) -> tuple:
        pares = self.matcher(iou, threshold)
        tp = len(pares)
        fp = iou.shape[0] - tp
        fn = iou.shape[1] - tp
        return (tp, fp, fn)
