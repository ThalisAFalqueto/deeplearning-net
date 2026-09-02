"""Média de average_precision sobre os 10 limiares de IoU."""

import torch
import torch.nn as nn

from src.metrics.instance.iou_matrix import IoUMatrix
from src.metrics.instance.average_precision import AveragePrecision


THRESHOLDS = torch.round(torch.arange(0.50, 1.00, 0.05), decimals=2)


class MeanAveragePrecision(nn.Module):
    """Média de average_precision sobre THRESHOLDS."""

    def __init__(self, matcher: nn.Module = None):
        super().__init__()
        self.iou_matrix = IoUMatrix()
        self.ap = AveragePrecision(matcher)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple:
        iou = self.iou_matrix(pred, gt)
        dict_thresholds = {}
        for th in THRESHOLDS:
            ap = self.ap(iou, th.item())
            dict_thresholds[th.item()] = ap

        values = torch.tensor(list(dict_thresholds.values()), dtype=torch.float64)
        map_val = values.mean().item()
        return (map_val, dict_thresholds)

    def backward(self, *grad_outputs):
        raise NotImplementedError
