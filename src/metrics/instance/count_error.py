"""Erro absoluto de contagem de objetos."""

import torch
import torch.nn as nn


class CountError(nn.Module):
    """Erro absoluto de contagem de objetos entre predição e gabarito."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> int:
        pred_unq = torch.unique(pred[pred != 0])
        gt_unq = torch.unique(gt[gt != 0])
        return abs(len(pred_unq) - len(gt_unq))

    def backward(self, *grad_outputs):
        raise NotImplementedError
