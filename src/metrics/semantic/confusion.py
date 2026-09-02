"""Conta pixels verdadeiro-positivos, falso-positivos e falso-negativos."""

import torch
import torch.nn as nn


class Confusion(nn.Module):
    """Conta pixels verdadeiro-positivos, falso-positivos e falso-negativos."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple:
        pred, gt = pred.bool(), gt.bool()
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        return (tp, fp, fn)

    def backward(self, *grad_outputs):
        raise NotImplementedError
