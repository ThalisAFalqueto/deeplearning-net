"""Conta pixels verdadeiro-positivos, falso-positivos e falso-negativos."""

import torch


class Confusion:
    """Conta pixels verdadeiro-positivos, falso-positivos e falso-negativos."""

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple:
        pred, gt = pred.bool(), gt.bool()
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        return (tp, fp, fn)
