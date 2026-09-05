"""Erro absoluto de contagem de objetos."""

import torch


class CountError:
    """Erro absoluto de contagem de objetos entre predição e gabarito."""

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> int:
        pred_unq = torch.unique(pred[pred != 0])
        gt_unq = torch.unique(gt[gt != 0])
        return abs(len(pred_unq) - len(gt_unq))
