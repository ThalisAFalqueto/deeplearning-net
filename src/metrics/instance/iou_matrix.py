"""Calcula o IoU entre todos os pares de objetos."""

import torch
import torch.nn as nn


class IoUMatrix(nn.Module):
    """Calcula o IoU entre todos os pares de objetos."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_unq = torch.unique(pred[pred != 0])
        gt_unq = torch.unique(gt[gt != 0])
        len_pred, len_gt = len(pred_unq), len(gt_unq)
        matrix = torch.zeros((len_pred, len_gt))

        for indice_pred, label_pred in enumerate(pred_unq):
            for indice_gt, label_gt in enumerate(gt_unq):
                intersec = ((pred == label_pred) & (gt == label_gt)).sum()
                union = (pred == label_pred).sum() + (gt == label_gt).sum() - intersec
                iou = intersec / union
                matrix[indice_pred, indice_gt] = iou

        return matrix

    def backward(self, *grad_outputs):
        raise NotImplementedError
