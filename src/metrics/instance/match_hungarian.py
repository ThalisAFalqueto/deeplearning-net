"""Matching ótimo de objetos (algoritmo Húngaro)."""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class MatchHungarian(nn.Module):
    """Casa objetos maximizando a soma total de IoU (atribuição ótima)."""

    def forward(self, iou: torch.Tensor, threshold: float) -> list:
        rows, cols = linear_sum_assignment(-iou.numpy())
        pares = []
        for indice in range(len(rows)):
            i, j = rows[indice], cols[indice]
            value = iou[i, j]
            if value >= threshold:
                pares.append((i, j))
        return pares

    def backward(self, *grad_outputs):
        raise NotImplementedError
