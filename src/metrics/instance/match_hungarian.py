"""Matching ótimo de objetos (algoritmo Húngaro)."""

import torch
from scipy.optimize import linear_sum_assignment


class MatchHungarian:
    """Casa objetos maximizando a soma total de IoU (atribuição ótima)."""

    def __call__(self, iou: torch.Tensor, threshold: float) -> list:
        rows, cols = linear_sum_assignment(-iou.numpy())
        pares = []
        for indice in range(len(rows)):
            i, j = rows[indice], cols[indice]
            value = iou[i, j]
            if value >= threshold:
                pares.append((i, j))
        return pares
