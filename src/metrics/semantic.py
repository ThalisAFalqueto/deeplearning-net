"""Métricas semânticas sobre a máscara binária.

Comparam duas máscaras binárias da imagem inteira e ignoram a divisão em objetos.
"""

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray):
    """Conta pixels verdadeiro-positivos, falso-positivos e falso-negativos.

    Args:
        pred: máscara binária prevista.
        gt: máscara binária verdadeira. (ground truth)

    Returns:
        Tupla (tp, fp, fn) de inteiros.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return (tp, fp, fn)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union: TP / (TP + FP + FN), em pixels."""
    tp, fp, fn = confusion(pred, gt)
    denominador = tp + fp + fn
    if denominador == 0:  # significa que tudo é fundo
        return 1
    return tp / denominador


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Coeficiente de Dice: 2·TP / (2·TP + FP + FN), em pixels."""
    tp, fp, fn = confusion(pred, gt)
    denominador = 2 * tp + fp + fn
    if denominador == 0:  # significa que tudo é fundo
        return 1
    return 2 * tp / denominador


def to_binary(labels: np.ndarray) -> np.ndarray:
    """Converte um label map de instâncias em máscara binária."""
    return labels > 0
