"""Métricas semânticas sobre a máscara binária.

Comparam duas máscaras binárias da imagem inteira e ignoram a divisão em objetos.
"""

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray):
    """Conta pixels verdadeiro-positivos, falso-positivos e falso-negativos.

    Args:
        pred: máscara binária prevista.
        gt: máscara binária verdadeira.

    Returns:
        Tupla (tp, fp, fn) de inteiros.
    """
    raise NotImplementedError


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union: TP / (TP + FP + FN), em pixels."""
    raise NotImplementedError


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Coeficiente de Dice: 2·TP / (2·TP + FP + FN), em pixels."""
    raise NotImplementedError


def to_binary(labels: np.ndarray) -> np.ndarray:
    """Converte um label map de instâncias em máscara binária."""
    raise NotImplementedError
