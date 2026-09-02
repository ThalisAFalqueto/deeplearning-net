"""Converte um mapa de probabilidade em label map de instâncias via limiar + componentes conexos."""

import numpy as np
from scipy import ndimage


def labels_from_probability(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converte um mapa de probabilidade em label map de instâncias.

    Args:
        prob: array (H, W) de floats em [0, 1] — a saída da rede após a sigmoide.
        threshold: corte para considerar um pixel como objeto.

    Returns:
        Array (H, W) de inteiros: 0 = fundo, 1..N = instâncias.
    """
    mask = prob > threshold
    labels, num = ndimage.label(mask)
    return labels
