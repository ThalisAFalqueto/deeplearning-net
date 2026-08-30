"""Métrica de segmentação de instâncias.

Máscaras são *label maps*: arrays inteiros (H, W) com 0 = fundo e 1, 2, 3… = objetos
(não necessariamente contíguos). É o formato devolvido por `scipy.ndimage.label`.

Definição adotada:
    1. IoU entre cada par (objeto previsto, objeto real).
    2. Matching 1-para-1, permitido apenas quando IoU >= t.
    3. TP/FP/FN contam objetos, não pixels.
    4. AP(t) = TP / (TP + FP + FN)          (convenção DSB2018)
    5. mAP = média de AP(t) sobre t = 0,50; 0,55; …; 0,95
"""

import numpy as np

THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2)


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Calcula o IoU entre todos os pares de objetos.

    Args:
        pred: label map previsto, shape (H, W).
        gt: label map verdadeiro, shape (H, W).

    Returns:
        Array (N, M) com o IoU entre o i-ésimo objeto previsto e o j-ésimo objeto real.
        N e M excluem o fundo. Se um dos lados não tiver objetos, a shape correspondente
        é zero.
    """
    raise NotImplementedError


def match_greedy(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Casa objetos previstos e reais gulosamente, por IoU decrescente.

    Args:
        iou: matriz (N, M) de IoU.
        threshold: IoU mínimo para um par ser aceito.

    Returns:
        Lista de pares (i, j). Cada índice aparece no máximo uma vez de cada lado.
    """
    raise NotImplementedError


def match_hungarian(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Casa objetos maximizando a soma total de IoU (atribuição ótima).

    Args:
        iou: matriz (N, M) de IoU.
        threshold: IoU mínimo para um par ser aceito.

    Returns:
        Lista de pares (i, j). Cada índice aparece no máximo uma vez de cada lado.
    """
    raise NotImplementedError


def counts_at_threshold(iou: np.ndarray, threshold: float, matcher=match_greedy):
    """Conta objetos verdadeiro-positivos, falso-positivos e falso-negativos.

    Args:
        iou: matriz (N, M) de IoU.
        threshold: IoU mínimo para um par ser aceito.
        matcher: regra de matching a usar.

    Returns:
        Tupla (tp, fp, fn) de inteiros.
    """
    raise NotImplementedError


def average_precision(iou: np.ndarray, threshold: float, matcher=match_greedy) -> float:
    """Precisão média num único limiar: TP / (TP + FP + FN)."""
    raise NotImplementedError


def mean_average_precision(pred: np.ndarray, gt: np.ndarray, matcher=match_greedy):
    """Média de average_precision sobre THRESHOLDS.

    Args:
        pred: label map previsto.
        gt: label map verdadeiro.
        matcher: regra de matching a usar.

    Returns:
        Tupla (mAP, por_limiar), onde por_limiar é um dict {t: AP(t)}.
    """
    raise NotImplementedError


def count_error(pred: np.ndarray, gt: np.ndarray) -> int:
    """Erro absoluto de contagem de objetos entre predição e gabarito."""
    raise NotImplementedError
