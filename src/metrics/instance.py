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
from scipy.optimize import linear_sum_assignment


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
    pred_unq = np.unique(pred[pred != 0])
    gt_unq = np.unique(gt[gt != 0])
    len_pred, len_gt = len(pred_unq), len(gt_unq)
    matrix = np.zeros((len_pred, len_gt))

    for indice_pred, label_pred in enumerate(pred_unq):
        for indice_gt, label_gt in enumerate(gt_unq):
            intersec = ((pred == label_pred) & (gt == label_gt)).sum()
            union = np.abs((pred == label_pred).sum()) + np.abs((gt == label_gt).sum()) - intersec
            iou = intersec / union
            matrix[indice_pred, indice_gt] = iou

    return matrix


def match_greedy(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Casa objetos previstos e reais gulosamente, por IoU decrescente.

    Args:
        iou: matriz (N, M) de IoU.
        threshold: IoU mínimo para um par ser aceito.

    Returns:
        Lista de pares (i, j). Cada índice aparece no máximo uma vez de cada lado.
    """
    pred_used = {label: False for label in range(iou.shape[0])}
    gt_used = {label: False for label in range(iou.shape[1])}

    flat = iou.flatten()  # transforma num vetor
    order = np.argsort(flat)[::-1]  # ordena do maior pro menor, pegando o índice
    pares = []
    for indice_iou in order:
        i, j = np.unravel_index(indice_iou, iou.shape)
        if (pred_used[i] is True) or (gt_used[j] is True):
            continue
        value = iou[i, j]
        if value < threshold:
            break
        pares.append((i, j))
        pred_used[i], gt_used[j] = True, True
    return pares


def match_hungarian(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Casa objetos maximizando a soma total de IoU (atribuição ótima).

    Args:
        iou: matriz (N, M) de IoU.
        threshold: IoU mínimo para um par ser aceito.

    Returns:
        Lista de pares (i, j). Cada índice aparece no máximo uma vez de cada lado.
    """
    # retorna o menor custo, por isso passamos o iou negativo. devolve uma tupla de array (linha, coluna)
    best_match = linear_sum_assignment(-iou)
    pares = []
    rows, columns = best_match[0], best_match[1]
    for indice in range(len(best_match[0])):
        i, j = rows[indice], columns[indice]
        value = iou[i, j]
        if value >= threshold:
            pares.append((i, j))

    return pares


def counts_at_threshold(iou: np.ndarray, threshold: float, matcher=match_greedy):
    """Conta objetos verdadeiro-positivos, falso-positivos e falso-negativos.

    Args:
        iou: matriz (N, M) de IoU.
        threshold: IoU mínimo para um par ser aceito.
        matcher: regra de matching a usar.

    Returns:
        Tupla (tp, fp, fn) de inteiros.
    """
    pares = matcher(iou, threshold)
    tp = len(pares)
    len_rows, len_columns = iou.shape[0], iou.shape[1]
    fp = len_rows - tp
    fn = len_columns - tp
    return (tp, fp, fn)


def average_precision(iou: np.ndarray, threshold: float, matcher=match_greedy) -> float:
    """Precisão média num único limiar: TP / (TP + FP + FN)."""
    tp, fp, fn = counts_at_threshold(iou, threshold, matcher)
    denominator = tp + fp + fn
    if denominator == 0:
        return 1
    return tp / denominator


def mean_average_precision(pred: np.ndarray, gt: np.ndarray, matcher=match_greedy):
    """Média de average_precision sobre THRESHOLDS.

    Args:
        pred: label map previsto.
        gt: label map verdadeiro.
        matcher: regra de matching a usar.

    Returns:
        Tupla (mAP, por_limiar), onde por_limiar é um dict {t: AP(t)}.
    """
    iou = iou_matrix(pred, gt)
    dict_thresholds = {}
    for th in THRESHOLDS:
        ap = average_precision(iou, th, matcher)
        dict_thresholds[th] = ap

    map = np.array(list(dict_thresholds.values())).mean()
    return (map, dict_thresholds)


def count_error(pred: np.ndarray, gt: np.ndarray) -> int:
    """Erro absoluto de contagem de objetos entre predição e gabarito."""
    pred_unq = np.unique(pred[pred != 0])
    gt_unq = np.unique(gt[gt != 0])
    return abs(len(pred_unq) - len(gt_unq))
