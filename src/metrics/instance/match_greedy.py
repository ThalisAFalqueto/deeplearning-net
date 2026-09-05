"""Matching guloso de objetos por IoU decrescente."""

import torch


class MatchGreedy:
    """Casa objetos previstos e reais gulosamente, por IoU decrescente."""

    def __call__(self, iou: torch.Tensor, threshold: float) -> list:
        pred_used = {label: False for label in range(iou.shape[0])}
        gt_used = {label: False for label in range(iou.shape[1])}

        flat = iou.flatten()
        order = torch.argsort(flat, descending=True)
        pares = []
        for indice_iou in order:
            i, j = divmod(indice_iou.item(), iou.shape[1])
            if pred_used[i] or gt_used[j]:
                continue
            value = iou[i, j]
            if value < threshold:
                break
            pares.append((i, j))
            pred_used[i], gt_used[j] = True, True
        return pares
