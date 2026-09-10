"""Tarefa binária: cross-entropy sobre um único canal de foreground.

É a perda das Partes 0 e 1 — a máscara binária pura. Não separa objetos encostados (para
isso existe a ``center_offset``), mas continua sendo o baseline contra o qual a Trilha C é
comparada, e o caminho que exercita a :class:`~src.models.heads.BinaryHead`.

Mesma interface de :class:`~src.losses.center_offset.CenterOffsetTask`: ``build_targets`` +
``__call__(outputs, targets)`` devolvendo ``(perda, dict de componentes)``.
"""

import torch
import torch.nn as nn


class BCELoss:
    """BCEWithLogits sobre o canal 0, com alvo ``(labels > 0)``.

    Args:
        pos_weight: peso da classe positiva passado a ``nn.BCEWithLogitsLoss``. Útil quando
            o foreground é uma fração pequena da imagem. ``0`` (padrão) desliga o reforço.
    """

    def __init__(self, pos_weight: float = 0.0):
        pw = torch.tensor(float(pos_weight)) if pos_weight else None
        self._bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def build_targets(self, labels: torch.Tensor, device) -> torch.Tensor:
        """Label map (B, H, W) → máscara binária (B, 1, H, W) float."""
        return (labels > 0).float().unsqueeze(1).to(device)

    def __call__(self, outputs, targets):
        """``outputs`` é o tensor ``(B, 1, H, W)`` de logits da cabeça binária."""
        perda = self._bce(outputs, targets)
        return perda, {"bce": float(perda.detach())}
