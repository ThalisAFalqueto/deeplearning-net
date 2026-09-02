"""Converte um label map de instâncias em máscara binária."""

import torch


def to_binary(labels: torch.Tensor) -> torch.Tensor:
    """Converte um label map de instâncias em máscara binária.

    Args:
        labels: tensor de rótulos de instância (qualquer inteiro > 0 é objeto).

    Returns:
        Máscara booleana com True onde há objeto.
    """
    return labels > 0
