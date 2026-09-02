"""Converte um label map de instâncias em máscara binária."""

import torch
import torch.nn as nn


class ToBinary(nn.Module):
    """Converte um label map de instâncias em máscara binária."""

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return labels > 0

    def backward(self, *grad_outputs):
        raise NotImplementedError
