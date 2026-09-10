"""Cabeças de tarefa: transformam as features do backbone em saídas e as saídas em objetos.

O backbone (UNet, SegNet, ResUNet, PSPNet) só produz um mapa de features
``(B, feature_channels, H, W)`` na resolução da entrada. É a cabeça que decide **o que a
rede prevê** e como isso vira um label map de instâncias:

    BinaryHead          1 conv 1x1   → 1 logit de foreground        (tarefa ``bce``)
    CenterOffsetHeads   3 convs 1x1  → seg + heatmap + offsets      (tarefa ``center_offset``)

O tipo de cabeça é escolhido pela factory a partir de ``model.loss.name`` — a mesma fonte
que escolhe a perda (ver ``src/losses/factory.py``). ``foreground_prob`` e ``decode`` vivem
aqui porque interpretam o layout da saída, que é justamente o que a cabeça define.
"""

import numpy as np
import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    """Cabeça binária: uma conv 1x1 para o logit de foreground.

    Decodifica por limiar + componentes conexos — o método ingênuo da Parte 1, que funde
    objetos encostados num só.
    """

    def __init__(self, feature_channels: int, out_channels: int = 1):
        super().__init__()
        self.head = nn.Conv2d(feature_channels, out_channels, kernel_size=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)

    def foreground_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Probabilidade de foreground, (B, H, W)."""
        return torch.sigmoid(outputs)[:, 0]

    def decode(self, outputs: torch.Tensor, decode_cfg: dict) -> np.ndarray:
        """Saída de UMA imagem (1, H, W) → label map por limiar + componentes conexos."""
        from src.utils import labels_from_probability

        prob = torch.sigmoid(outputs)[0].detach().cpu().numpy()
        return labels_from_probability(prob, decode_cfg["threshold"])


class CenterOffsetHeads(nn.Module):
    """Três cabeças 1x1 sobre as mesmas features: segmentação, heatmap de centros e offsets.

    Elas veem exatamente o mesmo mapa de features — o que difere é o que cada uma aprende a
    extrair dele. A decodificação em instâncias usa picos do heatmap + atribuição pelo
    offset (``decode_center_offset``).
    """

    def __init__(self, feature_channels: int):
        super().__init__()
        self.seg_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.heatmap_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(feature_channels, 2, kernel_size=1)

    def forward(self, f: torch.Tensor) -> tuple:
        """Devolve os três logits/mapas crus: (seg, heatmap, offsets).

        A sigmoide fica na perda e na decodificação — ``BCEWithLogitsLoss`` é numericamente
        mais estável recebendo o logit cru. Os offsets são previstos direto, em pixels.
        """
        return self.seg_head(f), self.heatmap_head(f), self.offset_head(f)

    def foreground_prob(self, outputs) -> torch.Tensor:
        """Probabilidade de foreground, (B, H, W) — para o IoU/Dice da validação."""
        return torch.sigmoid(outputs[0])[:, 0]

    def decode(self, outputs, decode_cfg: dict) -> np.ndarray:
        """Saída de UMA imagem — (1,H,W), (1,H,W), (2,H,W) — → label map de instâncias."""
        from src.utils.decode_center_offset import decode_center_offset

        seg_logits, heatmap_logits, offsets = outputs
        return decode_center_offset(
            seg_logits[0].cpu(),
            heatmap_logits[0].cpu(),
            offsets.cpu(),
            fg_threshold=decode_cfg.get("threshold", 0.5),
            peak_threshold=decode_cfg.get("peak_threshold", 0.5),
            nms_kernel=decode_cfg.get("nms_kernel", 3),
        )
