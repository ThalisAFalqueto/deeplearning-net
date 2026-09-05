"""U-Net da Trilha C: três cabeças em vez de uma.

A Parte 1 mostrou que uma máscara binária não separa objetos encostados — não há nada nela
que distinga dois núcleos colados de um só. A Parte 2 resolve isso mudando **o que a rede
prevê**, e é este arquivo que declara essa mudança:

    segmentação   (B, 1, H, W)   "este pixel é objeto?"
    heatmap       (B, 1, H, W)   "este pixel é o centro de um objeto?"
    offsets       (B, 2, H, W)   "quanto andar em (y, x) para chegar no centro do MEU objeto"

O encoder e o decoder são herdados da ``UNet`` sem uma linha de alteração — o enunciado
exige que sejam os mesmos da Parte 1 ("mantenham o encoder-decoder da Parte 1 e mudem o que
ele prevê"). O que muda são as convoluções 1x1 finais: três, uma por saída.

Cada saída pede uma perda diferente, e as três são somadas com pesos:

    L = γ · CE(segmentação) + α · L2(heatmap) + β · L1(offsets)
"""

import numpy as np
import torch
import torch.nn as nn

from src.data.targets import batch_center_offset_targets
from src.losses.center_offset import CenterOffsetLoss
from src.models.unet import UNet


class UNetImproved(UNet):
    """U-Net com três cabeças: segmentação, heatmap de centros e offsets.

    Args:
        in_channels: canais da imagem de entrada.
        base: filtros do primeiro nível do encoder.
        depth: quantas vezes reduz a resolução.
        loss_cfg: pesos e parâmetros da perda combinada. Chaves aceitas: ``w_seg``,
            ``w_heatmap``, ``w_offset``, ``pos_weight``.

    Shape:
        entrada  (B, in_channels, H, W)
        saída    três tensores: (B,1,H,W), (B,1,H,W), (B,2,H,W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 16,
        depth: int = 3,
        loss_cfg: dict | None = None,
    ):
        # out_channels=1 aqui é irrelevante: a cabeça única da UNet é substituída abaixo.
        super().__init__(in_channels, out_channels=1, base=base, depth=depth)
        del self.head

        ch = self.feature_channels
        # Uma conv 1x1 por saída. Elas veem exatamente as mesmas features do decoder —
        # o que difere é o que cada uma aprende a extrair delas.
        self.seg_head = nn.Conv2d(ch, 1, kernel_size=1)
        self.heatmap_head = nn.Conv2d(ch, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(ch, 2, kernel_size=1)

        self.criterion = CenterOffsetLoss(**(loss_cfg or {}))

    def forward(self, x):
        """
        Returns:
            Tupla (seg_logits, heatmap_logits, offsets). Os dois primeiros são logits —
            a sigmoide fica na perda e na decodificação, não aqui, porque
            ``BCEWithLogitsLoss`` é numericamente mais estável recebendo o logit cru.
            Os offsets são previstos direto, em pixels.
        """
        f = self.features(x)
        return self.seg_head(f), self.heatmap_head(f), self.offset_head(f)

    # ------------------------------------------------------------------ interface comum

    def build_targets(self, labels: torch.Tensor, device):
        """Constrói heatmap gaussiano e offsets a partir do label map do gabarito.

        Returns:
            Dicionário com ``foreground`` (B,1,H,W), ``heatmap`` (B,1,H,W) e
            ``offsets`` (B,2,H,W).
        """
        alvos = batch_center_offset_targets(labels)
        return {chave: valor.to(device) for chave, valor in alvos.items()}

    def compute_loss(self, outputs, targets):
        """Soma ponderada das três perdas.

        Args:
            outputs: a tupla devolvida por ``forward``.
            targets: o dicionário devolvido por ``build_targets``.

        Returns:
            (perda total, dict com as três componentes). O dict aparece no log por época —
            com três perdas somadas, o total sozinho esconde qual delas estagnou.
        """
        seg_logits, heatmap_logits, offsets = outputs
        return self.criterion(seg_logits, heatmap_logits, offsets, targets)

    def foreground_prob(self, outputs) -> torch.Tensor:
        """Probabilidade de foreground, (B, H, W) — para o IoU/Dice da validação."""
        return torch.sigmoid(outputs[0])[:, 0]

    def decode(self, outputs, decode_cfg: dict) -> np.ndarray:
        """Uma imagem → label map de instâncias, por picos e atribuição ao centro.

        Args:
            outputs: as três saídas de UMA imagem — (1,H,W), (1,H,W), (2,H,W).
            decode_cfg: ``threshold`` (foreground), ``peak_threshold``, ``nms_kernel``.

        Returns:
            Array (H, W) de inteiros: 0 = fundo, 1..N = instâncias.
        """
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
