"""U-Net da Trilha C: backbone U-Net + três cabeças (segmentação, heatmap, offsets).

Depois da refatoração que separou backbone, cabeça e perda, esta classe é só um atalho de
compatibilidade: ``Segmenter(UNet(...), CenterOffsetHeads(...))``. Código novo deve pedir o
modelo à ``ModelFactoryRegistry`` (que escolhe a cabeça pelo ``model.loss.name``); esta
classe existe porque testes e notebooks a constroem diretamente.

    L = γ · CE(segmentação) + α · L2(heatmap) + β · L1(offsets)   ← agora em CenterOffsetTask

As três cabeças 1×1 são a única diferença em relação à U-Net da Parte 1 — o encoder-decoder
é herdado sem uma linha de alteração.
"""

from src.models.heads import CenterOffsetHeads
from src.models.segmenter import Segmenter
from src.models.unet import UNet


class UNetImproved(Segmenter):
    """UNet backbone + CenterOffsetHeads. Não possui perda (fica na LossFactoryRegistry).

    Args:
        in_channels: canais da imagem de entrada.
        base: filtros do primeiro nível do encoder.
        depth: quantas vezes reduz a resolução.
        loss_cfg: aceito e **ignorado** — a perda deixou de ser responsabilidade do modelo.
            Mantido só para não quebrar chamadas antigas.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 16,
        depth: int = 3,
        loss_cfg: dict | None = None,
    ):
        backbone = UNet(in_channels=in_channels, base=base, depth=depth)
        super().__init__(backbone, CenterOffsetHeads(backbone.feature_channels))

    # --- atalhos para os blocos do backbone e da cabeça, usados por testes e inspeção
    @property
    def encoders(self):
        return self.backbone.encoders

    @property
    def bottleneck(self):
        return self.backbone.bottleneck

    @property
    def ups(self):
        return self.backbone.ups

    @property
    def decoders(self):
        return self.backbone.decoders

    @property
    def seg_head(self):
        return self.head.seg_head

    @property
    def heatmap_head(self):
        return self.head.heatmap_head

    @property
    def offset_head(self):
        return self.head.offset_head
