"""PSPNet — encoder + Pyramid Pooling Module (image pooling).

A ideia da PSPNet (Zhao et al., 2017) é que uma conv, por mais profunda que seja, agrega
contexto de uma vizinhança limitada. Para segmentar bem, a rede precisa de contexto de
várias escalas ao mesmo tempo — inclusive global. O Pyramid Pooling Module resolve isso
resumindo o mapa de features em grades de 1×1, 2×2, 3×3 e 6×6 (a de 1×1 é a média global da
imagem), projetando cada uma e concatenando de volta.

Como backbone deste projeto:

    encoder (DoubleConv + MaxPool, igual à U-Net)  →  bottleneck  →  PPM
        →  upsample bilinear até a resolução da entrada  →  projeção 3×3  →  (B, base, H, W)

Não há decoder com skips — o PPM substitui o decoder no papel de reunir contexto. A cabeça
de tarefa e a perda ficam fora (ver ``src/models/heads.py`` e ``src/losses/factory.py``).

Atenção: o ramo de bin 1×1 alimenta um ``BatchNorm2d`` com um tensor ``(B, C, 1, 1)`` — em
modo treino isso exige ``batch_size > 1`` (todos os configs usam 8). Nos testes, usar
``B >= 2`` ou ``.eval()``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import DoubleConv


class PyramidPoolingModule(nn.Module):
    """Pyramid pooling: resume ``x`` em várias grades, projeta e concatena de volta.

    Args:
        in_ch: canais de entrada (e de saída — o módulo devolve a mesma largura).
        bins: tamanhos das grades de ``AdaptiveAvgPool2d``. A grade 1 é a média global.
    """

    def __init__(self, in_ch: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = tuple(bins)
        reduction = max(1, in_ch // len(self.bins))
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_ch, reduction, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction),
                nn.ReLU(inplace=True),
            )
            for b in self.bins
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_ch + reduction * len(self.bins), in_ch, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        return self.project(torch.cat(feats, dim=1))


class PSPNet(nn.Module):
    """Encoder da U-Net + Pyramid Pooling Module, como backbone.

    Args:
        in_channels: canais da imagem de entrada.
        base: filtros no primeiro nível do encoder; dobra a cada descida.
        depth: quantas vezes reduz a resolução.
        bins: grades do Pyramid Pooling Module.

    Shape:
        entrada  (B, in_channels, H, W)
        saída    (B, base, H, W)   — features na resolução da entrada
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 16,
        depth: int = 3,
        bins=(1, 2, 3, 6),
    ):
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        ch = in_channels
        for d in range(depth):
            out = base * (2 ** d)
            self.encoders.append(DoubleConv(ch, out))
            ch = out
        self.pool = nn.MaxPool2d(2)

        bottleneck_ch = base * (2 ** depth)
        self.bottleneck = DoubleConv(ch, bottleneck_ch)
        self.ppm = PyramidPoolingModule(bottleneck_ch, bins=bins)

        # projeção final: volta à largura `base`, para as cabeças serem idênticas às das
        # outras arquiteturas
        self.project = nn.Sequential(
            nn.Conv2d(bottleneck_ch, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.feature_channels = base

    def features(self, x):
        """Encoder → bottleneck → PPM → upsample até a entrada → projeção.

        Returns:
            (B, base, H, W) — features na resolução da entrada.
        """
        entrada_hw = x.shape[-2:]
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.ppm(x)
        x = F.interpolate(x, size=entrada_hw, mode="bilinear", align_corners=False)
        return self.project(x)

    def forward(self, x):
        return self.features(x)

    def receptive_field(self) -> int:
        """Campo receptivo **local** da pilha de convoluções, em pixels da entrada.

        Aplica a mesma recorrência da U-Net (slides 35-38) sobre o encoder, o bottleneck e
        a conv 3×3 do ``project`` do PPM. É deliberadamente uma subestimativa: o ramo de
        bin 1×1 do PPM é a média global, então o contexto efetivo é a imagem inteira. O
        número aqui serve ao diagnóstico da Parte 5, que o compara com o tamanho dos
        objetos — para isso interessa o alcance da pilha conv, não o do pooling global.
        """
        r, j = 1, 1
        for _ in range(self.depth):
            for _ in range(2):
                r += (3 - 1) * j
            r += (2 - 1) * j
            j *= 2
        for _ in range(2):        # bottleneck DoubleConv
            r += (3 - 1) * j
        r += (3 - 1) * j          # conv 3x3 do project do PPM
        return r
