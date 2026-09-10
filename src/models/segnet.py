"""SegNet — encoder-decoder com max unpooling e índices.

Diferente da U-Net, o decoder não usa transpose convolution. Em vez disso,
o max pooling do encoder guarda os índices dos valores máximos, e o decoder
os recupera com ``MaxUnpool2d``.

É um **backbone**: ``features(x)`` devolve ``(B, feature_channels, H, W)``. A cabeça de
tarefa e a perda ficam fora (ver ``src/models/heads.py`` e ``src/losses/factory.py``).
Atenção: o decoder reduz canais pela metade a cada nível, então ``feature_channels`` acaba
em ``base / 2`` (não ``base`` como na U-Net) — a cabeça é dimensionada a partir desse
atributo, então isso é transparente.
"""

import torch.nn as nn


class SegNetEncoderBlock(nn.Module):
    """Bloco do encoder: convoluções + max pool com guarda de índices."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x)
        return x, indices


class SegNetDecoderBlock(nn.Module):
    """Bloco do decoder: max unpooling + convoluções.

    A cada passo, os canais são reduzidos pela metade após o unpooling.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = self.conv(x)
        return x


class SegNet(nn.Module):
    """SegNet com profundidade configurável.

    Args:
        in_channels: canais da imagem de entrada (1 para escala de cinza).
        base: número de filtros no primeiro nível.
        depth: quantas vezes reduz a resolução.

    Shape:
        entrada  (B, in_channels, H, W)
        saída    (B, feature_channels, H, W)   — features na resolução da entrada
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 16,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth

        # encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for d in range(depth):
            out = base * (2 ** d)
            self.encoders.append(SegNetEncoderBlock(ch, out))
            ch = out

        # bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # decoder: cada bloco reduz canais pela metade
        self.decoders = nn.ModuleList()
        for _ in reversed(range(depth)):
            self.decoders.append(SegNetDecoderBlock(ch))
            ch = ch // 2

        # canais de saída do backbone, para a factory dimensionar a cabeça
        self.feature_channels = ch

    def features(self, x):
        """Encoder → bridge → decoder. Os índices do pooling ficam locais a esta chamada.

        Returns:
            (B, feature_channels, H, W) — features na resolução da entrada.
        """
        indices_list = []
        for encoder in self.encoders:
            x, indices = encoder(x)
            indices_list.append(indices)

        x = self.bridge(x)

        for decoder, indices in zip(self.decoders, reversed(indices_list)):
            x = decoder(x, indices)

        return x

    def forward(self, x):
        return self.features(x)

    def receptive_field(self) -> int:
        r, j = 1, 1
        for _ in range(self.depth):
            for _ in range(2):
                r += (3 - 1) * j
            r += (2 - 1) * j
            j *= 2
        for _ in range(2):
            r += (3 - 1) * j
        return r
