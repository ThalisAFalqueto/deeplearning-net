"""ResUNet — U-Net com Residual Blocks.

Substitui os blocos convolucionais básicos da U-Net por Residual Blocks,
facilitando o treinamento de redes mais profundas. As skip connections
continuam sendo por concatenação, como na U-Net original.

Tipos de atalho na arquitetura:

1. Residual connection (dentro do bloco):
       x ────────────────┐
       │                 ↓
       └→ Conv → BN → ReLU → Conv → BN → (+) → output
                             ↑
                             └──── identity x

2. U-Net skip connection (entre encoder e decoder):
       Encoder feature ──────────────┐
                                     ↓
       Decoder feature ──────────→ CONCATENATE
                                     ↓
                                  Residual Block
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bloco residual com duas convoluções 3x3.

    A soma ``F(x) + x`` é a conexão residual. Quando os canais mudam,
    uma convolução 1x1 na identity projeta x para o mesmo espaço.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        self.projection = None
        if stride != 1 or in_ch != out_ch:
            self.projection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.projection is not None:
            identity = self.projection(identity)
        out = self.relu(out + identity)
        return out


class ResUNet(nn.Module):
    """ResUNet com profundidade configurável.

    É um **backbone**: ``features(x)`` devolve ``(B, base, H, W)``. A cabeça de tarefa e a
    perda ficam fora (ver ``src/models/heads.py`` e ``src/losses/factory.py``).

    Args:
        in_channels: canais da imagem de entrada.
        base: filtros no primeiro nível do encoder.
        depth: quantas vezes reduz a resolução.

    Shape:
        entrada  (B, in_channels, H, W)
        saída    (B, base, H, W)   — features na resolução da entrada
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 16,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth

        # encoder: Residual Blocks + max pooling
        self.encoders = nn.ModuleList()
        ch = in_channels
        for d in range(depth):
            out = base * (2 ** d)
            self.encoders.append(ResidualBlock(ch, out))
            ch = out
        self.pool = nn.MaxPool2d(2)

        # bridge
        self.bridge = ResidualBlock(ch, ch)

        # decoder: upsample, concatena skip, Residual Block
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = base * (2 ** d)
            self.ups.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock(skip_ch * 2, skip_ch))
            ch = skip_ch

        # canais de saída do backbone, para a factory dimensionar a cabeça
        self.feature_channels = ch

    def features(self, x):
        """Encoder → bridge → decoder (skips por concatenação).

        Returns:
            (B, base, H, W) — features na resolução da entrada.
        """
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bridge(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

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
