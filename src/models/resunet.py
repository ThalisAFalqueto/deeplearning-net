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

    Args:
        in_channels: canais da imagem de entrada.
        out_channels: canais de saída.
        base: filtros no primeiro nível do encoder.
        depth: quantas vezes reduz a resolução.

    Shape:
        entrada  (B, in_channels, H, W)
        saída    (B, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
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

        # cabeça
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)
        self.feature_channels = ch
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
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

        return self.head(x)

    def features(self, x):
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

    # ------------------------------------------------------------------ interface comum
    def build_targets(self, labels: torch.Tensor, device):
        return (labels > 0).float().unsqueeze(1).to(device)

    def compute_loss(self, outputs, targets):
        perda = self._bce(outputs, targets)
        return perda, {"bce": float(perda.detach())}

    def foreground_prob(self, outputs) -> torch.Tensor:
        return torch.sigmoid(outputs)[:, 0]

    def decode(self, outputs, decode_cfg: dict):
        from src.utils import labels_from_probability

        prob = torch.sigmoid(outputs)[0].detach().cpu().numpy()
        return labels_from_probability(prob, decode_cfg["threshold"])

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
