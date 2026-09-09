"""SegNet — encoder-decoder com max unpooling e índices.

Diferente da U-Net, o decoder não usa transpose convolution. Em vez disso,
o max pooling do encoder guarda os índices dos valores máximos, e o decoder
os recupera com ``MaxUnpool2d``.

Interface comum com ``UNet`` e ``UNetImproved``:
    - ``build_targets``, ``compute_loss``, ``foreground_prob``, ``decode``
"""

import torch
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
        out_channels: canais de saída.
        base: número de filtros no primeiro nível.
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

        # cabeça
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)
        self.feature_channels = ch
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        indices_list = []
        for encoder in self.encoders:
            x, indices = encoder(x)
            indices_list.append(indices)

        x = self.bridge(x)

        for decoder, indices in zip(self.decoders, reversed(indices_list)):
            x = decoder(x, indices)

        return self.head(x)

    def features(self, x):
        indices_list = []
        for encoder in self.encoders:
            x, indices = encoder(x)
            indices_list.append(indices)

        x = self.bridge(x)

        for decoder, indices in zip(self.decoders, reversed(indices_list)):
            x = decoder(x, indices)

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
