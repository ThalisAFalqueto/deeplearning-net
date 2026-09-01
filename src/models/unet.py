"""U-Net — encoder-decoder com skip connections.

Uma única implementação serve as três Partes; o que muda é só `out_channels`:

    Partes 0 e 1   out_channels = 1        logit de foreground
    Parte 2 (B)    out_channels = 1 + D    foreground + D canais de embedding

O decoder usa ConvTranspose2d com kernel 2 e stride 2. Kernel divisível pelo stride
significa que as contribuições não se sobrepõem de forma desigual — é o que evita os
artefatos de xadrez que aparecem, por exemplo, com kernel 3 e stride 2.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Duas convoluções 3x3, cada uma seguida de BatchNorm e ReLU.

    É o bloco básico da U-Net original. O padding 1 mantém a resolução, então os
    tamanhos do encoder e do decoder batem e a concatenação do skip não precisa de
    recorte (o `copy and crop` do artigo original).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net com profundidade configurável.

    Args:
        in_channels: canais da imagem de entrada (1 para escala de cinza).
        out_channels: canais de saída (ver docstring do módulo).
        base: número de filtros no primeiro nível; dobra a cada descida.
        depth: quantas vezes reduz a resolução.

    Shape:
        entrada  (B, in_channels, H, W)
        saída    (B, out_channels, H, W)   — mesma resolução da entrada
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

        # --- encoder (contract): a cada nível, DoubleConv e depois pooling
        self.encoders = nn.ModuleList()
        ch = in_channels
        for d in range(depth):
            out = base * (2 ** d)
            self.encoders.append(DoubleConv(ch, out))
            ch = out
        self.pool = nn.MaxPool2d(2)

        # --- bottleneck: o nível mais fundo, maior campo receptivo
        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2

        # --- decoder (expand): upsample, concatena o skip, DoubleConv
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = base * (2 ** d)
            self.ups.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            # entrada da DoubleConv = canais do upsample + canais do skip
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
            ch = skip_ch

        # --- cabeça: conv 1x1 não olha vizinhança, só combina canais
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)          # guarda ANTES do pooling: resolução cheia
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # concatena na dimensão dos canais: o decoder recebe a semântica que ele
            # trouxe do fundo MAIS a localização precisa que o encoder guardou
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.head(x)

    def receptive_field(self) -> int:
        """Campo receptivo teórico de um pixel de saída, em pixels da entrada.

        Aplica a recorrência dos slides 35-38, camada por camada:
            j_l = j_{l-1} * s_l
            r_l = r_{l-1} + (k_l - 1) * j_{l-1}

        Necessário na Parte 5, onde o diagnóstico das falhas compara este número com
        a distribuição de tamanhos dos objetos do dataset.
        """
        r, j = 1, 1
        # encoder: cada nível tem 2 convs 3x3 (stride 1) e 1 pooling 2x2 (stride 2)
        for _ in range(self.depth):
            for _ in range(2):
                r += (3 - 1) * j
            r += (2 - 1) * j
            j *= 2
        # bottleneck: mais 2 convs 3x3
        for _ in range(2):
            r += (3 - 1) * j
        return r
