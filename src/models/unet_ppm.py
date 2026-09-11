"""U-Net + Pyramid Pooling Module no bottleneck — o Eixo 3 da Parte 3.

O enunciado pede pyramid pooling "acoplado ao seu decoder" e pergunta se o contexto global
ajuda a *separar* instâncias ou só a *classificar* melhor. Para isolar essa pergunta, este
backbone muda **uma coisa só** em relação à U-Net: entre o bottleneck e o decoder entra o
``PyramidPoolingModule`` (o mesmo de ``src/models/pspnet.py``).

    encoder  →  bottleneck  →  PPM  →  decoder com skips (idêntico ao da U-Net)

A ``PSPNet`` do projeto responde a outra pergunta: ela troca o decoder inteiro por um
upsample bilinear, então perde as skips junto — e o Eixo 1 mostrou que as skips pesam muito
para instâncias. Aqui encoder, bottleneck e decoder são os da U-Net, parâmetro por parâmetro
(ver ``tests/test_models.py``); a diferença na comparação é o PPM.

O PPM mantém a largura do bottleneck (entra e sai com ``base * 2**depth`` canais), por isso o
decoder não muda. Custo: o PPM acrescenta parâmetros (a conv 3×3 de projeção dele é a maior
parte) — declarar a contagem junto do resultado.

Atenção, herdada do PPM: o ramo de bin 1×1 aplica ``BatchNorm2d`` sobre um tensor
``(B, C, 1, 1)``, o que em modo treino exige ``batch_size > 1``.
"""

import torch

from src.models.pspnet import PyramidPoolingModule
from src.models.unet import UNet


class UNetPPM(UNet):
    """U-Net com Pyramid Pooling Module entre o bottleneck e o decoder.

    Args:
        in_channels: canais da imagem de entrada.
        base: filtros no primeiro nível; dobra a cada descida.
        depth: quantas vezes reduz a resolução.
        bins: grades do Pyramid Pooling Module. A grade 1 é a média global da imagem.

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
        super().__init__(in_channels=in_channels, base=base, depth=depth)
        # mesma largura do bottleneck na entrada e na saída: o decoder herdado não muda
        self.ppm = PyramidPoolingModule(base * (2 ** depth), bins=bins)

    def features(self, x):
        """Igual a ``UNet.features``, com o PPM aplicado logo depois do bottleneck.

        Returns:
            (B, base, H, W) — features na resolução da entrada.
        """
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # a única diferença para a U-Net: o bottleneck recebe o resumo da imagem em
        # grades 1×1, 2×2, 3×3 e 6×6 antes de descer pelo decoder
        x = self.ppm(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return x

    def receptive_field(self) -> int:
        """Campo receptivo **local** do encoder, em pixels da entrada.

        É o da U-Net mais a conv 3×3 de projeção do PPM, que roda na resolução do
        bottleneck (salto ``2**depth``). Mesma convenção da ``PSPNet``: o ramo de bin 1×1 é
        a média global, então o contexto efetivo é a imagem inteira; este número mede só o
        alcance da pilha de convoluções, que é o que a Parte 5 compara com o tamanho dos
        objetos.
        """
        return super().receptive_field() + (3 - 1) * (2 ** self.depth)
