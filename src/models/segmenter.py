"""Compõe um backbone com uma cabeça de tarefa.

É o que a ``ModelFactoryRegistry.build`` devolve e o que os dois engines (treino e
avaliação) consomem. O backbone só sabe extrair features; a cabeça sabe o que prever e como
decodificar. Juntar os dois num ``nn.Module`` mantém a interface que os engines já esperam
(``model(x)``, ``model.foreground_prob``, ``model.decode``, ``model.receptive_field``) sem
que nenhum deles precise saber da separação.
"""

import torch.nn as nn


class Segmenter(nn.Module):
    """backbone + cabeça.

    Args:
        backbone: expõe ``features(x) -> (B, feature_channels, H, W)``, ``feature_channels``,
            ``depth`` e ``receptive_field()``.
        head: expõe ``forward(features)``, ``foreground_prob(outputs)`` e
            ``decode(outputs, decode_cfg)``.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # espelhados para quem lê direto do modelo (logs, cabeças de variantes)
        self.feature_channels = backbone.feature_channels
        self.depth = backbone.depth

    def forward(self, x):
        return self.head(self.backbone.features(x))

    def features(self, x):
        return self.backbone.features(x)

    def foreground_prob(self, outputs):
        return self.head.foreground_prob(outputs)

    def decode(self, outputs, decode_cfg):
        return self.head.decode(outputs, decode_cfg)

    def receptive_field(self) -> int:
        return self.backbone.receptive_field()
