"""Constrói o modelo a partir da configuração.

Mesmo padrão do ``DatasetFactoryRegistry`` em ``src/data/factory.py``: trocar ``model.name``
no YAML troca a arquitetura do backbone. Trocar ``model.loss.name`` troca a cabeça **e** a
perda — a escolha vem de uma função só (``resolve_task_name``), então cabeça e perda nunca
dessincronizam.

    backbone   unet | unet_improved | segnet | resunet | pspnet   (só extrai features)
    cabeça     bce          → BinaryHead        (1 logit de foreground)
               center_offset → CenterOffsetHeads (seg + heatmap + offsets)

O que a factory devolve é sempre um ``Segmenter`` (backbone + cabeça).
"""

from abc import ABC, abstractmethod

import torch.nn as nn

from src.losses.factory import resolve_task_name
from src.models.heads import BinaryHead, CenterOffsetHeads
from src.models.pspnet import PSPNet
from src.models.resunet import ResUNet
from src.models.segmenter import Segmenter
from src.models.segnet import SegNet
from src.models.unet import UNet

# ``unet_improved`` é o mesmo backbone da U-Net — o "improved" era só a troca de cabeça,
# que agora é escolhida pela perda. Mantido como alias para os configs da Parte 2.
_BACKBONES = {
    "unet": UNet,
    "unet_improved": UNet,
    "segnet": SegNet,
    "resunet": ResUNet,
    "pspnet": PSPNet,
}


class ModelFactory(ABC):
    """Contrato comum das factories de modelo."""

    @abstractmethod
    def build(self, cfg) -> nn.Module:
        """Instancia o modelo a partir de um TrainConfig ou EvalConfig."""


class SegmenterFactory(ModelFactory):
    """Monta ``Segmenter(backbone, cabeça)`` — a cabeça vem de ``model.loss.name``."""

    def __init__(self, backbone_cls):
        self._backbone_cls = backbone_cls

    def build(self, cfg) -> nn.Module:
        backbone = self._backbone_cls(
            in_channels=cfg.model.get("in_channels", 1),
            base=cfg.model["base"],
            depth=cfg.model["depth"],
        )
        if resolve_task_name(cfg) == "bce":
            head = BinaryHead(backbone.feature_channels, cfg.model.get("out_channels", 1))
        else:
            head = CenterOffsetHeads(backbone.feature_channels)
        return Segmenter(backbone, head)


class ModelFactoryRegistry:
    _factories = {nome: SegmenterFactory(cls) for nome, cls in _BACKBONES.items()}

    @classmethod
    def get(cls, cfg) -> ModelFactory:
        """Escolhe a factory declarada em ``model.name``.

        Sem essa chave, assume ``unet`` — o que mantém funcionando os configs escritos
        antes de existir mais de um modelo.
        """
        nome = cfg.model.get("name", "unet")
        if nome not in cls._factories:
            raise ValueError(
                f"model.name desconhecido: {nome!r}. "
                f"Disponíveis: {sorted(cls._factories)}"
            )
        return cls._factories[nome]

    @classmethod
    def build(cls, cfg) -> nn.Module:
        """Atalho: escolhe a factory e já instancia."""
        return cls.get(cfg).build(cfg)
