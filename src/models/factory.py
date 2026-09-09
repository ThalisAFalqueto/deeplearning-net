"""Constrói o modelo a partir da configuração.

Mesmo padrão do ``DatasetFactoryRegistry`` em ``src/data/factory.py``: trocar ``model.name``
no YAML troca a arquitetura, e com ela o que a rede prevê, qual perda otimiza isso e como
decodificar a saída em objetos — porque cada modelo carrega esse contrato.

    unet            1 saída     BCE                  limiar + componentes conexos
    segnet          1 saída     BCE                  limiar + componentes conexos
    resunet         1 saída     BCE                  limiar + componentes conexos
    unet_improved   3 saídas    CE + L2 + L1         picos + atribuição ao centro

É isso que permite rodar as Partes 1, 2 e 3 com o mesmo comando, mudando só o config, e
comparar as métricas lado a lado como o enunciado exige.
"""

from abc import ABC, abstractmethod

import torch.nn as nn

from src.models.unet import UNet
from src.models.unet_improved import UNetImproved
from src.models.segnet import SegNet
from src.models.resunet import ResUNet


class ModelFactory(ABC):
    """Contrato comum das factories de modelo."""

    @abstractmethod
    def build(self, cfg) -> nn.Module:
        """Instancia o modelo a partir de um TrainConfig ou EvalConfig."""


class UNetFactory(ModelFactory):
    """U-Net binária das Partes 0 e 1."""

    def build(self, cfg) -> nn.Module:
        return UNet(
            in_channels=cfg.model.get("in_channels", 1),
            out_channels=cfg.model.get("out_channels", 1),
            base=cfg.model["base"],
            depth=cfg.model["depth"],
        )


class UNetImprovedFactory(ModelFactory):
    """U-Net de três cabeças da Parte 2 (Trilha C)."""

    def build(self, cfg) -> nn.Module:
        return UNetImproved(
            in_channels=cfg.model.get("in_channels", 1),
            base=cfg.model["base"],
            depth=cfg.model["depth"],
            loss_cfg=cfg.model.get("loss", {}),
        )


class SegNetFactory(ModelFactory):
    """SegNet — max unpooling com índices no decoder."""

    def build(self, cfg) -> nn.Module:
        return SegNet(
            in_channels=cfg.model.get("in_channels", 1),
            out_channels=cfg.model.get("out_channels", 1),
            base=cfg.model["base"],
            depth=cfg.model["depth"],
        )


class ResUNetFactory(ModelFactory):
    """ResUNet — U-Net com Residual Blocks."""

    def build(self, cfg) -> nn.Module:
        return ResUNet(
            in_channels=cfg.model.get("in_channels", 1),
            out_channels=cfg.model.get("out_channels", 1),
            base=cfg.model["base"],
            depth=cfg.model["depth"],
        )


class ModelFactoryRegistry:
    _factories = {
        "unet": UNetFactory(),
        "segnet": SegNetFactory(),
        "resunet": ResUNetFactory(),
        "unet_improved": UNetImprovedFactory(),
    }

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
