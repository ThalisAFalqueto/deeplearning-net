"""Escolhe a perda a partir do config — mesmo padrão do ``ModelFactoryRegistry``.

A perda deixou de ser responsabilidade do modelo: o YAML declara ``model.loss.name`` e esta
factory instancia o objeto de tarefa correspondente. O modelo só expõe arquitetura.

    bce             1 saída      BCEWithLogits                    (Partes 0 e 1)
    center_offset   3 saídas     CE + L2(heatmap) + L1(offsets)   (Trilha C, Parte 2)

O nome resolvido aqui é a **fonte única de verdade**: a mesma função escolhe a perda e a
cabeça do modelo (ver ``src/models/factory.py``). Uma ``BinaryHead`` com perda
``center_offset`` — ou o contrário — nunca é uma combinação válida, então não há uma chave
``model.head`` separada para dessincronizar.
"""

from abc import ABC, abstractmethod

from src.losses.bce import BCELoss
from src.losses.center_offset import CenterOffsetTask

# Chaves que só fazem sentido para a perda combinada. Se aparecerem no bloco `loss` sem um
# `name` explícito, é center_offset. `pos_weight` fica de fora: BCEWithLogits também o aceita.
_WEIGHT_KEYS = {"w_seg", "w_heatmap", "w_offset"}


def resolve_task_name(cfg) -> str:
    """Nome da tarefa a partir de ``cfg.model``, sem levantar erro.

    Prioridade:
        1. ``model.loss.name`` explícito.
        2. o bloco ``model.loss`` traz peso de componente (``w_seg`` etc.) → ``center_offset``.
        3. heurística pelo ``model.name``: ``unet_improved`` → ``center_offset``; resto → ``bce``.

    A regra 3 mantém configs e checkpoints escritos antes desta refatoração funcionando sem
    edição. A resolução é leitura pura — nunca muta ``cfg.model`` (que é compartilhado entre
    o TrainConfig e o EvalConfig do mesmo YAML).
    """
    loss_cfg = cfg.model.get("loss") or {}
    if loss_cfg.get("name"):
        return loss_cfg["name"]
    if _WEIGHT_KEYS & set(loss_cfg):
        return "center_offset"
    return "center_offset" if cfg.model.get("name") == "unet_improved" else "bce"


def _loss_params(cfg) -> dict:
    """Parâmetros do bloco ``model.loss`` sem a chave organizacional ``name``."""
    return {k: v for k, v in (cfg.model.get("loss") or {}).items() if k != "name"}


class LossFactory(ABC):
    """Contrato comum das factories de perda."""

    @abstractmethod
    def build(self, cfg):
        """Instancia o objeto de tarefa a partir de um TrainConfig ou EvalConfig."""


class BCELossFactory(LossFactory):
    def build(self, cfg):
        return BCELoss(**_loss_params(cfg))


class CenterOffsetLossFactory(LossFactory):
    def build(self, cfg):
        return CenterOffsetTask(**_loss_params(cfg))


class LossFactoryRegistry:
    _factories = {
        "bce": BCELossFactory(),
        "center_offset": CenterOffsetLossFactory(),
    }

    @classmethod
    def get(cls, cfg) -> LossFactory:
        """Escolhe a factory pelo nome resolvido de ``cfg.model.loss``."""
        nome = resolve_task_name(cfg)
        if nome not in cls._factories:
            raise ValueError(
                f"loss desconhecida: {nome!r}. "
                f"Disponíveis: {sorted(cls._factories)}"
            )
        return cls._factories[nome]

    @classmethod
    def build(cls, cfg):
        """Atalho: escolhe a factory e já instancia."""
        return cls.get(cfg).build(cfg)
