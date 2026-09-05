"""Tarefas: o que a rede prevê, qual perda otimiza isso, e como decodificar em objetos.

São as três decisões acopladas que o enunciado cobra. Cada Parte do trabalho escolhe uma
combinação diferente, e o resto do pipeline — dados, modelo, treino, métricas, figuras —
permanece idêntico:

    binary          1 canal    BCE                        limiar + componentes conexos
    center_offset   4 canais   BCE + L2 + L1              picos + atribuição ao centro

Trocar de tarefa é trocar uma linha no YAML. É isso que permite comparar as duas com as
mesmas métricas lado a lado, como o enunciado exige na Parte 2.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from src.data.targets import batch_center_offset_targets
from src.losses.center_offset import CenterOffsetLoss
from src.utils import labels_from_probability


class Task(ABC):
    """Contrato comum entre as tarefas."""

    out_channels: int

    @abstractmethod
    def build_targets(self, labels: torch.Tensor, device) -> dict:
        """Constrói os alvos a partir do label map do gabarito, (B, H, W)."""

    @abstractmethod
    def loss(self, logits: torch.Tensor, targets) -> tuple[torch.Tensor, dict]:
        """Perda total e as componentes (para o log por época)."""

    @abstractmethod
    def foreground_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Probabilidade de foreground, (B, H, W) — usada pelo IoU/Dice da validação."""

    @abstractmethod
    def decode(self, logits: torch.Tensor, decode_cfg: dict) -> np.ndarray:
        """Converte a saída de UMA imagem, (C, H, W), em label map de instâncias."""


class BinaryTask(Task):
    """Parte 1: um logit de foreground, decodificado por componentes conexos."""

    out_channels = 1

    def __init__(self, cfg: dict | None = None):
        self._bce = nn.BCEWithLogitsLoss()

    def build_targets(self, labels, device):
        return (labels > 0).float().unsqueeze(1).to(device)

    def loss(self, logits, targets):
        perda = self._bce(logits, targets)
        return perda, {"bce": float(perda.detach())}

    def foreground_prob(self, logits):
        return torch.sigmoid(logits)[:, 0]

    def decode(self, logits, decode_cfg):
        prob = torch.sigmoid(logits)[0].cpu().numpy()
        return labels_from_probability(prob, decode_cfg["threshold"])


class CenterOffsetTask(Task):
    """Parte 2, Trilha C: foreground + heatmap de centros + offsets."""

    out_channels = 4

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self._loss = CenterOffsetLoss(
            w_seg=cfg.get("w_seg", 1.0),
            w_heatmap=cfg.get("w_heatmap", 1.0),
            w_offset=cfg.get("w_offset", 1.0),
            pos_weight=cfg.get("pos_weight", 0.0),
        )

    def build_targets(self, labels, device):
        alvos = batch_center_offset_targets(labels)
        return {chave: valor.to(device) for chave, valor in alvos.items()}

    def loss(self, logits, targets):
        return self._loss(logits, targets)

    def foreground_prob(self, logits):
        return torch.sigmoid(logits)[:, 0]

    def decode(self, logits, decode_cfg):
        # importado aqui porque é a peça em desenvolvimento; assim o resto do pipeline
        # continua importável mesmo antes de ela existir
        from src.utils.decode_center_offset import decode_center_offset

        return decode_center_offset(
            logits[0].cpu(),
            logits[1].cpu(),
            logits[2:4].cpu(),
            fg_threshold=decode_cfg.get("threshold", 0.5),
            peak_threshold=decode_cfg.get("peak_threshold", 0.3),
            nms_kernel=decode_cfg.get("nms_kernel", 3),
        )


_TASKS = {"binary": BinaryTask, "center_offset": CenterOffsetTask}


def get_task(cfg) -> Task:
    """Instancia a tarefa declarada no config.

    Args:
        cfg: TrainConfig ou EvalConfig. Lê ``model["task"]``; sem essa chave, assume
            ``binary`` — o que mantém os configs da Parte 1 funcionando sem alteração.

    Returns:
        Instância de :class:`Task`.
    """
    nome = cfg.model.get("task", "binary")
    if nome not in _TASKS:
        raise ValueError(
            f"task desconhecida: {nome!r}. Disponíveis: {sorted(_TASKS)}"
        )

    task = _TASKS[nome](cfg.model.get("loss", {}))

    esperado = task.out_channels
    if cfg.model["out_channels"] != esperado:
        raise ValueError(
            f"task {nome!r} precisa de out_channels={esperado}, "
            f"mas o config tem {cfg.model['out_channels']}"
        )
    return task
