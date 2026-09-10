"""Testes da LossFactoryRegistry — a perda escolhida pelo config, desacoplada do modelo."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import pytest
import torch

from src.losses.factory import LossFactoryRegistry, resolve_task_name


class ConfigFalso:
    def __init__(self, model: dict):
        self.model = model


def test_loss_factory_bce():
    """O quê: a tarefa `bce` constrói alvo binário e devolve (perda, {'bce': ...})."""
    tarefa = LossFactoryRegistry.build(ConfigFalso({"name": "unet", "loss": {"name": "bce"}}))

    alvo = tarefa.build_targets(torch.zeros(2, 16, 16, dtype=torch.long), torch.device("cpu"))
    assert alvo.shape == (2, 1, 16, 16)

    perda, comp = tarefa(torch.zeros(2, 1, 16, 16), alvo)
    assert torch.isfinite(perda)
    assert set(comp) == {"bce"}


def test_loss_factory_center_offset():
    """O quê: a tarefa `center_offset` monta o dict de alvos e pontua a tupla de 3 saídas."""
    tarefa = LossFactoryRegistry.build(
        ConfigFalso({"name": "unet_improved", "loss": {"name": "center_offset"}})
    )

    labels = torch.zeros(2, 24, 24, dtype=torch.long)
    labels[:, 4:10, 4:10] = 1
    alvos = tarefa.build_targets(labels, torch.device("cpu"))
    assert set(alvos) == {"foreground", "heatmap", "offsets"}
    assert alvos["foreground"].shape == (2, 1, 24, 24)
    assert alvos["offsets"].shape == (2, 2, 24, 24)

    saidas = (torch.zeros(2, 1, 24, 24), torch.zeros(2, 1, 24, 24), torch.zeros(2, 2, 24, 24))
    perda, comp = tarefa(saidas, alvos)
    assert torch.isfinite(perda)
    assert set(comp) == {"seg", "heatmap", "offset"}


def test_loss_factory_infere_sem_name():
    """O quê: sem `loss.name`, o nome é inferido — chave de peso ou heurística por model.name."""
    assert resolve_task_name(ConfigFalso({"name": "unet_improved"})) == "center_offset"
    assert resolve_task_name(ConfigFalso({"name": "segnet"})) == "bce"
    assert resolve_task_name(ConfigFalso({"name": "resunet", "loss": {"pos_weight": 50}})) == "bce"
    assert resolve_task_name(ConfigFalso({"name": "segnet", "loss": {"w_seg": 1.0}})) == "center_offset"


def test_loss_factory_rejeita_desconhecida():
    """O quê: `loss.name` inválido falha na hora, com a lista de opções."""
    with pytest.raises(ValueError, match="desconhecida"):
        LossFactoryRegistry.build(ConfigFalso({"name": "unet", "loss": {"name": "focal"}}))
