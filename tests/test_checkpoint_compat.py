"""Testes do carregamento de checkpoints no formato antigo de nomes de pesos.

A separação dos modelos em ``Segmenter(backbone, cabeça)`` renomeou as chaves do
``state_dict`` sem mudar nenhum tensor. Estes testes garantem que checkpoints anteriores
continuam carregando — e que o carregamento continua estrito para checkpoints errados.

Os testes não dependem de arquivos ``.pth`` (que não vão para o git): o formato antigo é
reconstruído a partir de um modelo atual, removendo os prefixos novos.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import pytest
import torch

from src.models.checkpoint import load_model_weights, remap_legacy_keys
from src.models.factory import ModelFactoryRegistry


def construir(tarefa: str):
    """Modelo atual pela factory: tarefa 'bce' (binário) ou 'center_offset' (Trilha C)."""
    nome = "unet" if tarefa == "bce" else "unet_improved"
    cfg = SimpleNamespace(model={"name": nome, "base": 8, "depth": 2, "loss": {"name": tarefa}})
    return ModelFactoryRegistry.build(cfg)


def formato_antigo(state_dict: dict) -> dict:
    """Reproduz os nomes de antes da refatoração: tira UM prefixo 'backbone.' ou 'head.'.

    ``backbone.encoders.0...`` -> ``encoders.0...``; ``head.head.weight`` -> ``head.weight``;
    ``head.seg_head.weight`` -> ``seg_head.weight``.
    """
    antigo = {}
    for chave, valor in state_dict.items():
        for prefixo in ("backbone.", "head."):
            if chave.startswith(prefixo):
                chave = chave[len(prefixo):]
                break
        antigo[chave] = valor
    return antigo


@pytest.mark.parametrize("tarefa", ["bce", "center_offset"])
def test_formato_antigo_carrega_com_os_mesmos_valores(tarefa):
    """O quê: um state_dict com nomes antigos carrega e reproduz exatamente os pesos.
    Por quê: é o que faz os checkpoints das Partes 0, 1 e 2 voltarem a funcionar.
    Como: exporta um modelo, renomeia para o formato antigo, carrega num modelo novo e
          compara tensor a tensor.
    """
    origem = construir(tarefa)
    destino = construir(tarefa)

    remapeado = load_model_weights(destino, formato_antigo(origem.state_dict()))

    assert remapeado
    for chave, valor in origem.state_dict().items():
        assert torch.equal(valor, destino.state_dict()[chave]), chave


@pytest.mark.parametrize("tarefa", ["bce", "center_offset"])
def test_formato_atual_carrega_sem_remapear(tarefa):
    """O quê: checkpoint já no formato atual carrega direto, sem tradução.
    Por quê: os checkpoints das ablações de 10/09 já estão no formato novo; o carregador
              não pode mexer neles.
    Como: exporta e reimporta sem renomear.
    """
    origem, destino = construir(tarefa), construir(tarefa)
    assert load_model_weights(destino, origem.state_dict()) is False


def test_cabeca_errada_e_recusada():
    """O quê: pesos da cabeça binária não carregam num modelo da Trilha C.
    Por quê: o carregamento tem que continuar estrito — carregar pela metade em silêncio
              produziria uma rede com cabeças aleatórias e números sem sentido.
    Como: state_dict antigo do binário num modelo de três cabeças.
    """
    binario = formato_antigo(construir("bce").state_dict())
    with pytest.raises(KeyError):
        remap_legacy_keys(binario, construir("center_offset"))


def test_chave_desconhecida_e_recusada():
    """O quê: uma chave que não corresponde a nada no modelo levanta erro.
    Por quê: indica checkpoint de outra arquitetura; melhor falhar do que ignorar.
    Como: acrescenta uma chave inventada ao state_dict.
    """
    estado = construir("bce").state_dict()
    estado["camada_que_nao_existe.weight"] = torch.zeros(1)
    with pytest.raises(KeyError):
        remap_legacy_keys(estado, construir("bce"))


CHECKPOINTS_LOCAIS = [
    ("outputs/p1/best.pth", "bce"),
    ("outputs/p0/best.pth", "bce"),
    ("outputs/p2_synthetic/best.pth", "center_offset"),
]


@pytest.mark.parametrize("caminho, tarefa", CHECKPOINTS_LOCAIS)
def test_checkpoints_reais_anteriores_carregam(caminho, tarefa):
    """O quê: os checkpoints reais das Partes 0–2 carregam, estritamente.
    Por quê: é o caso de uso de verdade. Os arquivos não vão para o git, então o teste é
              pulado em máquinas que não os têm. Um checkpoint retreinado depois de 10/09 já
              está no formato novo — por isso o teste não exige que tenha havido remapeamento
              (quem cobre o formato antigo é o primeiro teste deste arquivo).
    Como: carrega no modelo base 16 / depth 3, que é a configuração em que foram treinados;
          ``load_model_weights`` levanta erro se alguma chave ou forma não bater.
    """
    arquivo = _TESTS_DIR.parent / caminho
    if not arquivo.exists():
        pytest.skip(f"{caminho} não existe nesta máquina")

    nome = "unet" if tarefa == "bce" else "unet_improved"
    cfg = SimpleNamespace(model={"name": nome, "base": 16, "depth": 3, "loss": {"name": tarefa}})
    modelo = ModelFactoryRegistry.build(cfg)
    ck = torch.load(arquivo, map_location="cpu", weights_only=False)
    load_model_weights(modelo, ck["model"])
