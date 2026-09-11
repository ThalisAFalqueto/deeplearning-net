"""Testes da U-Net + Pyramid Pooling Module (Eixo 3 da Parte 3).

O teste central é ``test_so_o_ppm_difere_da_unet``: para a ablação responder à pergunta do
enunciado (contexto global ajuda a separar instâncias?), a única diferença para a baseline
tem que ser o PPM. ``test_ppm_traz_contexto_global`` mostra o que o PPM acrescenta: um pixel
de saída passa a depender da imagem inteira.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import pytest
import torch

from src.losses.center_offset import CenterOffsetTask
from src.models import (
    BinaryHead,
    CenterOffsetHeads,
    ModelFactoryRegistry,
    Segmenter,
    UNet,
    UNetPPM,
)


class ConfigFalso:
    """Imita o TrainConfig, que é o que as factories consomem."""

    def __init__(self, model: dict):
        self.model = model


def test_contrato_de_backbone():
    """O quê: features na resolução da entrada, com a largura que as cabeças esperam.
    Por quê: as cabeças são as mesmas de todas as arquiteturas; se a largura mudasse, a
              factory montaria uma cabeça incompatível.
    Como: forward num lote de 2 (o ramo de bin 1×1 do PPM tem BatchNorm).
    """
    bb = UNetPPM(base=8, depth=2)
    f = bb.features(torch.randn(2, 1, 64, 64))

    assert f.shape == (2, 8, 64, 64)
    assert bb.feature_channels == 8


def test_so_o_ppm_difere_da_unet():
    """O quê: encoder, bottleneck e decoder são os da U-Net, parâmetro por parâmetro.
    Por quê: é o que torna a comparação com a baseline uma ablação do PPM — e o que a
              diferencia da PSPNet do projeto, que tira o decoder junto.
    Como: compara nomes e formas do state_dict, tirando as chaves ``ppm.*``.
    """
    unet = UNet(base=16, depth=3).state_dict()
    ppm = UNetPPM(base=16, depth=3).state_dict()

    sem_ppm = {k: v.shape for k, v in ppm.items() if not k.startswith("ppm.")}
    assert sem_ppm == {k: v.shape for k, v in unet.items()}
    assert any(k.startswith("ppm.") for k in ppm)


def test_campo_receptivo():
    """O quê: o campo receptivo local é o da U-Net mais a conv 3×3 do PPM no bottleneck.
    Por quê: a Parte 5 compara esse número com o diâmetro dos núcleos.
    Como: base 16 / depth 3 — U-Net 68 px; a conv 3×3 com salto 8 soma (3-1)·8 = 16.
    """
    assert UNet(base=16, depth=3).receptive_field() == 68
    assert UNetPPM(base=16, depth=3).receptive_field() == 84


def test_ppm_traz_contexto_global():
    """O quê: mexer na metade de baixo da imagem muda a saída no pixel (0, 0) só com o PPM.
    Por quê: é a propriedade que o Eixo 3 testa. Na U-Net cada pixel de saída vê uma janela
              limitada; o bin 1×1 do PPM é a média da imagem inteira.
    Como: os mesmos pesos nas duas redes (a U-Net copia tudo menos o PPM), modo eval (o
          BatchNorm usa as estatísticas guardadas e não mistura pixels), perturbação nas
          linhas 64–127 e leitura do pixel (0, 0).

    O teste compara zero com não-zero, não tamanhos: com pesos aleatórios o efeito do PPM é
    pequeno (~1e-7 numa saída ~1e-2). Mas se o pixel não dependesse da região perturbada, a
    conta seria feita sobre exatamente os mesmos números e daria o mesmo resultado bit a bit
    — é o que acontece na U-Net. Qualquer diferença é dependência real.
    """
    torch.manual_seed(0)
    com_ppm = UNetPPM(base=8, depth=2).eval()
    sem_ppm = UNet(base=8, depth=2).eval()
    sem_ppm.load_state_dict(
        {k: v for k, v in com_ppm.state_dict().items() if not k.startswith("ppm.")}
    )

    x = torch.randn(1, 1, 128, 128)
    x_perturbado = x.clone()
    x_perturbado[..., 64:, :] += 10.0

    with torch.no_grad():
        delta_unet = (sem_ppm(x) - sem_ppm(x_perturbado))[..., 0, 0].abs().max()
        delta_ppm = (com_ppm(x) - com_ppm(x_perturbado))[..., 0, 0].abs().max()

    assert delta_unet == 0            # fora do alcance da U-Net: exatamente igual
    assert delta_ppm > 0              # com o PPM, o pixel (0, 0) muda


@pytest.mark.parametrize("perda, cabeca", [("bce", BinaryHead),
                                           ("center_offset", CenterOffsetHeads)])
def test_factory_monta(perda, cabeca):
    """O quê: ``model.name: unet_ppm`` monta um Segmenter com a cabeça escolhida pela perda.
    Por quê: é o que os configs e o runner de ablação usam.
    """
    modelo = ModelFactoryRegistry.build(
        ConfigFalso({"name": "unet_ppm", "base": 8, "depth": 2, "loss": {"name": perda}})
    )
    assert isinstance(modelo, Segmenter)
    assert isinstance(modelo.backbone, UNetPPM)
    assert isinstance(modelo.head, cabeca)


def test_treina_um_passo_com_gradiente_no_ppm():
    """O quê: um passo de treino da Trilha C leva gradiente até o PPM e às três cabeças.
    Por quê: um módulo fora do caminho do gradiente não aprenderia nada — a ablação mediria
              ruído.
    Como: lote 2, dois quadrados como instâncias, backward da perda combinada.
    """
    bb = UNetPPM(base=8, depth=2)
    modelo = Segmenter(bb, CenterOffsetHeads(bb.feature_channels))
    tarefa = CenterOffsetTask()

    labels = torch.zeros(2, 64, 64, dtype=torch.long)
    labels[:, 8:20, 8:20] = 1
    labels[:, 36:52, 36:52] = 2

    alvos = tarefa.build_targets(labels, torch.device("cpu"))
    perda, _ = tarefa(modelo(torch.randn(2, 1, 64, 64)), alvos)
    perda.backward()

    for nome, p in bb.ppm.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"sem gradiente em ppm.{nome}"
    for conv in (modelo.head.seg_head, modelo.head.heatmap_head, modelo.head.offset_head):
        assert conv.weight.grad.abs().sum() > 0
