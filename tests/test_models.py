"""Testes dos modelos e da factory.

O teste central é `test_backbone_identico_ao_da_parte_1`: o enunciado da Parte 2 exige
manter o encoder-decoder da Parte 1 e mudar apenas o que ele prevê. Esse teste prova, com
contagem de parâmetros, que foi isso que aconteceu.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch

from src.models import (
    BinaryHead,
    CenterOffsetHeads,
    ModelFactoryRegistry,
    PSPNet,
    ResUNet,
    Segmenter,
    SegNet,
    UNet,
    UNetImproved,
)
from src.losses.center_offset import CenterOffsetTask
from src.losses.factory import LossFactoryRegistry


class ConfigFalso:
    """Imita o TrainConfig, que é o que as factories consomem."""

    def __init__(self, model: dict):
        self.model = model


# ------------------------------------------------------------------------- UNetImproved

def test_forward_devolve_tres_saidas():
    """O quê: o modelo devolve três tensores com as formas certas.
    Por quê: é o contrato da Trilha C. Uma forma errada só apareceria como erro de
              broadcast no meio do treino, ou pior, como broadcast silencioso.
    Como: confere shapes de segmentação (1 canal), heatmap (1) e offsets (2).
    """
    modelo = UNetImproved(base=8, depth=2)
    seg, heatmap, offsets = modelo(torch.randn(2, 1, 64, 64))

    assert seg.shape == (2, 1, 64, 64)
    assert heatmap.shape == (2, 1, 64, 64)
    assert offsets.shape == (2, 2, 64, 64)


def test_backbone_identico_ao_da_parte_1():
    """O quê: encoder, bottleneck e decoder têm exatamente os mesmos parâmetros nos dois.
    Por quê: o enunciado exige "mantenham o encoder-decoder da Parte 1 e mudem o que ele
              prevê". Se o backbone crescesse, a melhora de mAP teria duas causas
              misturadas — representação nova E mais capacidade — e o argumento da
              apresentação cairia.
    Como: compara a contagem de parâmetros de cada bloco entre UNet e UNetImproved.
    """
    p1 = UNet(in_channels=1, base=16, depth=3)
    p2 = UNetImproved(1, base=16, depth=3)

    for bloco in ("encoders", "bottleneck", "ups", "decoders"):
        n1 = sum(p.numel() for p in getattr(p1, bloco).parameters())
        n2 = sum(p.numel() for p in getattr(p2, bloco).parameters())
        assert n1 == n2, f"{bloco} difere: {n1} vs {n2}"


def test_campo_receptivo_identico():
    """O quê: os dois modelos enxergam a mesma região da entrada.
    Por quê: o campo receptivo é propriedade do backbone, e a Parte 5 vai comparar esse
              número com a distribuição de tamanhos dos objetos. Se mudasse entre as
              Partes, o diagnóstico de falhas não seria comparável.
    Como: 68 px na configuração padrão, nos dois.
    """
    assert UNet(base=16, depth=3).receptive_field() == UNetImproved(base=16, depth=3).receptive_field()


def test_diferenca_esta_so_nas_cabecas():
    """O quê: toda a diferença de parâmetros vem das convoluções 1x1 finais.
    Por quê: é o número a ter na mão se perguntarem na apresentação quanto o modelo
              cresceu. Três cabeças 1x1 custam pouquíssimo perto do decoder.
    Como: total(P2) - total(P1) == cabeças(P2) - cabeça(P1).
    """
    p1 = UNet(in_channels=1, base=16, depth=3)   # backbone puro, sem cabeça
    p2 = UNetImproved(1, base=16, depth=3)

    diferenca_total = sum(p.numel() for p in p2.parameters()) - sum(p.numel() for p in p1.parameters())
    cabecas_p2 = sum(
        p.numel()
        for cabeca in (p2.seg_head, p2.heatmap_head, p2.offset_head)
        for p in cabeca.parameters()
    )

    # toda a diferença de parâmetros são as três convs 1x1 da cabeça
    assert diferenca_total == cabecas_p2
    # e isso é irrelevante perto do total (menos de 0,1%)
    assert diferenca_total / sum(p.numel() for p in p2.parameters()) < 0.001


def test_treino_de_ponta_a_ponta():
    """O quê: alvos -> perda -> backward produz gradiente nas três cabeças.
    Por quê: uma cabeça sem gradiente nunca aprende, e isso é silencioso: a perda total
              continua caindo por conta das outras duas.
    Como: um passo completo com um label map de dois objetos.
    """
    modelo = UNetImproved(base=8, depth=2)
    tarefa = CenterOffsetTask()
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[0, 5:12, 5:12] = 1
    labels[0, 20:28, 20:28] = 2

    alvos = tarefa.build_targets(labels, torch.device("cpu"))
    perda, componentes = tarefa(modelo(torch.randn(1, 1, 32, 32)), alvos)
    perda.backward()

    assert set(componentes) == {"seg", "heatmap", "offset"}
    for nome, cabeca in (("seg", modelo.seg_head), ("heatmap", modelo.heatmap_head),
                         ("offset", modelo.offset_head)):
        assert cabeca.weight.grad.abs().sum() > 0, f"sem gradiente na cabeça {nome}"


def test_decode_devolve_label_map():
    """O quê: `decode` devolve um array inteiro no formato que a métrica consome.
    Por quê: é a ponte entre a saída da rede e o mAP. Tipo ou forma errados quebrariam
              a avaliação inteira.
    Como: passa a saída de uma imagem e confere tipo e shape.
    """
    modelo = UNetImproved(base=8, depth=2)
    saidas = modelo(torch.randn(1, 1, 32, 32))
    uma_imagem = tuple(s[0] for s in saidas)

    labels = modelo.decode(uma_imagem, {"threshold": 0.5, "peak_threshold": 0.5})
    labels = np.asarray(labels)

    assert labels.shape == (32, 32)
    assert np.issubdtype(labels.dtype, np.integer)


# ------------------------------------------------------------------------------ factory

def test_factory_escolhe_pelo_nome():
    """O quê: `model.name` seleciona o backbone; `model.loss.name` seleciona a cabeça.
    Por quê: é o que permite rodar as Partes 1 e 2 com o mesmo comando, trocando só o YAML.
    Como: constrói pelo registry e confere backbone + cabeça.
    """
    base = {"base": 8, "depth": 2}

    m = ModelFactoryRegistry.build(ConfigFalso({"name": "unet", **base}))
    assert isinstance(m, Segmenter)
    assert isinstance(m.backbone, UNet)
    assert isinstance(m.head, BinaryHead)          # sem bloco loss → bce

    m2 = ModelFactoryRegistry.build(ConfigFalso({"name": "unet_improved", **base}))
    assert isinstance(m2, Segmenter)
    assert isinstance(m2.backbone, UNet)
    assert isinstance(m2.head, CenterOffsetHeads)  # unet_improved → center_offset


def test_factory_assume_unet_sem_nome():
    """O quê: config sem `model.name` constrói o backbone UNet com cabeça binária.
    Por quê: os configs da Parte 1 foram escritos antes de existir mais de um modelo e
              precisam continuar funcionando.
    Como: config sem a chave.
    """
    modelo = ModelFactoryRegistry.build(ConfigFalso({"base": 8, "depth": 2}))
    assert isinstance(modelo, Segmenter)
    assert isinstance(modelo.backbone, UNet)
    assert isinstance(modelo.head, BinaryHead)


def test_factory_rejeita_nome_desconhecido():
    """O quê: nome inválido levanta erro com a lista de opções.
    Por quê: um typo no YAML precisa falhar na hora, e não construir silenciosamente o
              modelo errado e produzir números sem sentido.
    Como: nome inexistente.
    """
    with pytest.raises(ValueError, match="desconhecido"):
        ModelFactoryRegistry.build(ConfigFalso({"name": "unet_xyz", "base": 8, "depth": 2}))


def test_factory_repassa_config_da_perda():
    """O quê: os pesos do YAML chegam na perda construída pela LossFactoryRegistry.
    Por quê: `pos_weight` é o botão da ablação do Eixo 2. Se o config fosse ignorado, a
              ablação mediria sempre a mesma coisa.
    Como: constrói com pesos específicos e lê de volta.
    """
    cfg = ConfigFalso({"name": "unet_improved", "base": 8, "depth": 2,
                       "loss": {"name": "center_offset", "w_offset": 0.25, "pos_weight": 42.0}})
    tarefa = LossFactoryRegistry.build(cfg)

    assert tarefa.criterion.w_offset == 0.25
    assert tarefa.criterion.pos_weight == 42.0


# --------------------------------------------------------- backbones, cabeças e Segmenter

@pytest.mark.parametrize("Backbone", [UNet, SegNet, ResUNet, PSPNet])
def test_backbones_todos_com_tres_cabecas(Backbone):
    """O quê: qualquer backbone + CenterOffsetHeads devolve as três saídas na forma certa.
    Por quê: a ablação agora roda segmentação de instância em todas as arquiteturas.
    Como: um forward com lote 2 e confere shapes.
    """
    bb = Backbone(base=8, depth=2)
    modelo = Segmenter(bb, CenterOffsetHeads(bb.feature_channels))
    seg, heatmap, offsets = modelo(torch.randn(2, 1, 64, 64))

    assert seg.shape == (2, 1, 64, 64)
    assert heatmap.shape == (2, 1, 64, 64)
    assert offsets.shape == (2, 2, 64, 64)


def test_segmenter_delega():
    """O quê: o Segmenter repassa receptive_field ao backbone e decode/foreground à cabeça.
    Por quê: os engines chamam esses métodos no modelo sem saber da separação.
    """
    bb = UNet(base=8, depth=2)
    modelo = Segmenter(bb, BinaryHead(bb.feature_channels))

    assert modelo.receptive_field() == UNet(base=8, depth=2).receptive_field()
    prob = modelo.foreground_prob(modelo(torch.randn(2, 1, 32, 32)))
    assert prob.shape == (2, 32, 32)

    labels = np.asarray(modelo.decode(modelo(torch.randn(1, 1, 32, 32))[0], {"threshold": 0.5}))
    assert labels.shape == (32, 32)
    assert np.issubdtype(labels.dtype, np.integer)


def test_head_selecionada_pela_loss():
    """O quê: a cabeça vem de `model.loss.name` (com os mesmos fallbacks da LossFactory)."""
    base = {"base": 8, "depth": 2}

    def cabeca(model_dict):
        return ModelFactoryRegistry.build(ConfigFalso(model_dict)).head

    assert isinstance(cabeca({"name": "unet", "loss": {"name": "bce"}, **base}), BinaryHead)
    assert isinstance(cabeca({"name": "unet", "loss": {"name": "center_offset"}, **base}),
                      CenterOffsetHeads)
    assert isinstance(cabeca({"name": "unet", **base}), BinaryHead)
    assert isinstance(cabeca({"name": "unet_improved", **base}), CenterOffsetHeads)
    assert isinstance(cabeca({"name": "segnet", "loss": {"w_offset": 0.1}, **base}),
                      CenterOffsetHeads)


# ---------------------------------------------------------------------------------- PSPNet

def test_pspnet_contrato():
    """O quê: o PSPNet é um backbone — features na resolução da entrada, RF inteiro."""
    bb = PSPNet(base=8, depth=2)
    f = bb.features(torch.randn(2, 1, 64, 64))

    assert f.shape == (2, 8, 64, 64)
    assert bb.feature_channels == 8
    assert isinstance(bb.receptive_field(), int) and bb.receptive_field() > 0


def test_pspnet_segmenter_treina():
    """O quê: PSPNet + CenterOffsetHeads treina um passo com gradiente nas três convs.
    Como: lote 2 (o ramo bin=1 do PPM tem BatchNorm sobre célula 1x1).
    """
    bb = PSPNet(base=8, depth=2)
    modelo = Segmenter(bb, CenterOffsetHeads(bb.feature_channels))
    tarefa = CenterOffsetTask()

    labels = torch.zeros(2, 64, 64, dtype=torch.long)
    labels[:, 8:20, 8:20] = 1
    labels[:, 36:52, 36:52] = 2

    alvos = tarefa.build_targets(labels, torch.device("cpu"))
    perda, componentes = tarefa(modelo(torch.randn(2, 1, 64, 64)), alvos)
    perda.backward()

    assert set(componentes) == {"seg", "heatmap", "offset"}
    for nome, conv in (("seg", modelo.head.seg_head), ("heatmap", modelo.head.heatmap_head),
                       ("offset", modelo.head.offset_head)):
        assert conv.weight.grad.abs().sum() > 0, f"sem gradiente na cabeça {nome}"


def test_pspnet_small_input():
    """O quê: entrada pequena (bottleneck menor que a maior grade do PPM) não quebra."""
    bb = PSPNet(base=8, depth=3)
    modelo = Segmenter(bb, CenterOffsetHeads(bb.feature_channels))
    seg, heatmap, offsets = modelo(torch.randn(2, 1, 32, 32))
    assert seg.shape == (2, 1, 32, 32)
