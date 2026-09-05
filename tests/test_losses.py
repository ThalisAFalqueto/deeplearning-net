"""Testes da perda combinada da Trilha C."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch

from src.data.targets import batch_center_offset_targets
from src.losses.center_offset import CenterOffsetLoss


def labels_com_dois_discos(size=48):
    labels = np.zeros((size, size), dtype=np.int64)
    y, x = np.ogrid[:size, :size]
    labels[(y - 15) ** 2 + (x - 15) ** 2 <= 36] = 1
    labels[(y - 32) ** 2 + (x - 32) ** 2 <= 49] = 2
    return torch.from_numpy(labels)


def saidas_perfeitas(alvos):
    """Constrói as três saídas da rede que reproduzem exatamente os alvos.

    A perda de segmentação usa BCEWithLogits e a do heatmap aplica sigmoid, então as duas
    precisam do logit inverso do alvo. Os offsets são previstos direto.
    """
    def inv_sigmoid(p, eps=1e-6):
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))

    return (inv_sigmoid(alvos["foreground"]),
            inv_sigmoid(alvos["heatmap"]),
            alvos["offsets"])


@pytest.fixture
def alvos():
    return batch_center_offset_targets(labels_com_dois_discos().unsqueeze(0))


def test_alvos_perfeitos_dao_perda_quase_zero(alvos):
    """O quê: prever exatamente o alvo zera as três componentes.
    Por quê: é o teste de sanidade da perda. Se o mínimo não estiver no alvo, a rede
              converge para outra coisa e nada denuncia.
    Como: constrói os logits que reproduzem os alvos e confere as três componentes.
    """
    total, comp = CenterOffsetLoss()(*saidas_perfeitas(alvos), alvos)

    assert comp["seg"] < 1e-4
    assert comp["heatmap"] < 1e-6
    assert comp["offset"] < 1e-6
    assert float(total) < 1e-3


def test_heatmap_zerado_e_penalizado(alvos):
    """O quê: prever heatmap todo zero produz perda maior que prever o alvo.
    Por quê: é a solução degenerada que o desbalanceamento torna atraente — o pico é 1
              pixel em 65.536, e prever zero em tudo dá erro médio minúsculo.
    Como: compara a perda do heatmap correto com a do heatmap zerado.
    """
    perda = CenterOffsetLoss()
    correto = perda.heatmap_loss(alvos["heatmap"], alvos["heatmap"])
    zerado = perda.heatmap_loss(torch.zeros_like(alvos["heatmap"]), alvos["heatmap"])

    assert float(correto) == pytest.approx(0.0, abs=1e-6)
    assert float(zerado) > float(correto)


def test_pos_weight_aumenta_a_penalidade_nos_centros(alvos):
    """O quê: pos_weight maior penaliza mais o mesmo erro sobre os centros.
    Por quê: é o parâmetro que permite a ablação do Eixo 2 (efeito do desbalanceamento).
              Se ele não mudar nada, a ablação mediria ruído.
    Como: mesmo erro (heatmap zerado) sob três valores de pos_weight.
    """
    zerado = torch.zeros_like(alvos["heatmap"])
    perdas = [
        float(CenterOffsetLoss(pos_weight=w).heatmap_loss(zerado, alvos["heatmap"]))
        for w in (0.0, 10.0, 100.0)
    ]
    assert perdas[0] < perdas[1] < perdas[2]


def test_offset_ignora_o_fundo(alvos):
    """O quê: erro de offset em pixel de fundo não afeta a perda.
    Por quê: pixel de fundo não pertence a objeto nenhum — o alvo zero ali é convenção, não
              resposta certa. Se o fundo entrasse, a rede gastaria capacidade prevendo zero
              em 75% da imagem e o gradiente dos pixels úteis seria diluído.
    Como: estraga os offsets só onde não há foreground e confere que a perda não muda.
    """
    perda = CenterOffsetLoss()
    pred = alvos["offsets"].clone()
    limpa = perda.offset_loss(pred, alvos["offsets"], alvos["foreground"])

    fundo = alvos["foreground"].expand_as(pred) == 0
    pred[fundo] = 999.0
    suja = perda.offset_loss(pred, alvos["offsets"], alvos["foreground"])

    assert float(limpa) == pytest.approx(float(suja))


def test_offset_penaliza_erro_no_foreground(alvos):
    """O quê: erro dentro do foreground aumenta a perda.
    Por quê: o complemento do teste anterior — mascarar não pode ter mascarado demais.
    Como: soma 1 px de erro a todos os offsets; a L1 média deve refletir isso.
    """
    perda = CenterOffsetLoss()
    errado = alvos["offsets"] + 1.0
    assert float(perda.offset_loss(errado, alvos["offsets"], alvos["foreground"])) > 0.9


def test_imagem_sem_foreground_nao_explode():
    """O quê: imagem sem nenhum objeto não causa divisão por zero.
    Por quê: a L1 dos offsets divide pelo número de pixels de foreground. Vazio -> 0/0.
    Como: label map todo zero.
    """
    alvos = batch_center_offset_targets(torch.zeros(1, 32, 32, dtype=torch.long))
    z = torch.zeros(1, 1, 32, 32)
    total, comp = CenterOffsetLoss()(z, z, torch.zeros(1, 2, 32, 32), alvos)
    assert torch.isfinite(total)
    assert comp["offset"] == pytest.approx(0.0)


def test_gradiente_flui(alvos):
    """O quê: backward() produz gradiente não nulo nos quatro canais.
    Por quê: um canal sem gradiente nunca aprende, e isso é silencioso — a perda total
              continua caindo por conta dos outros.
    Como: logits com requires_grad; confere gradiente em cada canal separadamente.
    """
    seg = torch.zeros(1, 1, 48, 48, requires_grad=True)
    hm = torch.zeros(1, 1, 48, 48, requires_grad=True)
    off = torch.zeros(1, 2, 48, 48, requires_grad=True)
    total, _ = CenterOffsetLoss()(seg, hm, off, alvos)
    total.backward()

    for tensor, nome in ((seg, "segmentação"), (hm, "heatmap"), (off, "offsets")):
        assert tensor.grad.abs().sum() > 0, f"sem gradiente em {nome}"


def test_pesos_alteram_a_contribuicao(alvos):
    """O quê: zerar o peso de uma componente remove a contribuição dela do total.
    Por quê: os pesos existem para equilibrar três perdas de escalas diferentes. Se não
              funcionassem, ajustar o equilíbrio seria impossível.
    Como: compara o total com w_offset=0 contra a soma das outras duas componentes.
    """
    saidas = (torch.randn(1, 1, 48, 48), torch.randn(1, 1, 48, 48), torch.randn(1, 2, 48, 48))
    total_sem_offset, comp = CenterOffsetLoss(w_offset=0.0)(*saidas, alvos)
    assert float(total_sem_offset) == pytest.approx(comp["seg"] + comp["heatmap"], rel=1e-5)
