"""Testes da métrica — rodam sem treinar nada.

Estes testes existem porque a métrica é a peça mais perigosa do trabalho: se ela estiver
errada, TODO número da apresentação está errado e nada denuncia. Um matching com bug
produz números plausíveis.

Rode com:   pytest tests/ -v
"""

import sys
from pathlib import Path

# Adiciona o próprio diretório (tests/) e o pai (deeplearning-net/) ao path
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TESTS_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_TESTS_DIR))

import torch
import pytest

from helpers.setup_environment import setup_environment
setup_environment()

from src.metrics import instance, semantic


# --------------------------------------------------------------------------- helpers

def three_touching_blocks(size: int = 3):
    """3 objetos `size`x`size` encostados em fila — o caso central do trabalho.

        111222333
        111222333
        111222333
    """
    gt = torch.zeros((size, size * 3), dtype=torch.int64)
    for k in range(3):
        gt[:, k * size:(k + 1) * size] = k + 1
    return gt


def object_with_exact_iou(target_iou: float, canvas=(20, 20)):
    """Um objeto previsto contido no real, com IoU exatamente `target_iou`.

    O gabarito é um bloco 10x10 (100 px). A predição são os primeiros k px dele,
    com k = target_iou * 100. Como pred ⊂ gt: IoU = k/100.
    """
    gt = torch.zeros(canvas, dtype=torch.int64)
    gt[0:10, 0:10] = 1

    k = int(round(target_iou * 100))
    pred = torch.zeros(canvas, dtype=torch.int64)
    for r, c in torch.argwhere(gt == 1)[:k]:
        pred[r, c] = 1
    return pred, gt


# ------------------------------------------------------------------ IoU e Dice (pixels)

@pytest.fixture
def to_binary():
    return semantic.ToBinary()


@pytest.fixture
def iou_metric():
    return semantic.IoU()


@pytest.fixture
def dice_metric():
    return semantic.Dice()


@pytest.fixture
def confusion():
    return semantic.Confusion()


def test_to_binary_separa_fundo_de_objeto(to_binary):
    labels = torch.tensor([[0, 1],
                           [2, 0]])
    esperado = torch.tensor([[False, True],
                             [True, False]])
    assert torch.equal(to_binary(labels), esperado)


def test_to_binary_tudo_fundo(to_binary):
    vazio = torch.zeros((3, 3), dtype=torch.int64)
    assert not to_binary(vazio).any()


def test_to_binary_labels_nao_contiguos(to_binary):
    """Labels 1, 5, 9 são todos objeto — só o 0 é fundo."""
    labels = torch.tensor([[1, 5],
                           [9, 0]])
    esperado = torch.tensor([[True, True],
                             [True, False]])
    assert torch.equal(to_binary(labels), esperado)


def test_to_binary_preserva_forma_e_devolve_bool(to_binary):
    labels = torch.tensor([[0, 1, 2],
                           [3, 0, 4]])
    r = to_binary(labels)
    assert r.shape == labels.shape
    assert r.dtype == torch.bool


def test_iou_dice_identicos(iou_metric, dice_metric):
    gt = three_touching_blocks()
    mask = semantic.ToBinary()(gt)
    assert iou_metric(mask, mask) == pytest.approx(1.0)
    assert dice_metric(mask, mask) == pytest.approx(1.0)


def test_iou_dice_grade_6x6(confusion, iou_metric, dice_metric):
    """Dois quadrados 3x3 deslocados de (1,1): TP=4, FP=5, FN=5."""
    gt = torch.zeros((6, 6), dtype=torch.bool)
    gt[1:4, 1:4] = True
    pred = torch.zeros((6, 6), dtype=torch.bool)
    pred[2:5, 2:5] = True

    assert confusion(pred, gt) == (4, 5, 5)
    assert iou_metric(pred, gt) == pytest.approx(4 / 14)
    assert dice_metric(pred, gt) == pytest.approx(8 / 18)


# ------------------------------------------------------------------------ matriz de IoU

@pytest.fixture
def iou_matrix():
    return instance.IoUMatrix()


def test_iou_matrix_identidade(iou_matrix):
    """Gabarito contra ele mesmo: diagonal 1, resto 0."""
    gt = three_touching_blocks()
    m = iou_matrix(gt, gt)
    assert m.shape == (3, 3)
    assert torch.allclose(m, torch.eye(3))


def test_iou_matrix_ignora_o_fundo(iou_matrix):
    """O fundo (0) não é objeto: não pode virar linha nem coluna."""
    gt = three_touching_blocks()
    assert iou_matrix(gt, gt).shape == (3, 3)


def test_iou_matrix_labels_nao_contiguos(iou_matrix):
    """Labels 1, 5, 9 devem funcionar igual a 1, 2, 3."""
    gt = three_touching_blocks()
    esparso = torch.zeros_like(gt)
    for novo, antigo in zip([1, 5, 9], [1, 2, 3]):
        esparso[gt == antigo] = novo
    assert torch.allclose(iou_matrix(esparso, esparso), torch.eye(3))


# ----------------------------------------------------------------------------- matching

@pytest.fixture
def match_greedy():
    return instance.MatchGreedy()


@pytest.fixture
def match_hungarian():
    return instance.MatchHungarian()


def test_guloso_e_hungaro_divergem(match_greedy, match_hungarian):
    """Caso construído para o guloso errar.

        iou = [[0.90, 0.80],      pred A
               [0.85, 0.00]]      pred B
               gt1   gt2

    Guloso: pega (A,gt1)=0.90 primeiro, e aí B só tem 0.00 com gt2 -> 1 par.
    Húngaro: prefere (A,gt2)=0.80 + (B,gt1)=0.85 = 1.65 > 0.90 -> 2 pares.

    O enunciado (Parte 1, item 4) pede a regra explícita justamente porque isso muda
    os números. Este teste é material de apresentação.
    """
    iou = torch.tensor([[0.90, 0.80],
                        [0.85, 0.00]])

    counts_greedy = instance.CountsAtThreshold(match_greedy)
    counts_hungarian = instance.CountsAtThreshold(match_hungarian)

    guloso = sorted(match_greedy(iou, 0.5))
    hungaro = sorted(match_hungarian(iou, 0.5))

    assert guloso == [(0, 0)]
    assert hungaro == [(0, 1), (1, 0)]
    assert counts_greedy(iou, 0.5) == (1, 1, 1)
    assert counts_hungarian(iou, 0.5) == (2, 0, 0)


def test_matching_respeita_o_limiar(match_greedy, match_hungarian):
    """Um par com IoU 0.45 não pode casar nem no limiar mais permissivo (0.50)."""
    iou = torch.tensor([[0.45]])
    assert match_greedy(iou, 0.5) == []
    assert match_hungarian(iou, 0.5) == []


def test_matching_e_um_para_um(match_greedy, match_hungarian):
    """Um objeto previsto não pode casar com dois reais, nem vice-versa."""
    iou = torch.tensor([[0.9, 0.9],
                        [0.9, 0.9]])
    for matcher in (match_greedy, match_hungarian):
        pares = matcher(iou, 0.5)
        assert len(pares) == 2
        assert len({i for i, _ in pares}) == 2
        assert len({j for _, j in pares}) == 2


# ---------------------------------------------------------------------------- AP e mAP

@pytest.fixture
def map_metric():
    return instance.MeanAveragePrecision()


@pytest.fixture
def count_error_metric():
    return instance.CountError()


def test_gabarito_como_predicao_da_map_1(map_metric):
    """O teste que precede todos os outros. Se este falhar, nada mais vale."""
    gt = three_touching_blocks()
    m_ap, por_limiar = map_metric(gt, gt)
    assert m_ap == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in por_limiar.values())


def test_predicao_vazia(map_metric):
    gt = three_touching_blocks()
    vazio = torch.zeros_like(gt)
    m_ap, _ = map_metric(vazio, gt)
    assert m_ap == pytest.approx(0.0)

    counts = instance.CountsAtThreshold()
    assert counts(instance.IoUMatrix()(vazio, gt), 0.5) == (0, 0, 3)


def test_predicao_do_nada(map_metric):
    """Previu 2 objetos onde não havia nada: 2 falsos positivos."""
    gt = torch.zeros((3, 9), dtype=torch.int64)
    pred = three_touching_blocks()[:, :6]
    pred = torch.nn.functional.pad(pred, (0, 3))
    m_ap, _ = map_metric(pred, gt)
    assert m_ap == pytest.approx(0.0)

    counts = instance.CountsAtThreshold()
    assert counts(instance.IoUMatrix()(pred, gt), 0.5) == (0, 2, 0)


def test_ambos_vazios(map_metric):
    """Convenção: acertar que não há nada vale 1.0."""
    vazio = torch.zeros((5, 5), dtype=torch.int64)
    m_ap, _ = map_metric(vazio, vazio)
    assert m_ap == pytest.approx(1.0)


@pytest.mark.parametrize("iou_alvo, map_esperado", [
    (0.95, 1.0),
    (0.88, 0.8),
    (0.72, 0.5),
    (0.62, 0.3),
    (0.45, 0.0),
])
def test_map_e_fracao_de_limiares(iou_alvo, map_esperado, map_metric):
    """mAP NÃO é média de IoU: é a fração das 10 marcas em que o objeto passa."""
    pred, gt = object_with_exact_iou(iou_alvo)
    m_ap, _ = map_metric(pred, gt)
    assert m_ap == pytest.approx(map_esperado, abs=1e-9)


# ------------------------------------------------- O CASO DO TRABALHO (Parte 1 inteira)

def test_tres_nucleos_grudados(iou_metric, dice_metric, map_metric, count_error_metric):
    """IoU semântico perfeito E mAP zero — a tese da apresentação, em miniatura.

    3 objetos reais de 3x3 encostados. O baseline (limiar + componentes conexos) enxerga
    UM blob de 27 px, porque na máscara binária não há nada que os separe.

        IoU por par = 9/27 = 0,33  ->  abaixo até do limiar mais permissivo (0,50)
    """
    gt = three_touching_blocks()
    blob = (gt > 0).long()

    to_binary = semantic.ToBinary()
    assert iou_metric(to_binary(blob), to_binary(gt)) == pytest.approx(1.0)
    assert dice_metric(to_binary(blob), to_binary(gt)) == pytest.approx(1.0)

    iou_mat = instance.IoUMatrix()(blob, gt)
    assert torch.allclose(iou_mat, torch.full((1, 3), 9 / 27))

    m_ap, _ = map_metric(blob, gt)
    assert m_ap == pytest.approx(0.0)

    counts = instance.CountsAtThreshold()
    assert counts(iou_mat, 0.5) == (0, 1, 3)
    assert count_error_metric(blob, gt) == 2


def test_erro_de_contagem(count_error_metric):
    gt = three_touching_blocks()
    assert count_error_metric(gt, gt) == 0
    assert count_error_metric(torch.zeros_like(gt), gt) == 3
