"""Testes da métrica — rodam sem treinar nada.

Estes testes existem porque a métrica é a peça mais perigosa do trabalho: se ela estiver
errada, TODO número da apresentação está errado e nada denuncia. Um matching com bug
produz números plausíveis.

Cada teste é documentado com:
  - **O quê**: o que o teste verifica
  - **Por quê**: por que o teste é necessário
  - **Como**: mecânica do teste

Rode com:   pytest tests/test_metrics.py -v
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
from src.utils import to_binary as _to_binary_fn


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
    return _to_binary_fn


@pytest.fixture
def iou_metric():
    return semantic.IoU()


@pytest.fixture
def dice_metric():
    return semantic.Dice()


@pytest.fixture
def confusion():
    return semantic.Confusion()


# === to_binary ================================================================

def test_to_binary_separa_fundo_de_objeto(to_binary):
    """O quê: separa fundo (label 0) de objetos (labels != 0).
    Por quê: a métrica semântica opera em máscara binária, não em label map.
    Como: passa um label map 2x2 com fundo em (0,0) e (1,1), verifica saída bool.
    """
    labels = torch.tensor([[0, 1],
                           [2, 0]])
    esperado = torch.tensor([[False, True],
                             [True, False]])
    assert torch.equal(_to_binary_fn(labels), esperado)


def test_to_binary_tudo_fundo(to_binary):
    """O quê: quando tudo é fundo, a máscara binária é toda False.
    Por quê: edge case — sem nenhum objeto, .any() deve ser False. Evita divisão por zero
              em IoU/Dice quando denominador == 0.
    Como: passa um tensor de zeros, verifica .any() == False.
    """
    vazio = torch.zeros((3, 3), dtype=torch.int64)
    assert not _to_binary_fn(vazio).any()


def test_to_binary_labels_nao_contiguos(to_binary):
    """O quê: labels não contíguos (1, 5, 9) todos mapear para True.
    Por quê: a rede pode produzir labels esparsos; só o 0 é fundo.
    Como: passa labels [1, 5, 9, 0], verifica que só posição [1,1] (label 0) é False.
    """
    labels = torch.tensor([[1, 5],
                           [9, 0]])
    esperado = torch.tensor([[True, True],
                             [True, False]])
    assert torch.equal(_to_binary_fn(labels), esperado)


def test_to_binary_preserva_forma_e_devolve_bool(to_binary):
    """O quê: shape preservado e dtype é bool.
    Por quê: downstream espera bool para operações & e ~ bit a bit.
    Como: passa label map 2x3, verifica shape e dtype == torch.bool.
    """
    labels = torch.tensor([[0, 1, 2],
                           [3, 0, 4]])
    r = _to_binary_fn(labels)
    assert r.shape == labels.shape
    assert r.dtype == torch.bool


# === Confusion, IoU, Dice =======================================================

def test_iou_dice_identicos(iou_metric, dice_metric):
    """O quê: IoU e Dice de um máscara contra ela mesma == 1.0.
    Por quê: sanity check — se não for 1.0, a fórmula está bugada.
    Como: passa três blocos colados contra si mesmo.
    """
    gt = three_touching_blocks()
    mask = _to_binary_fn(gt)
    assert iou_metric(mask, mask) == pytest.approx(1.0)
    assert dice_metric(mask, mask) == pytest.approx(1.0)


def test_iou_dice_grade_6x6(confusion, iou_metric, dice_metric):
    """O quê: conta TP/FP/FN e calcula IoU/Dice em um caso manuscrito.
    Por quê: valida a fórmula contra valores conhecidos. Dois quadrados 3x3
              deslocados de (1,1): overlap de 4px, TP=4, FP=5, FN=5.
    Como: grade 6x6, blocos 3x3 em (1,1) e (2,2). IoU = 4/14, Dice = 8/18.
    """
    gt = torch.zeros((6, 6), dtype=torch.bool)
    gt[1:4, 1:4] = True
    pred = torch.zeros((6, 6), dtype=torch.bool)
    pred[2:5, 2:5] = True

    assert confusion(pred, gt) == (4, 5, 5)
    assert iou_metric(pred, gt) == pytest.approx(4 / 14)
    assert dice_metric(pred, gt) == pytest.approx(8 / 18)


# ------------------------------------------------------------------ matriz de IoU

@pytest.fixture
def iou_matrix():
    return instance.IoUMatrix()


def test_iou_matrix_identidade(iou_matrix):
    """O quê: IoU de um label map contra ele mesmo = matriz identidade.
    Por quê: diagonal deve ser 1 (mesmo objeto), fora-diagonal 0 (nenhum overlap).
              Se falhar, o indexamento de labels está errado.
    Como: três blocos colados, IoU(gt, gt) == I₃.
    """
    gt = three_touching_blocks()
    m = iou_matrix(gt, gt)
    assert m.shape == (3, 3)
    assert torch.allclose(m, torch.eye(3))


def test_iou_matrix_ignora_o_fundo(iou_matrix):
    """O quê: o fundo (label 0) não aparece como linha/coluna na matriz.
    Por quê: o fundo não é um objeto — não deve gerar pares de IoU.
    Como: três blocos colados → matriz 3x3, não 4x4.
    """
    gt = three_touching_blocks()
    assert iou_matrix(gt, gt).shape == (3, 3)


def test_iou_matrix_labels_nao_contiguos(iou_matrix):
    """O quê: labels esparsos (1,5,9) produzem a mesma matriz que (1,2,3).
    Por quê: a métrica itera sobre np.unique, não sobre índices de array.
              Um bug aqui faria a métrica pegar o label 0 ou pular labels.
    Como: relabeleia 1→1, 2→5, 3→9, verifica matriz identidade idêntica.
    """
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
    """O quê: matching guloso e húngaro dão resultados diferentes no mesmo caso.
    Por quê: o enunciado (Parte 1, item 4) pede a regra explícita porque muda os números.
              Este teste é material de apresentação — mostra que a escolha do matcher afeta
              o mAP. Também valida que ambos respeitam o limiar de IoU.
    Como: matriz 2x2 onde guloso pega (A,gt1)=0.90 e perde (B,gt2) por threshold;
          húngaro troca e pega ambos: (A,gt2)=0.80 + (B,gt1)=0.85.
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
    """O quê: pares com IoU abaixo do limiar são rejeitados.
    Por quê: o mAP usa 10 limiares (0.50–0.95). Um par com IoU 0.45 não passa
              nem no limiar mais permissivo (0.50). Se o matcher ignorar o limiar,
              o mAP fica inflado.
    Como: matriz 1x1 com IoU=0.45, threshold=0.5 → nenhum par.
    """
    iou = torch.tensor([[0.45]])
    assert match_greedy(iou, 0.5) == []
    assert match_hungarian(iou, 0.5) == []


def test_matching_e_um_para_um(match_greedy, match_hungarian):
    """O quê: matching é bijetivo — um objeto não casa com dois.
    Por quê: em segmentação de instâncias, cada GT casa com no máximo um pred.
              Se um pred casar com dois GTs, um deles vira falso negativo.
    Como: matriz 2x2 com todos = 0.9, threshold 0.5 → exatamente 2 pares,
          2 linhas distintas e 2 colunas distintas.
    """
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
    """O quê: predição == gabarito → mAP = 1.0 em todos os limiares.
    Por quê: este é o teste de partida obrigatório. Se falhar, NADA nos testes posteriores
              é confiável — o bug está na métrica, não no modelo. Valida todo o pipeline:
              iou_matrix → matching → counts → AP → média.
    Como: passa three_touching_blocks como predição e como gabarito.
    """
    gt = three_touching_blocks()
    m_ap, por_limiar = map_metric(gt, gt)
    assert m_ap == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in por_limiar.values())


def test_predicao_vazia(map_metric):
    """O quê: prever zero objetos quando existem 3 → mAP = 0.0, FP=0, FN=3.
    Por quê: edge case — não prever nada quando deveria. Todos os 3 GTs são FN,
              e não há FP para onde bater. mAP deve ser 0 (nenhum objeto casou).
    Como: pred = matriz de zeros, gt = três blocos.
    """
    gt = three_touching_blocks()
    vazio = torch.zeros_like(gt)
    m_ap, _ = map_metric(vazio, gt)
    assert m_ap == pytest.approx(0.0)

    counts = instance.CountsAtThreshold()
    assert counts(instance.IoUMatrix()(vazio, gt), 0.5) == (0, 0, 3)


def test_predicao_do_nada(map_metric):
    """O quê: prever 2 objetos onde não há nada → mAP=0, TP=0, FP=2, FN=0.
    Por quê: valida falso positivo. Predêe algo que não existe → mAP deve ser 0,
              e FN=0 porque não perdeu nenhum GT.
    Como: gt = tudo zeros, pred = dois blocos de 3x3.
    """
    gt = torch.zeros((3, 9), dtype=torch.int64)
    pred = three_touching_blocks()[:, :6]
    pred = torch.nn.functional.pad(pred, (0, 3))
    m_ap, _ = map_metric(pred, gt)
    assert m_ap == pytest.approx(0.0)

    counts = instance.CountsAtThreshold()
    assert counts(instance.IoUMatrix()(pred, gt), 0.5) == (0, 2, 0)


def test_ambos_vazios(map_metric):
    """O quê: gt vazio e pred vazio → mAP = 1.0 (convenção DSB2018).
    Por quê: "acertar que não há nada" é perfeito. Se devolvesse 0, o mAP de imagens
              sem núcleos ficaria 0, penalizando o modelo injustamente.
    Como: ambos zeros 5x5 → mAP = 1.0.
    """
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
    """O quê: mAP é a fração das 10 marcas (0.50–0.95) que o objeto passa, NÃO a IoU média.
    Por quê: é o conceito mais central e mais errado da disciplina (o Thalis errou duas vezes).
              IoU=0.88 passa em 8 de 10 marcas → mAP=0.8, não 0.88.
    Como: constrói pred ⊂ gt com IoU exata, verifica mAP contra a fração esperada.
    """
    pred, gt = object_with_exact_iou(iou_alvo)
    m_ap, _ = map_metric(pred, gt)
    assert m_ap == pytest.approx(map_esperado, abs=1e-9)


# ------------------------------------------------- O CASO DO TRABALHO (Parte 1 inteira)

def test_tres_nucleos_grudados(iou_metric, dice_metric, map_metric, count_error_metric):
    """O quê: 3 objetos reais encostados; o baseline (limiar + componentes conexos) enxerga
               UM blob → IoU semântico perfeito, mAP de instância zero.
    Por quê: este é o contraste central da apresentação. Um modelo que acerta quase todos
              os pixels (IoU=1.0) mas junta 3 objetos em 1 (mAP=0). Mostra POR QUE precisamos
              de segmentação de instância, não só semântica.
    Como: três blocos 3x3 colados → binário perfeito (IoU=1.0, Dice=1.0). Decodificação
          via connected components vira 1 blob. IoU par = 9/27 = 0.33 < 0.50 em todos os limiares.
    """
    gt = three_touching_blocks()
    blob = (gt > 0).long()

    assert iou_metric(_to_binary_fn(blob), _to_binary_fn(gt)) == pytest.approx(1.0)
    assert dice_metric(_to_binary_fn(blob), _to_binary_fn(gt)) == pytest.approx(1.0)

    iou_mat = instance.IoUMatrix()(blob, gt)
    assert torch.allclose(iou_mat, torch.full((1, 3), 9 / 27))

    m_ap, _ = map_metric(blob, gt)
    assert m_ap == pytest.approx(0.0)

    counts = instance.CountsAtThreshold()
    assert counts(iou_mat, 0.5) == (0, 1, 3)
    assert count_error_metric(blob, gt) == 2


def test_erro_de_contagem(count_error_metric):
    """O quê: erro absoluto de contagem entre predição e gabarito.
    Por quê: mAP pode ser 0 tanto por errar objetos quanto por errar a contagem.
              O erro de contagem é um sinal separado — o enunciado pede que ele apareça
              no eval ("erro de contagem ~N objetos/imagem").
    Como: gt vs gt → erro 0; gt vs zeros → erro = 3 (perdeu todos).
    """
    gt = three_touching_blocks()
    assert count_error_metric(gt, gt) == 0
    assert count_error_metric(torch.zeros_like(gt), gt) == 3
