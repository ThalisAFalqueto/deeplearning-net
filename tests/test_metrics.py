"""Testes da métrica — rodam sem treinar nada.

Estes testes existem porque a métrica é a peça mais perigosa do trabalho: se ela estiver
errada, TODO número da apresentação está errado e nada denuncia. Um matching com bug
produz números plausíveis.

Rode com:   pytest tests/ -v
"""

import numpy as np
import pytest

from src.metrics import instance, semantic


# --------------------------------------------------------------------------- helpers

def three_touching_blocks(size: int = 3):
    """3 objetos `size`x`size` encostados em fila — o caso central do trabalho.

        111222333
        111222333
        111222333
    """
    gt = np.zeros((size, size * 3), dtype=int)
    for k in range(3):
        gt[:, k * size:(k + 1) * size] = k + 1
    return gt


def object_with_exact_iou(target_iou: float, canvas=(20, 20)):
    """Um objeto previsto contido no real, com IoU exatamente `target_iou`.

    O gabarito é um bloco 10x10 (100 px). A predição são os primeiros k px dele,
    com k = target_iou * 100. Como pred ⊂ gt: IoU = k/100.
    """
    gt = np.zeros(canvas, dtype=int)
    gt[0:10, 0:10] = 1

    k = int(round(target_iou * 100))
    pred = np.zeros(canvas, dtype=int)
    for r, c in np.argwhere(gt == 1)[:k]:
        pred[r, c] = 1
    return pred, gt


# ------------------------------------------------------------------ IoU e Dice (pixels)

def test_iou_dice_identicos():
    gt = three_touching_blocks()
    mask = semantic.to_binary(gt)
    assert semantic.iou(mask, mask) == pytest.approx(1.0)
    assert semantic.dice(mask, mask) == pytest.approx(1.0)


def test_iou_dice_grade_6x6():
    """Dois quadrados 3x3 deslocados de (1,1): TP=4, FP=5, FN=5."""
    gt = np.zeros((6, 6), dtype=bool)
    gt[1:4, 1:4] = True          # quadrado 3x3
    pred = np.zeros((6, 6), dtype=bool)
    pred[2:5, 2:5] = True        # mesmo quadrado, deslocado (1,1)

    assert semantic.confusion(pred, gt) == (4, 5, 5)
    assert semantic.iou(pred, gt) == pytest.approx(4 / 14)
    assert semantic.dice(pred, gt) == pytest.approx(8 / 18)


# ------------------------------------------------------------------------ matriz de IoU

def test_iou_matrix_identidade():
    """Gabarito contra ele mesmo: diagonal 1, resto 0."""
    gt = three_touching_blocks()
    m = instance.iou_matrix(gt, gt)
    assert m.shape == (3, 3)
    np.testing.assert_allclose(m, np.eye(3))


def test_iou_matrix_ignora_o_fundo():
    """O fundo (0) não é objeto: não pode virar linha nem coluna."""
    gt = three_touching_blocks()
    assert instance.iou_matrix(gt, gt).shape == (3, 3)


def test_iou_matrix_labels_nao_contiguos():
    """Labels 1, 5, 9 devem funcionar igual a 1, 2, 3."""
    gt = three_touching_blocks()
    esparso = np.zeros_like(gt)
    for novo, antigo in zip([1, 5, 9], [1, 2, 3]):
        esparso[gt == antigo] = novo
    np.testing.assert_allclose(instance.iou_matrix(esparso, esparso), np.eye(3))


# ----------------------------------------------------------------------------- matching

def test_guloso_e_hungaro_divergem():
    """Caso construído para o guloso errar.

        iou = [[0.90, 0.80],      pred A
               [0.85, 0.00]]      pred B
               gt1   gt2

    Guloso: pega (A,gt1)=0.90 primeiro, e aí B só tem 0.00 com gt2 -> 1 par.
    Húngaro: prefere (A,gt2)=0.80 + (B,gt1)=0.85 = 1.65 > 0.90 -> 2 pares.

    O enunciado (Parte 1, item 4) pede a regra explícita justamente porque isso muda
    os números. Este teste é material de apresentação.
    """
    iou = np.array([[0.90, 0.80],
                    [0.85, 0.00]])

    guloso = sorted(instance.match_greedy(iou, 0.5))
    hungaro = sorted(instance.match_hungarian(iou, 0.5))

    assert guloso == [(0, 0)]
    assert hungaro == [(0, 1), (1, 0)]
    assert instance.counts_at_threshold(iou, 0.5, instance.match_greedy) == (1, 1, 1)
    assert instance.counts_at_threshold(iou, 0.5, instance.match_hungarian) == (2, 0, 0)


def test_matching_respeita_o_limiar():
    """Um par com IoU 0.45 não pode casar nem no limiar mais permissivo (0.50)."""
    iou = np.array([[0.45]])
    assert instance.match_greedy(iou, 0.5) == []
    assert instance.match_hungarian(iou, 0.5) == []


def test_matching_e_um_para_um():
    """Um objeto previsto não pode casar com dois reais, nem vice-versa."""
    iou = np.array([[0.9, 0.9],
                    [0.9, 0.9]])
    for matcher in (instance.match_greedy, instance.match_hungarian):
        pares = matcher(iou, 0.5)
        assert len(pares) == 2
        assert len({i for i, _ in pares}) == 2
        assert len({j for _, j in pares}) == 2


# ---------------------------------------------------------------------------- AP e mAP

def test_gabarito_como_predicao_da_map_1():
    """O teste que precede todos os outros. Se este falhar, nada mais vale."""
    gt = three_touching_blocks()
    m_ap, por_limiar = instance.mean_average_precision(gt, gt)
    assert m_ap == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in por_limiar.values())


def test_predicao_vazia():
    gt = three_touching_blocks()
    vazio = np.zeros_like(gt)
    m_ap, _ = instance.mean_average_precision(vazio, gt)
    assert m_ap == pytest.approx(0.0)
    assert instance.counts_at_threshold(instance.iou_matrix(vazio, gt), 0.5) == (0, 0, 3)


def test_predicao_do_nada():
    """Previu 2 objetos onde não havia nada: 2 falsos positivos."""
    gt = np.zeros((3, 9), dtype=int)
    pred = three_touching_blocks()[:, :6]      # 2 objetos
    pred = np.pad(pred, ((0, 0), (0, 3)))
    m_ap, _ = instance.mean_average_precision(pred, gt)
    assert m_ap == pytest.approx(0.0)
    assert instance.counts_at_threshold(instance.iou_matrix(pred, gt), 0.5) == (0, 2, 0)


def test_ambos_vazios():
    """Convenção: acertar que não há nada vale 1.0."""
    vazio = np.zeros((5, 5), dtype=int)
    m_ap, _ = instance.mean_average_precision(vazio, vazio)
    assert m_ap == pytest.approx(1.0)


@pytest.mark.parametrize("iou_alvo, map_esperado", [
    (0.95, 1.0),    # passa nas 10 marcas
    (0.88, 0.8),    # passa em 8   (0.88 < 0.90, então falha em 0.90 e 0.95)
    (0.72, 0.5),    # passa em 5
    (0.62, 0.3),    # passa em 3
    (0.45, 0.0),    # reprova em todas
])
def test_map_e_fracao_de_limiares(iou_alvo, map_esperado):
    """mAP NÃO é média de IoU: é a fração das 10 marcas em que o objeto passa."""
    pred, gt = object_with_exact_iou(iou_alvo)
    m_ap, _ = instance.mean_average_precision(pred, gt)
    assert m_ap == pytest.approx(map_esperado, abs=1e-9)


# ------------------------------------------------- O CASO DO TRABALHO (Parte 1 inteira)

def test_tres_nucleos_grudados():
    """IoU semântico perfeito E mAP zero — a tese da apresentação, em miniatura.

    3 objetos reais de 3x3 encostados. O baseline (limiar + componentes conexos) enxerga
    UM blob de 27 px, porque na máscara binária não há nada que os separe.

        IoU por par = 9/27 = 0,33  ->  abaixo até do limiar mais permissivo (0,50)
    """
    gt = three_touching_blocks()
    blob = (gt > 0).astype(int)          # tudo vira o objeto 1

    # A segmentação semântica está PERFEITA
    assert semantic.iou(semantic.to_binary(blob), semantic.to_binary(gt)) == pytest.approx(1.0)
    assert semantic.dice(semantic.to_binary(blob), semantic.to_binary(gt)) == pytest.approx(1.0)

    # E a de instâncias está zerada
    m = instance.iou_matrix(blob, gt)
    np.testing.assert_allclose(m, np.full((1, 3), 9 / 27))

    m_ap, _ = instance.mean_average_precision(blob, gt)
    assert m_ap == pytest.approx(0.0)
    assert instance.counts_at_threshold(m, 0.5) == (0, 1, 3)
    assert instance.count_error(blob, gt) == 2


def test_erro_de_contagem():
    gt = three_touching_blocks()
    assert instance.count_error(gt, gt) == 0
    assert instance.count_error(np.zeros_like(gt), gt) == 3
