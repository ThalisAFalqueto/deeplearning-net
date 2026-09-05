"""Testes do agrupamento por densidade (Parte 1, item 5)."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import pytest

from src.evaluation.density import group_by_density


def registro(n_gt, map_val, iou=0.99, count_error=0):
    return {"n_gt": n_gt, "n_pred": n_gt, "iou": iou, "dice": iou,
            "map": map_val, "count_error": count_error}


def test_agrupa_em_faixas():
    """O quê: n imagens com densidades variadas viram no máximo n_bins faixas.
    Por quê: um ponto por imagem vira nuvem ilegível; a tendência só aparece agrupada.
    Como: densidades de 1 a 12 em 3 faixas.
    """
    registros = [registro(n, 0.5) for n in range(1, 13)]
    faixas = group_by_density(registros, n_bins=3)
    assert 0 < len(faixas) <= 3


def test_faixas_vazias_sao_descartadas():
    """O quê: faixas sem nenhuma imagem não aparecem no resultado.
    Por quê: uma faixa vazia não tem média — plotá-la produz um buraco no gráfico.
    Como: densidades só nos extremos (1-2 e 19-20), com 6 faixas; o meio fica vazio.
    """
    registros = [registro(n, 0.5) for n in (1, 2, 19, 20)]
    faixas = group_by_density(registros, n_bins=6)
    assert len(faixas) == 2
    for faixa in faixas:
        assert faixa["n_imagens"] > 0


def test_media_por_faixa():
    """O quê: cada faixa reporta a média das imagens que caíram nela.
    Por quê: é o número que vai para o gráfico; se estiver errado, a curva mente.
    Como: duas imagens na mesma faixa com mAP 0,2 e 0,8 -> média 0,5.
    """
    registros = [registro(5, 0.2), registro(5, 0.8)]
    faixas = group_by_density(registros, n_bins=1)
    assert len(faixas) == 1
    assert faixas[0]["map"] == pytest.approx(0.5)
    assert faixas[0]["n_imagens"] == 2


def test_a_tendencia_aparece():
    """O quê: com mAP caindo conforme a densidade sobe, as faixas refletem a queda.
    Por quê: é literalmente o que o enunciado pede ("a tendência tem que ficar visível").
              Se o agrupamento embaralhar as faixas, o gráfico não mostra nada.
    Como: mAP construído como 1/n_gt; a primeira faixa tem que ser maior que a última.
    """
    registros = [registro(n, 1.0 / n) for n in range(1, 21)]
    faixas = group_by_density(registros, n_bins=4)
    assert faixas[0]["map"] > faixas[-1]["map"]


def test_todas_as_imagens_sao_contadas():
    """O quê: nenhuma imagem se perde no agrupamento.
    Por quê: uma imagem fora de todas as faixas (erro clássico no limite superior)
              distorce as médias silenciosamente.
    Como: soma de n_imagens das faixas == total de registros.
    """
    registros = [registro(n, 0.5) for n in range(1, 31)]
    faixas = group_by_density(registros, n_bins=5)
    assert sum(f["n_imagens"] for f in faixas) == len(registros)


def test_uma_imagem_so():
    """O quê: um único registro produz uma faixa com aquele valor.
    Por quê: caso de borda em que mínimo == máximo; largura de faixa zero costuma
              causar divisão por zero.
    Como: um registro, n_bins=6.
    """
    faixas = group_by_density([registro(7, 0.42)], n_bins=6)
    assert len(faixas) == 1
    assert faixas[0]["map"] == pytest.approx(0.42)
    assert faixas[0]["n_imagens"] == 1
