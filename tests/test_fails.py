"""Testes da Parte 5 — medidas de tamanho, modo de falha e diagnóstico.

O ponto central é `test_fragmentacao_nao_culpa_campo_receptivo`: campo receptivo curto faz o
modelo **fundir** objetos, nunca dividi-los. Se o diagnóstico apontar o campo receptivo numa
imagem que foi fragmentada, ele está explicando a falha errada — foi o que aconteceu na
primeira versão da Parte 5, em 3 das 5 falhas da galeria.

Nada aqui treina nem carrega checkpoint: as cenas são desenhadas à mão.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest

from src.fails.runner import (
    FailGalleryRunner,
    _medidas,
    _modo_de_falha,
    maior_blob,
    medidas_instancias,
)

RF = 68


def cena(lado: int = 256) -> np.ndarray:
    return np.zeros((lado, lado), dtype=np.int64)


def pinta_disco(rotulos: np.ndarray, cy: int, cx: int, raio: int, rotulo: int) -> np.ndarray:
    yy, xx = np.mgrid[:rotulos.shape[0], :rotulos.shape[1]]
    rotulos[(yy - cy) ** 2 + (xx - cx) ** 2 <= raio * raio] = rotulo
    return rotulos


def registro(**campos) -> dict:
    """Registro de uma imagem com os campos que o diagnóstico lê."""
    base = {
        "n_gt": 10, "n_pred": 10, "map": 0.05, "modalidade": "histology",
        "modo_falha": "contagem_proxima",
        "diametro_max_objeto": 20.0, "extensao_max_objeto": 25.0,
        "diametro_maior_blob": 20.0, "extensao_maior_blob": 25.0,
        "instancias_no_maior_blob": 1, "pred_no_maior_blob": 1,
    }
    base.update(campos)
    return base


def diagnostica(r: dict, mediana_n_gt: float = 40.0) -> str:
    return FailGalleryRunner(None, "sem_checkpoint")._diagnostico(r, RF, mediana_n_gt)


# ------------------------------------------------------------------------------- medidas

def test_extensao_e_diametro_equivalente_divergem_em_objeto_alongado():
    """O quê: num objeto alongado a extensão passa do campo receptivo e o diâmetro
    equivalente não.
    Por quê: é a diferença que decide a conclusão da parte obrigatória. Medir só o diâmetro
              equivalente esconde os núcleos que o campo receptivo de fato não cobre.
    Como: retângulo de 80×6 px — extensão 80, área 480, diâmetro equivalente ~24,7.
    """
    mascara = np.zeros((256, 256), dtype=bool)
    mascara[100:180, 100:106] = True

    equivalente, extensao = _medidas(mascara)

    assert extensao == 80
    assert equivalente == pytest.approx(2 * np.sqrt(480 / np.pi), rel=1e-6)
    assert extensao > RF > equivalente


def test_medidas_de_um_disco_coincidem():
    """O quê: num disco as duas medidas batem (a menos da discretização).
    Por quê: mostra que a divergência do teste anterior vem da forma, não da fórmula.
    """
    rotulos = pinta_disco(cena(), 128, 128, 20, 1)
    equivalente, extensao = medidas_instancias(rotulos)

    assert extensao[0] == pytest.approx(41, abs=1)
    assert equivalente[0] == pytest.approx(extensao[0], rel=0.05)


def test_medidas_instancias_uma_por_objeto():
    """O quê: uma medida por instância, ignorando o fundo."""
    rotulos = pinta_disco(pinta_disco(cena(), 60, 60, 10, 1), 180, 180, 15, 2)
    equivalente, extensao = medidas_instancias(rotulos)

    assert len(equivalente) == len(extensao) == 2
    assert extensao[1] > extensao[0]


def test_maior_blob_junta_nucleos_encostados():
    """O quê: dois núcleos encostados são 2 instâncias num único blob, e o blob é maior que
    qualquer um deles.
    Por quê: é o caso em que nenhum núcleo isolado passa do campo receptivo mas o aglomerado
              passa — a hipótese que o diagnóstico precisa poder testar.
    """
    rotulos = pinta_disco(pinta_disco(cena(), 128, 120, 18, 1), 128, 150, 18, 2)

    blob = maior_blob(rotulos)

    assert blob["instancias"] == 2
    assert blob["extensao"] > 2 * 18
    assert blob["mascara"].sum() == int((rotulos > 0).sum())


def test_maior_blob_em_imagem_vazia():
    """O quê: gabarito sem objeto nenhum não quebra (imagens de fundo existem no DSB2018)."""
    blob = maior_blob(cena())

    assert blob["instancias"] == 0
    assert blob["diametro"] == 0.0 and blob["extensao"] == 0.0
    assert not blob["mascara"].any()


# ------------------------------------------------------------------------- modo de falha

@pytest.mark.parametrize("n_gt, n_pred, esperado", [
    (9, 39, "fragmentou"),      # a falha 3 real da galeria
    (375, 115, "fundiu"),       # a falha 1 real da galeria
    (20, 21, "contagem_proxima"),
    (0, 5, "contagem_proxima"),  # gabarito vazio: não dá para falar em razão
])
def test_modo_de_falha(n_gt, n_pred, esperado):
    """O quê: a classificação pela contagem separa fragmentar de fundir.
    Por quê: é ela que decide se o campo receptivo pode ser a causa.
    """
    assert _modo_de_falha(n_gt, n_pred) == esperado


# --------------------------------------------------------------------------- diagnóstico

def test_fragmentacao_nao_culpa_campo_receptivo():
    """O quê: numa imagem fragmentada, o diagnóstico nega o campo receptivo — mesmo com o
    blob maior que o RF.
    Por quê: campo receptivo curto funde objetos, nunca os divide. Era o erro da primeira
              versão: o blob de 74 px "explicava" uma imagem em que 2 núcleos viraram 13.
    """
    texto = diagnostica(registro(
        modo_falha="fragmentou", n_gt=9, n_pred=39,
        extensao_maior_blob=74.0, instancias_no_maior_blob=2, pred_no_maior_blob=13))

    assert "Não é campo receptivo" in texto
    assert "9 núcleos viraram 39 rótulos" in texto
    assert "13 rótulo(s)" in texto
    assert "heatmap" in texto


def test_objeto_maior_que_o_rf_e_diagnosticado_pela_extensao():
    """O quê: quando o modelo funde e há objeto mais longo que o RF, o diagnóstico é o do
    enunciado — o pixel central não enxerga as duas bordas.
    Como: extensão 90 px > RF 68, com diâmetro equivalente 30 px (que sozinho não acusaria).
    """
    texto = diagnostica(registro(
        modo_falha="fundiu", n_gt=10, n_pred=4,
        extensao_max_objeto=90.0, diametro_max_objeto=30.0, extensao_maior_blob=90.0))

    assert "90 px de extensão" in texto and "68 px" in texto
    assert "nunca enxerga as duas bordas" in texto


def test_blob_maior_que_o_rf_quando_nenhum_nucleo_isolado_passa():
    """O quê: nenhum núcleo isolado passa do RF, mas o aglomerado passa — e o modelo fundiu.
    Por quê: é a hipótese legítima do campo receptivo para instâncias encostadas.
    """
    texto = diagnostica(registro(
        modo_falha="fundiu", n_gt=12, n_pred=6,
        extensao_max_objeto=40.0, extensao_maior_blob=95.0,
        instancias_no_maior_blob=5, pred_no_maior_blob=1))

    assert "Nenhum núcleo isolado passa do RF" in texto
    assert "95 px de extensão" in texto
    assert "offset não tem como escolher" in texto


def test_densidade_so_e_citada_quando_ha_densidade():
    """O quê: "densidade" só aparece quando a imagem tem muito mais núcleos que a mediana.
    Por quê: a primeira versão escrevia "13 núcleos numa imagem só (densidade extrema)".
    """
    denso = diagnostica(registro(modo_falha="fundiu", n_gt=375, n_pred=115,
                                 extensao_max_objeto=12.0, extensao_maior_blob=14.0),
                        mediana_n_gt=40.0)
    esparso = diagnostica(registro(modo_falha="fundiu", n_gt=13, n_pred=5,
                                   extensao_max_objeto=12.0, extensao_maior_blob=14.0),
                          mediana_n_gt=40.0)

    assert "densidade" in denso and "375 núcleos" in denso
    assert "densidade" not in esparso
    assert "não é de campo receptivo" in esparso


def test_contagem_proxima_aponta_as_bordas():
    """O quê: contagem certa com mAP baixo é erro de borda, não de separação."""
    texto = diagnostica(registro(modo_falha="contagem_proxima", n_gt=20, n_pred=21))
    assert "bordas" in texto and "não é a causa" in texto


# ------------------------------------------------------------------------------- seleção

def test_indices_forcados_saem_na_ordem_pedida():
    """O quê: --fails-indices devolve exatamente os índices pedidos, na ordem.
    Por quê: é o que permite montar o "depois" da correção nas MESMAS imagens do "antes".
    """
    registros = [{"idx": i, "map": i / 10, "count_error": i} for i in range(10)]
    runner = FailGalleryRunner(None, "sem_checkpoint", indices=[7, 2, 5])

    assert [r["idx"] for r in runner._selecionar(registros)] == [7, 2, 5]


def test_sem_indices_pega_os_piores_por_map():
    """O quê: sem --fails-indices, seleciona as n imagens de pior mAP."""
    registros = [{"idx": i, "map": 1.0 - i / 10, "count_error": 0} for i in range(10)]
    runner = FailGalleryRunner(None, "sem_checkpoint", n=3)

    assert [r["idx"] for r in runner._selecionar(registros)] == [9, 8, 7]
