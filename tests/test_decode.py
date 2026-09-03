"""Testes da decodificação por componentes conexos.

Estes testes validam o passo crítico entre a saída do modelo (probabilidade por pixel)
e a entrada da métrica de instância (label map com IDs). A decodificação é limiar +
componentes conexos — e é este método que FALHA quando objetos se tocam, produzindo
mAP baixo mesmo com IoU semântico alto.

Cada teste é documentado com:
  - **O quê**: o que o teste verifica
  - **Por quê**: por que o teste é necessário
  - **Como**: mecânica do teste

Rode com:   pytest tests/test_decode.py -v
"""

import numpy as np
import pytest

from helpers import setup_environment
setup_environment()

from src.utils import labels_from_probability, remove_small_objects


# === labels_from_probability ==================================================

def test_dois_objetos_separados():
    """O quê: duas manchas que não se tocam viram 2 instâncias distintas.
    Por quê: caso feliz — componentes conexos deve enumerar cada mancha isolada.
              Se falhar, ndimage.label não está numerando ou o limiar não foi aplicado.
    Como: duas manchas 3x3 em cantos opostos, threshold 0.5 → 2 labels, cada uma
          com 9 pixels.
    """
    prob = np.zeros((10, 10))
    prob[1:4, 1:4] = 0.9      # mancha A
    prob[6:9, 6:9] = 0.9      # mancha B, longe

    labels = labels_from_probability(prob, threshold=0.5)
    assert len(np.unique(labels)) - 1 == 2
    areas = sorted((labels == v).sum() for v in np.unique(labels) if v != 0)
    assert areas == [9, 9]


def test_objetos_encostados_viram_um_so():
    """O quê: dois quadrados que se tocam viram 1 blob (1 instância).
    Por quê: esta é a LIMITAÇÃO FUNDAMENTAL da decodificação ingêcia. Na máscara binária
              não há informação que separe os objetos. Este teste não é um bug para consertar —
              é a justificativa para a Parte 2 (embeddings discriminativos).
    Como: dois quadrados 4x3 colados horizontalmente (compartilham a borda em x=4),
          threshold 0.5 → 1 componente conexo.
    """
    prob = np.zeros((6, 8))
    prob[1:5, 1:4] = 0.9      # quadrado esquerdo
    prob[1:5, 4:7] = 0.9      # quadrado direito, colado no primeiro

    labels = labels_from_probability(prob, threshold=0.5)
    assert len(np.unique(labels)) - 1 == 1


def test_limiar_muda_o_resultado():
    """O quê: o limiar é hiperparâmetro — muda o número de objetos.
    Por quê: uma ponte de 1 pixel com probabilidade 0.6 faz o contrário: no limiar 0.5
              a ponte passa (1 objeto), no limiar 0.7 a ponte some (2 objetos).
              Isso mostra que a decodificação ingênua é frágil e o limiar precisa ser
              calibrado — ponto que abre a apresentação.
    Como: duas manchas separadas por 1px com prob 0.6. threshold 0.5 → 1 blob; 0.7 → 2.
    """
    prob = np.zeros((5, 9))
    prob[1:4, 1:4] = 0.9
    prob[1:4, 5:8] = 0.9
    prob[2, 4] = 0.6          # ponte de um pixel

    assert len(np.unique(labels_from_probability(prob, 0.5))) - 1 == 1
    assert len(np.unique(labels_from_probability(prob, 0.7))) - 1 == 2


def test_mapa_vazio():
    """O quê: mapa todo-zero decodifica como tudo fundo (0).
    Por quê: edge case — sem nenhum pixel acima do limiar. A saída deve preservar
              shape e ter max()==0 (nenhum objeto).
    Como: prob 5x5 de zeros → labels 5x5, max 0.
    """
    prob = np.zeros((5, 5))
    labels = labels_from_probability(prob, threshold=0.5)
    assert labels.shape == prob.shape
    assert labels.max() == 0


def test_saida_tem_o_formato_que_a_metrica_espera():
    """O quê: a saída é integer, 0=background, 1..N contíguos.
    Por quê: a métrica de instância (IoUMatrix, mAP) consome label maps neste formato.
              Se viesse float ou labels não contíguos, a métrica quebra.
    Como: duas manchas 2x2 → labels [0,1,2], dtype integer.
    """
    prob = np.zeros((10, 10))
    prob[1:3, 1:3] = 0.9
    prob[6:8, 6:8] = 0.9

    labels = labels_from_probability(prob, threshold=0.5)
    assert np.issubdtype(labels.dtype, np.integer)
    assert sorted(np.unique(labels)) == [0, 1, 2]


# === remove_small_objects =====================================================

def test_remove_small_objects():
    """O quê: instâncias com menos de min_area pixels são apagadas.
    Por quê: ruído no mapa de probabilidade vira manchinhas de 1-2px que passam do limiar
              e contam como falsos positivos no mAP. Remover por área é o pós-processamento
              mais barato e o efeito no mAP é um número que vale mostrar na apresentação.
    Como: label com objeto de 16px (sobrevive) e objeto de 1px (removido), min_area=5.
    """
    labels = np.zeros((10, 10), dtype=int)
    labels[1:5, 1:5] = 1      # 16 pixels — sobrevive
    labels[7, 7] = 2          # 1 pixel — é removido

    limpo = remove_small_objects(labels, min_area=5)
    assert sorted(np.unique(limpo)) == [0, 1]
    assert (limpo == 1).sum() == 16


def test_remove_small_objects_renumera():
    """O quê: após remover, labels são renumerados 0,1,2,... sem buracos.
    Por quê: buracos na numeração não quebram o cálculo mas confundem inspeção,
              contagem e visualização. O teste garante que o label 2 vira 1 quando
              o label 1 é removido.
    Como: label 1 (1px, removido) e label 2 (16px, sobrevive) → result 0,1.
    """
    labels = np.zeros((10, 10), dtype=int)
    labels[0, 0] = 1          # 1 pixel — removido
    labels[5:9, 5:9] = 2      # 16 pixels — fica

    limpo = remove_small_objects(labels, min_area=5)
    assert sorted(np.unique(limpo)) == [0, 1]
    assert (limpo == 1).sum() == 16
