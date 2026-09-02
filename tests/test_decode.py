"""Testes da decodificação por componentes conexos."""

import numpy as np

from helpers import setup_environment
setup_environment()

from src.utils import labels_from_probability, remove_small_objects

import pytest


def test_dois_objetos_separados():
    """Duas manchas que não se tocam viram duas instâncias."""
    prob = np.zeros((10, 10))
    prob[1:4, 1:4] = 0.9      # mancha A
    prob[6:9, 6:9] = 0.9      # mancha B, longe

    labels = labels_from_probability(prob, threshold=0.5)
    assert len(np.unique(labels)) - 1 == 2
    # os dois grupos têm 9 pixels cada
    areas = sorted((labels == v).sum() for v in np.unique(labels) if v != 0)
    assert areas == [9, 9]


def test_objetos_encostados_viram_um_so():
    """O fracasso do método, em miniatura: dois quadrados que se tocam viram 1 blob.

    Não é um bug da implementação — é a limitação da representação. A máscara binária
    não tem nenhuma informação que permita separar os dois.
    """
    prob = np.zeros((6, 8))
    prob[1:5, 1:4] = 0.9      # quadrado esquerdo
    prob[1:5, 4:7] = 0.9      # quadrado direito, colado no primeiro

    labels = labels_from_probability(prob, threshold=0.5)
    assert len(np.unique(labels)) - 1 == 1


def test_limiar_muda_o_resultado():
    """Uma ponte fraca entre duas manchas: o limiar decide se elas são 1 ou 2 objetos."""
    prob = np.zeros((5, 9))
    prob[1:4, 1:4] = 0.9
    prob[1:4, 5:8] = 0.9
    prob[2, 4] = 0.6          # ponte de um pixel, com probabilidade intermediária

    assert len(np.unique(labels_from_probability(prob, 0.5))) - 1 == 1   # ponte passa
    assert len(np.unique(labels_from_probability(prob, 0.7))) - 1 == 2   # ponte some


def test_mapa_vazio():
    prob = np.zeros((5, 5))
    labels = labels_from_probability(prob, threshold=0.5)
    assert labels.shape == prob.shape
    assert labels.max() == 0


def test_saida_tem_o_formato_que_a_metrica_espera():
    """0 = fundo, 1..N contíguos, inteiros."""
    prob = np.zeros((10, 10))
    prob[1:3, 1:3] = 0.9
    prob[6:8, 6:8] = 0.9

    labels = labels_from_probability(prob, threshold=0.5)
    assert np.issubdtype(labels.dtype, np.integer)
    assert sorted(np.unique(labels)) == [0, 1, 2]


def test_remove_small_objects():
    labels = np.zeros((10, 10), dtype=int)
    labels[1:5, 1:5] = 1      # 16 pixels — sobrevive
    labels[7, 7] = 2          # 1 pixel — é removido

    limpo = remove_small_objects(labels, min_area=5)
    assert sorted(np.unique(limpo)) == [0, 1]
    assert (limpo == 1).sum() == 16


def test_remove_small_objects_renumera():
    """Se o objeto 1 for removido, o 2 tem que virar 1 — sem buracos na numeração."""
    labels = np.zeros((10, 10), dtype=int)
    labels[0, 0] = 1          # 1 pixel — removido
    labels[5:9, 5:9] = 2      # 16 pixels — fica

    limpo = remove_small_objects(labels, min_area=5)
    assert sorted(np.unique(limpo)) == [0, 1]
    assert (limpo == 1).sum() == 16
