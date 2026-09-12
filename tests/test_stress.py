"""Testes da Parte 6 — as corrupções do teste de estresse.

Nenhum treina nem carrega checkpoint: corrupção é função pura sobre a imagem. O teste que
mais importa é `test_neutro_e_identidade`: a severidade 0 da curva de degradação tem que ser
*exatamente* a imagem limpa, senão o ponto de partida da curva já seria uma degradação.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch

from src.stress.corrupcoes import (
    CORRUPCOES,
    NEUTRO,
    borrao,
    brilho_contraste,
    ruido,
)

FAMILIAS = list(CORRUPCOES.values())


def imagem_aleatoria(lado: int = 64, seed: int = 0) -> torch.Tensor:
    gerador = torch.Generator().manual_seed(seed)
    return torch.rand(1, lado, lado, generator=gerador)


# ------------------------------------------------------------------------------- contrato

@pytest.mark.parametrize("familia", FAMILIAS, ids=lambda f: f.nome)
def test_neutro_e_identidade(familia):
    """O quê: com intensidade 0, a corrupção devolve a imagem idêntica bit a bit.
    Por quê: é o ponto de severidade 0 da curva. Se não for a imagem limpa, o mAP desse ponto
              não bate com o da Parte 2 e a curva inteira fica sem referência.
    """
    imagem = imagem_aleatoria()
    assert torch.equal(familia.aplicar(imagem, NEUTRO), imagem)


@pytest.mark.parametrize("familia", FAMILIAS, ids=lambda f: f.nome)
def test_preserva_forma_dtype_e_faixa(familia):
    """O quê: forma, dtype e faixa [0,1] sobrevivem a todas as intensidades.
    Por quê: a rede recebe esse tensor direto; qualquer uma das três coisas fora do lugar
              mediria erro de pipeline, não robustez do modelo.
    """
    imagem = imagem_aleatoria()
    for intensidade in familia.intensidades:
        saida = familia.aplicar(imagem, intensidade)
        assert saida.shape == imagem.shape
        assert saida.dtype == imagem.dtype
        assert float(saida.min()) >= 0.0 and float(saida.max()) <= 1.0


@pytest.mark.parametrize("familia", FAMILIAS, ids=lambda f: f.nome)
def test_nao_altera_a_imagem_de_entrada(familia):
    """O quê: a corrupção devolve um tensor novo e não mexe no original.
    Por quê: o runner reusa a mesma imagem nas 10 condições. Se uma corrupção alterasse o
              tensor no lugar, as condições seguintes receberiam a imagem já suja.
    """
    imagem = imagem_aleatoria()
    copia = imagem.clone()
    familia.aplicar(imagem, familia.intensidades[-1])
    assert torch.equal(imagem, copia)


@pytest.mark.parametrize("familia", FAMILIAS, ids=lambda f: f.nome)
def test_tres_intensidades(familia):
    """O quê: cada família tem exatamente três intensidades, crescentes.
    Por quê: o enunciado pede "em 3 intensidades".
    """
    assert len(familia.intensidades) == 3
    assert list(familia.intensidades) == sorted(familia.intensidades)
    assert all(i > 0 for i in familia.intensidades)


# --------------------------------------------------------------------------------- borrão

def test_borrao_borra_cada_vez_mais():
    """O quê: quanto maior o sigma, menor a variação da imagem.
    Por quê: prova que a função borra de fato, e que a ordem das intensidades é crescente em
              severidade — é o eixo x da curva.
    """
    imagem = imagem_aleatoria()
    desvios = [float(borrao(imagem, s).std()) for s in (0, 1, 2, 4)]
    assert desvios == sorted(desvios, reverse=True)


def test_borrao_nao_escurece_a_borda():
    """O quê: uma imagem constante continua constante depois do borrão.
    Por quê: pega regressão para `mode="constant"`, que criaria uma moldura escura — a rede
              leria essa moldura como borda de objeto e a curva mediria um artefato nosso.
    """
    constante = torch.full((1, 32, 32), 0.7)
    borrada = borrao(constante, 4)
    assert torch.allclose(borrada, constante, atol=1e-6)


# ---------------------------------------------------------------------------------- ruído

def test_ruido_e_reproduzivel_e_varia_por_imagem():
    """O quê: mesma imagem e mesma intensidade dão o mesmo ruído; índices diferentes, não.
    Por quê: sem isso, rodar a Parte 6 duas vezes daria números diferentes. E um ruído igual
              em todas as imagens seria um padrão fixo, não ruído.
    """
    imagem = imagem_aleatoria()
    assert torch.equal(ruido(imagem, 0.05, idx=3), ruido(imagem, 0.05, idx=3))
    assert not torch.equal(ruido(imagem, 0.05, idx=3), ruido(imagem, 0.05, idx=4))
    assert not torch.equal(ruido(imagem, 0.05, idx=3), ruido(imagem, 0.10, idx=3))


def test_ruido_cresce_com_sigma():
    """O quê: sigma maior afasta mais a imagem da original.
    Como: mede o desvio da diferença, e não da imagem, porque o corte em [0,1] comprime a
          medida direta numa imagem já clara.
    """
    imagem = torch.full((1, 64, 64), 0.5)
    afastamentos = [float((ruido(imagem, s) - imagem).std()) for s in (0.02, 0.05, 0.10)]
    assert afastamentos == sorted(afastamentos)


def test_ruido_corta_em_zero_e_um():
    """O quê: numa imagem quase saturada, o ruído é cortado e não estoura a faixa."""
    clara = torch.full((1, 64, 64), 0.98)
    com_ruido = ruido(clara, 0.10)
    assert float(com_ruido.max()) == pytest.approx(1.0)
    assert float(com_ruido.min()) >= 0.0


# ----------------------------------------------------------------------- brilho/contraste

def test_brilho_contraste_segue_a_formula():
    """O quê: `(img - 0,5)·(1-f) + 0,5 + f/2` com f=0,4 leva 0,8 em 0,88.
    Por quê: é a definição escrita no README e na docstring; um número conferido à mão evita
              que a fórmula mude sem ninguém notar.
    """
    saida = brilho_contraste(torch.tensor([[[0.8]]]), 0.4)
    assert float(saida) == pytest.approx(0.88, abs=1e-6)


def test_brilho_contraste_comprime_a_faixa_sem_estourar():
    """O quê: com força f, a transformação leva [0,1] em [f,1] — nunca sai da faixa.
    Por quê: é por isso que o corte dessa função nunca dispara. Documentar por teste evita
              alguém "consertar" um corte que não é usado.
    """
    extremos = torch.tensor([[[0.0, 1.0]]])
    for f in (0.2, 0.4, 0.6):
        saida = brilho_contraste(extremos, f)
        assert float(saida[0, 0, 0]) == pytest.approx(f, abs=1e-6)
        assert float(saida[0, 0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_brilho_contraste_reduz_o_contraste():
    """O quê: a amplitude da imagem encolhe conforme a força cresce."""
    imagem = imagem_aleatoria()
    amplitudes = [float(brilho_contraste(imagem, f).std()) for f in (0.0, 0.2, 0.4, 0.6)]
    assert amplitudes == sorted(amplitudes, reverse=True)
