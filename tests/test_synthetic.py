"""Testes do gerador sintético de elipses.

Validam o dataset que alimenta toda a Parte 0: as propriedades dos dados
(formato, tipos, range), a reprodutibilidade, a taxa de fusão (p_touch) e a
integração com o pipeline de decodificação.

Cada teste é documentado com:
  - **O quê**: o que o teste verifica
  - **Por quê**: por que o teste é necessário
  - **Como**: mecânica do teste

Rode com:   pytest tests/test_synthetic.py -v
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _TESTS_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_TESTS_DIR))

from helpers.setup_environment import setup_environment
setup_environment()

from src.data.synthetic import SyntheticEllipses
from src.utils import labels_from_probability

from scipy import ndimage


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def ds():
    """Dataset pequeno (10 imagens) para testes rápidos."""
    return SyntheticEllipses(
        n_samples=10, size=128, seed=42, min_obj=5, max_obj=20,
    )


# --------------------------------------------------------------------------- formato

def test_len(ds):
    """O quê: __len__ retorna o número de amostras pedido.
    Por quê: o DataLoader usa len() para saber quantas iterações por época.
              Se for errado, o loop de treino itera o número errado de vezes.
    Como: cria dataset com n_samples=10, verifica len == 10.
    """
    assert len(ds) == 10


def test_shape(ds):
    """O quê: image tem shape (1, 128, 128), label tem shape (128, 128).
    Por quê: o U-Net espera entrada (B, 1, H, W) e o BCEWithLogitsLoss compara
              com target (B, 1, H, W). A label é índice 2D por instância.
    Como: pega amostra 0, verifica shapes.
    """
    img, lbl = ds[0]
    assert img.shape == (1, 128, 128)
    assert lbl.shape == (128, 128)


def test_types(ds):
    """O quê: image é float32, label é int64.
    Por quê: float32 para imagens de entrada do modelo; int64 para labels de
              instância (índices usados como máscara booleana `labels == k`).
    Como: pega amostra 0, verifica dtypes.
    """
    img, lbl = ds[0]
    assert img.dtype == torch.float32
    assert lbl.dtype == torch.int64


def test_valores_no_range(ds):
    """O quê: todos os pixels da imagem estão no intervalo [0, 1].
    Por quê: a imagem passa pela sigmoid da rede e depois thresholding.
              Valores saturados (>1 ou <0) fazem a sigmoid travar (gradiente zero)
              e prejudicam o treinamento.
    Como: verifica img.min() >= 0 e img.max() <= 1.
    """
    img, _ = ds[0]
    assert img.min() >= 0.0
    assert img.max() <= 1.0


# --------------------------------------------------------------------------- labels

def test_labels_contiguos(ds):
    """O quê: labels são 0, 1, 2, ..., N sem buracos.
    Por quê: ndimage.label e iou_matrix esperam labels contíguos.
              Buracos quebram visualização e contagem de objetos.
    Como: para cada imagem, verifica sorted(unique(lbl)) == arange(N+1).
    """
    for i in range(len(ds)):
        _, lbl = ds[i]
        unq = torch.unique(lbl)
        assert unq[0] == 0
        assert torch.equal(unq, torch.arange(len(unq)))


def test_numero_de_objetos_no_range(ds):
    """O quê: cada imagem tem entre min_obj e max_obj objetos (5-20).
    Por quê: o enunciado exige 5-20 elipses. Se o gerador produzir menos,
              a métrica não é testada no intervalo certo. Se produzir mais,
              o treinamento demora e pode não convergir em 5 min.
    Como: conta unique(labels) - 1 para cada amostra, verifica 5 <= N <= 20.
    """
    for i in range(len(ds)):
        _, lbl = ds[i]
        n_obj = len(torch.unique(lbl)) - 1
        assert 5 <= n_obj <= 20


# --------------------------------------------------------------------------- reprodutibilidade

def test_reproducibilidade():
    """O quê: mesmo seed gera exatamente os mesmos dados.
    Por quê: o treinamento precisa ser reproduzível para ablações.
              O enunciado pede 2 seeds com média ± desvio — se não for
              determinístico, não dá pra comparar.
    Como: dois datasets com seed=99, verifica igualdade pixel-a-pixel.
    """
    ds1 = SyntheticEllipses(n_samples=5, seed=99)
    ds2 = SyntheticEllipses(n_samples=5, seed=99)
    for i in range(5):
        img1, lbl1 = ds1[i]
        img2, lbl2 = ds2[i]
        assert torch.equal(img1, img2)
        assert torch.equal(lbl1, lbl2)


def test_seeds_diferentes_produzem_dados_diferentes():
    """O quê: seeds diferentes produzem imagens diferentes.
    Por quê: se todos os seeds gerassem os mesmos dados, os 2 seeds da ablação
              seriam redundantes. O gerador precisa realmente variar.
    Como: ds seed=1 vs seed=2, verifica image[0] != image[0].
    """
    ds1 = SyntheticEllipses(n_samples=5, seed=1)
    ds2 = SyntheticEllipses(n_samples=5, seed=2)
    img1, _ = ds1[0]
    img2, _ = ds2[0]
    assert not torch.equal(img1, img2)


# --------------------------------------------------------------------------- fusão (p_touch)

def test_fusao_produz_fusao():
    """O quê: pelo menos uma imagem em 50 tem fusão (blobs < objetos desenhados).
    Por quê: o p_touch sorteado por imagem (0-0,9) deve produzir alguns casos de
              fusão. Se NUNCA houver fusão, o dataset não testa a limitação da
              decodificação ingência — e o mAP não cai.
    Como: itera 50 imagens, conta blobs conexos vs objetos, espera
              n_blobs < n_drawn em pelo menos uma.
    """
    ds = SyntheticEllipses(n_samples=50, seed=0)
    found_fusion = False
    for i in range(len(ds)):
        _, lbl = ds[i]
        n_drawn = len(torch.unique(lbl)) - 1
        binary = (lbl > 0).numpy().astype(np.uint8)
        n_blobs = ndimage.label(binary)[1]
        if n_blobs < n_drawn:
            found_fusion = True
            break
    assert found_fusion, "Nenhuma imagem com fusão encontrada em 50 amostras"


# --------------------------------------------------------------------------- integração decode

def test_decodificacao_decodifica(ds):
    """O quê: labels_from_probability consegue decodificar a imagem sintética.
    Por quê: verifica que a saída do dataset (image) é compatível com a entrada
              esperada pela decodificação (numpy array (H, W) de floats [0, 1]).
              É o elo entre o gerador de dados e o pipeline de avaliação.
    Como: pega imagem[0], passa para labels_from_probability, verifica shape.
    """
    img, lbl = ds[0]
    labels = labels_from_probability(img[0].numpy(), threshold=0.5)
    assert labels.shape == (128, 128)
    assert labels.max() >= 0
