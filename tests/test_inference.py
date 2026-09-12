"""Testes da inferência avulsa — o que o `notebooks/inferencia.ipynb` usa por baixo.

O enunciado pede um notebook que receba **o caminho de uma imagem qualquer**. "Qualquer"
inclui tamanho e modo de cor que a rede nunca viu: ela treinou em 256×256 em tons de cinza.
Estes testes cobrem justamente a ida e a volta desse ajuste, porque é onde dá errado em
silêncio — uma máscara devolvida em 256² para quem passou uma imagem 696×520 não casa com
pixel nenhum, e ninguém percebe olhando só a contagem.

Nada aqui depende do checkpoint treinado: o modelo é montado pequeno na hora e salvo na pasta
temporária do teste.
"""

import json
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.inference.predict import _preparar, _reamostrar_rotulos, carregar_modelo, prever
from src.models.factory import ModelFactoryRegistry


@pytest.fixture
def modelo_de_brinquedo(tmp_path):
    """Config + checkpoint de um modelo pequeno, para não depender de treino."""
    cfg = {
        "seed": 0,
        "data": {"kind": "dsb2018", "size": 256, "data_dir": "x", "val_fraction": 0.15},
        "model": {"name": "unet_improved", "base": 8, "depth": 2,
                  "loss": {"name": "center_offset"}},
        "train": {"epochs": 1, "batch_size": 2, "lr": 0.001, "num_workers": 0},
        "decode": {"threshold": 0.5, "peak_threshold": 0.5, "nms_kernel": 3},
        "output_dir": str(tmp_path / "saida"),
    }
    caminho_cfg = tmp_path / "cfg.yaml"
    caminho_cfg.write_text(yaml.safe_dump(cfg))

    from types import SimpleNamespace
    modelo = ModelFactoryRegistry.build(SimpleNamespace(model=cfg["model"]))
    caminho_ck = tmp_path / "pesos.pth"
    torch.save({"model": modelo.state_dict()}, caminho_ck)
    return str(caminho_cfg), str(caminho_ck)


def escrever_imagem(caminho: Path, largura: int, altura: int, modo: str = "L") -> Path:
    """Grava uma imagem com um quadrado claro no meio."""
    arr = np.zeros((altura, largura), dtype=np.uint8)
    arr[altura // 4: 3 * altura // 4, largura // 4: 3 * largura // 4] = 220
    img = Image.fromarray(arr, mode="L")
    if modo == "RGB":
        img = img.convert("RGB")
    img.save(caminho)
    return caminho


# ------------------------------------------------------------------ reamostragem de rótulos

def test_reamostrar_volta_ao_tamanho_original():
    """O quê: o mapa de rótulos volta exatamente no tamanho (altura, largura) pedido."""
    rotulos = np.array([[0, 1], [2, 2]], dtype=np.int64)
    saida = _reamostrar_rotulos(rotulos, (6, 4))          # (largura, altura), convenção do PIL
    assert saida.shape == (4, 6)


def test_reamostrar_nao_inventa_rotulos():
    """O quê: nenhum valor novo aparece ao ampliar.
    Por quê: é o motivo de usar vizinho mais próximo à mão em vez de interpolar — interpolação
              criaria rótulos intermediários entre dois objetos vizinhos, objetos fantasma que
              não existem em lugar nenhum.
    """
    rng = np.random.default_rng(0)
    rotulos = rng.integers(0, 5, size=(16, 16)).astype(np.int64)
    grande = _reamostrar_rotulos(rotulos, (97, 53))
    assert set(np.unique(grande)).issubset(set(np.unique(rotulos)))


def test_reamostrar_preserva_um_objeto_solto():
    """O quê: ampliar mantém o objeto, e reduzir não o apaga se ele for grande o bastante."""
    rotulos = np.zeros((32, 32), dtype=np.int64)
    rotulos[8:24, 8:24] = 7
    assert 7 in np.unique(_reamostrar_rotulos(rotulos, (128, 128)))
    assert 7 in np.unique(_reamostrar_rotulos(rotulos, (16, 16)))


# -------------------------------------------------------------------------- preparo da imagem

@pytest.mark.parametrize("largura, altura", [(256, 256), (320, 256), (696, 520), (97, 53)])
def test_preparar_normaliza_qualquer_tamanho(tmp_path, largura, altura):
    """O quê: qualquer tamanho vira (1, 256, 256) em [0,1], e o tamanho original é devolvido."""
    caminho = escrever_imagem(tmp_path / "img.png", largura, altura)
    tensor, tamanho = _preparar(caminho)

    assert tensor.shape == (1, 256, 256)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    assert tamanho == (largura, altura)


def test_preparar_trata_cor_e_cinza_igual(tmp_path):
    """O quê: a mesma imagem em RGB e em tons de cinza dá o mesmo tensor.
    Por quê: "uma imagem qualquer" inclui PNG colorido; se a conversão mudasse os valores, o
              resultado dependeria do modo de cor do arquivo.
    """
    cinza = escrever_imagem(tmp_path / "cinza.png", 120, 80, modo="L")
    colorida = escrever_imagem(tmp_path / "cor.png", 120, 80, modo="RGB")
    assert torch.equal(_preparar(cinza)[0], _preparar(colorida)[0])


def test_arquivo_inexistente_da_erro_claro(tmp_path):
    """O quê: caminho errado levanta FileNotFoundError, não um erro obscuro de tensor."""
    with pytest.raises(FileNotFoundError, match="imagem não encontrada"):
        _preparar(tmp_path / "nao_existe.png")


def test_checkpoint_inexistente_da_erro_claro(tmp_path):
    """O quê: checkpoint ausente explica onde está o modelo final."""
    with pytest.raises(FileNotFoundError, match="checkpoint não encontrado"):
        carregar_modelo("configs/p2_dsb2018.yaml", str(tmp_path / "nao_existe.pth"))


# ------------------------------------------------------------------------------ ponta a ponta

def test_prever_devolve_mascara_no_tamanho_da_entrada(tmp_path, modelo_de_brinquedo):
    """O quê: numa imagem não quadrada, o mapa de rótulos volta em (altura, largura) dela.
    Por quê: é o erro silencioso que este módulo existe para evitar.
    """
    caminho_cfg, caminho_ck = modelo_de_brinquedo
    modelo, cfg, _ = carregar_modelo(caminho_cfg, caminho_ck)
    imagem = escrever_imagem(tmp_path / "retangular.png", 320, 200)

    r = prever(imagem, modelo, cfg)

    assert r["rotulos"].shape == (200, 320)
    assert r["imagem"].shape == (200, 320)
    assert r["probabilidade"].shape == (256, 256)     # a rede sempre vê 256²


def test_contagem_bate_com_os_rotulos(tmp_path, modelo_de_brinquedo):
    """O quê: `contagem` é o número de rótulos distintos diferentes de zero."""
    caminho_cfg, caminho_ck = modelo_de_brinquedo
    modelo, cfg, _ = carregar_modelo(caminho_cfg, caminho_ck)
    imagem = escrever_imagem(tmp_path / "img.png", 256, 256)

    r = prever(imagem, modelo, cfg)

    esperado = len(np.unique(r["rotulos"][r["rotulos"] > 0]))
    assert r["contagem"] == esperado


def test_sem_tamanho_original_devolve_256(tmp_path, modelo_de_brinquedo):
    """O quê: `tamanho_original=False` devolve como a rede viu, para inspecionar o modelo."""
    caminho_cfg, caminho_ck = modelo_de_brinquedo
    modelo, cfg, _ = carregar_modelo(caminho_cfg, caminho_ck)
    imagem = escrever_imagem(tmp_path / "img.png", 320, 200)

    r = prever(imagem, modelo, cfg, tamanho_original=False)

    assert r["rotulos"].shape == (256, 256)


def test_notebook_e_json_valido_e_sem_saidas():
    """O quê: o entregável `inferencia.ipynb` é um notebook válido e não carrega saídas.
    Por quê: saídas embutidas incham o diff e escondem se o notebook realmente roda. O
              avaliador executa; o repositório guarda só o código.
    """
    nb = json.loads((_TESTS_DIR.parent / "notebooks" / "inferencia.ipynb").read_text())

    assert nb["nbformat"] == 4
    celulas_codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert celulas_codigo, "o notebook não tem célula de código"
    assert all(not c.get("outputs") for c in celulas_codigo)
    assert all(c.get("execution_count") is None for c in celulas_codigo)
