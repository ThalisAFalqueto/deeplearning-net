"""Testes da Parte 4 — mosaico, grade de tiles e costura.

A maioria dos testes usa um **oráculo**: em vez de uma rede, a "predição" de cada tile é o
gabarito recortado naquele tile. Assim o que se testa é só a costura — se a ingênua parte um
objeto aqui, a culpa é da costura, não da rede.

A cena de teste tem 448×448 px: com tile 256 e passo 192 são 2×2 tiles, sobreposição de 64 px
(192–255) e divisa entre donos em 224, nos dois eixos.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.targets import center_offset_targets
from src.metrics.instance import MeanAveragePrecision
from src.mosaic import (
    costura_densa,
    costura_ingenua,
    fusao_por_iou,
    grade_tiles,
    inferir_tiles,
    mapa_de_donos,
    media_densa,
    montar_mosaico,
    montar_mosaicos,
    peso_interior,
    posicoes,
)
from src.mosaic.runner import ids_que_cruzam
from src.utils.decode_center_offset import decode_center_offset

TILE, PASSO, LADO = 256, 192, 448

# (centro y, centro x, raio)
CENA = [
    (100, 224, 10),   # 1: cruza a divisa vertical (x = 224), inteiro dentro da faixa
    (224, 224, 12),   # 2: no cruzamento das duas divisas — pedaços em 4 tiles
    (224, 400, 10),   # 3: cruza a divisa horizontal
    (300, 256, 8),    # 4: cruza a BORDA do tile da esquerda (x = 256), mas não a divisa
    (60, 60, 9),      # 5: interior
    (380, 380, 9),    # 6: interior
]


def cena_de_discos(lado: int = LADO, discos=CENA) -> np.ndarray:
    yy, xx = np.mgrid[:lado, :lado]
    rotulos = np.zeros((lado, lado), dtype=np.int64)
    for i, (cy, cx, r) in enumerate(discos, start=1):
        rotulos[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return rotulos


def oraculo_por_tile(gt: np.ndarray, grade, tile: int) -> list[np.ndarray]:
    """Predição perfeita de cada tile, numerada de 1 a n dentro do tile — como uma
    decodificação de verdade, que não sabe os números usados no tile vizinho."""
    saida = []
    for y, x in grade:
        recorte = gt[y:y + tile, x:x + tile]
        _, compacto = np.unique(recorte, return_inverse=True)   # 0 (fundo) continua 0
        saida.append(compacto.reshape(recorte.shape))
    return saida


def n_objetos(rotulos: np.ndarray) -> int:
    return int(len(np.unique(rotulos[rotulos > 0])))


# ------------------------------------------------------------------------------- grade

def test_posicoes_dos_tiles():
    """O quê: onde os tiles começam, incluindo o último encostado na borda.
    Por quê: sem o tile encostado, a faixa final da imagem ficaria sem previsão.
    """
    assert posicoes(1024, 256, 192) == [0, 192, 384, 576, 768]
    assert posicoes(448, 256, 192) == [0, 192]
    assert posicoes(500, 256, 192) == [0, 192, 244]
    assert posicoes(256, 256, 192) == [0]


@pytest.mark.parametrize("lado", [1024, 500])
def test_grade_cobre_tudo_e_donos_ficam_dentro_do_tile(lado):
    """O quê: todo pixel é coberto por algum tile, e o tile dono de cada pixel o contém.
    Por quê: um pixel sem cobertura ficaria sem previsão; um dono que não contém o pixel
              faria a costura ler fora do tile.
    """
    grade = grade_tiles(lado, lado, TILE, PASSO)
    cobertura = np.zeros((lado, lado), dtype=int)
    donos = mapa_de_donos(lado, lado, TILE, PASSO)
    for k, (y, x) in enumerate(grade):
        cobertura[y:y + TILE, x:x + TILE] += 1
        fora = np.ones((lado, lado), dtype=bool)
        fora[y:y + TILE, x:x + TILE] = False
        assert not (fora & (donos == k)).any(), f"tile {k} dono de pixel fora dele"
    assert cobertura.min() >= 1


def test_divisas_no_meio_da_sobreposicao():
    """O quê: com 1024/256/192, as divisas entre donos caem em 224, 416, 608 e 800.
    Por quê: é a "parte interna" do slide — cada tile descarta os 32 px mais perto da borda.
    """
    donos = mapa_de_donos(1024, 1024, TILE, PASSO)
    trocas = np.nonzero(np.diff(donos[0]))[0] + 1
    assert trocas.tolist() == [224, 416, 608, 800]


def test_peso_interior():
    """O quê: o peso é 1 no interior, cai até a borda e nunca zera."""
    w = peso_interior(TILE, rampa=64)
    assert w.max() == 1.0 and w[128, 128] == 1.0
    assert w.min() > 0
    assert w[0, 0] == pytest.approx(1 / 64 ** 2)
    assert np.allclose(w, w[::-1, ::-1]) and np.allclose(w, w.T)


# ----------------------------------------------------------------------------- mosaico

def test_montar_mosaico_ids_unicos_e_blocos_no_lugar():
    """O quê: 4 imagens 8×8 viram uma 16×16; cada bloco na sua posição; rótulos sem colisão.
    Por quê: o núcleo 1 de uma imagem e o núcleo 1 de outra são objetos diferentes.
    """
    amostras = []
    for j in range(4):
        rot = torch.zeros(8, 8, dtype=torch.int64)
        rot[1:3, 1:3], rot[5:7, 5:7] = 1, 2
        amostras.append((torch.full((1, 8, 8), float(j)), rot))

    imagem, rotulos = montar_mosaico(amostras)

    assert imagem.shape == (1, 16, 16) and rotulos.shape == (16, 16)
    assert n_objetos(rotulos) == 8
    for j in range(4):
        r, c = divmod(j, 2)
        assert (imagem[0, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] == j).all()


def test_montar_mosaicos_segue_a_ordem_e_descarta_o_resto():
    """O quê: com ``ordem``, os mosaicos pegam as imagens nessa ordem; o que não fecha um
    mosaico inteiro fica de fora.
    Por quê: é como o runner agrupa as imagens por modalidade.
    """
    dataset = [(torch.full((1, 4, 4), float(i)), torch.zeros(4, 4, dtype=torch.int64))
               for i in range(10)]

    mosaicos = montar_mosaicos(dataset, lado=2, ordem=[9, 8, 7, 6, 5, 4, 3, 2, 1])

    assert [fontes for _, _, fontes in mosaicos] == [[9, 8, 7, 6], [5, 4, 3, 2]]
    assert mosaicos[0][0][0, 0, 0] == 9


# ----------------------------------------------------------------------- mapas densos

def test_media_densa_reconstroi_modelo_pixel_a_pixel():
    """O quê: se a rede só olha o próprio pixel, costurar os tiles reproduz exatamente a
    passada na imagem inteira — com média simples, ponderada ou só pela parte interna.
    Por quê: isola a aritmética da costura. Com a rede de verdade, a diferença que sobrar é
              contexto perdido na borda do tile, não erro de índice.
    Como: conv 1×1 aleatória como "rede".
    """
    torch.manual_seed(0)
    rede = nn.Conv2d(1, 3, kernel_size=1).eval()
    imagem = torch.randn(1, LADO, LADO)
    grade = grade_tiles(LADO, LADO, TILE, PASSO)
    donos = mapa_de_donos(LADO, LADO, TILE, PASSO)

    with torch.no_grad():
        inteira = rede(imagem[None])[0]
    saidas, tamanhos = inferir_tiles(rede, imagem, grade, TILE)

    assert tamanhos is None and saidas.shape == (len(grade), 3, TILE, TILE)
    for costurada in (media_densa(saidas, grade, LADO, LADO),
                      media_densa(saidas, grade, LADO, LADO, peso_interior(TILE, 64)),
                      costura_densa(saidas, grade, donos, TILE)):
        assert torch.allclose(costurada, inteira, atol=1e-5)


# ------------------------------------------------------------------ rótulos (oráculo)

def test_ids_que_cruzam():
    """O quê: os objetos 1, 2 e 3 cruzam uma divisa; o 4 cruza só a borda de um tile."""
    gt = cena_de_discos()
    donos = mapa_de_donos(LADO, LADO, TILE, PASSO)
    assert ids_que_cruzam(gt, donos).tolist() == [1, 2, 3]


def test_costura_ingenua_parte_os_objetos_da_divisa():
    """O quê: mesmo com predição perfeita em cada tile, a costura ingênua parte os objetos
    que cruzam a divisa — o do cruzamento vira 4.
    Por quê: é o item 3 da Parte 4. Mostra que o problema é da costura, não da rede: os
              números de instância de tiles diferentes não conversam.
    Como: objetos 1, 2 e 3 cruzam divisa e viram 2 + 4 + 2 pedaços; os outros 3 ficam
          inteiros — 11 no total, contra 6 no gabarito.
    """
    gt = cena_de_discos()
    grade = grade_tiles(LADO, LADO, TILE, PASSO)
    donos = mapa_de_donos(LADO, LADO, TILE, PASSO)

    ingenua = costura_ingenua(oraculo_por_tile(gt, grade, TILE), grade, donos, TILE)

    assert n_objetos(ingenua) == 2 + 4 + 2 + 3
    # o objeto 4 cruza a borda do tile, mas a divisa não: fica inteiro
    assert n_objetos(ingenua * (gt == 4)) == 1


def test_fusao_por_iou_devolve_o_gabarito():
    """O quê: com predição perfeita por tile, a fusão por IoU na faixa devolve exatamente os
    6 objetos — mAP 1.
    Por quê: é a correção A. Na faixa de sobreposição os dois tiles veem os mesmos pixels,
              então os pedaços do mesmo núcleo coincidem ali.
    """
    gt = cena_de_discos()
    grade = grade_tiles(LADO, LADO, TILE, PASSO)
    donos = mapa_de_donos(LADO, LADO, TILE, PASSO)

    fundida = fusao_por_iou(oraculo_por_tile(gt, grade, TILE), grade, donos, TILE)

    assert n_objetos(fundida) == 6
    m_ap, _ = MeanAveragePrecision()(torch.from_numpy(fundida), torch.from_numpy(gt))
    assert m_ap == pytest.approx(1.0)


def _saidas_oraculo_trilha_c(gt, grade, tile) -> torch.Tensor:
    """Saída "perfeita" da Trilha C em cada tile, calculada só com o que o tile vê.

    Um núcleo cortado pela borda do tile tem, nesse tile, o centro do pedaço visível — é o
    que uma rede de verdade também faria. Logits: ±10 no foreground; logit do heatmap.
    """
    saidas = []
    for y, x in grade:
        alvos = center_offset_targets(gt[y:y + tile, x:x + tile])
        seg = (alvos["foreground"] * 2 - 1) * 10.0
        heat = torch.logit(alvos["heatmap"].clamp(1e-4, 1 - 1e-4))
        saidas.append(torch.cat([seg, heat, alvos["offsets"]]))
    return torch.stack(saidas)


def test_media_interior_recupera_objetos_sem_fusao():
    """O quê: média dos mapas densos, ponderada pelo interior, e decodificação única no
    mosaico recuperam os 6 objetos — inclusive o 4, que o tile da esquerda vê cortado.
    Por quê: é a correção B, a que a Trilha C permite. Offsets são deslocamentos, não
              números de objeto: a média entre tiles faz sentido. O peso de interior faz o
              tile que vê o núcleo inteiro dominar o que o vê cortado na borda.
    """
    gt = cena_de_discos()
    grade = grade_tiles(LADO, LADO, TILE, PASSO)
    saidas = _saidas_oraculo_trilha_c(gt, grade, TILE)

    media = media_densa(saidas, grade, LADO, LADO, peso_interior(TILE, TILE - PASSO))
    rotulos = decode_center_offset(media[0], media[1], media[2:], peak_threshold=0.5)

    assert n_objetos(rotulos) == 6
    m_ap, _ = MeanAveragePrecision()(torch.from_numpy(rotulos), torch.from_numpy(gt))
    assert m_ap == pytest.approx(1.0)
