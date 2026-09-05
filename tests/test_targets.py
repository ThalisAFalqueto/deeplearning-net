"""Testes da geração de alvos da Trilha C (centro + offsets).

Um alvo errado torna todo o resto inútil: a rede aprenderia a prever a coisa errada e nada
denunciaria, porque a perda cairia normalmente.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch

from src.data.targets import center_offset_targets, instance_sigma


def disco(size, cy, cx, raio, label=1, canvas=None):
    """Desenha um disco de raio `raio` centrado em (cy, cx)."""
    if canvas is None:
        canvas = np.zeros((size, size), dtype=np.int64)
    y, x = np.ogrid[:size, :size]
    canvas[(y - cy) ** 2 + (x - cx) ** 2 <= raio ** 2] = label
    return canvas


def test_heatmap_tem_maximo_no_centroide():
    """O quê: o pico do heatmap coincide com o centroide do objeto.
    Por quê: a decodificação localiza objetos procurando picos. Se o pico não estiver no
              centro, os offsets apontam para um lugar e o pico está em outro.
    Como: um disco em (20, 30); o argmax do heatmap tem que cair ali.
    """
    labels = disco(64, 20, 30, raio=6)
    hm = center_offset_targets(labels)["heatmap"][0].numpy()

    pico = np.unravel_index(hm.argmax(), hm.shape)
    assert pico == (20, 30)
    assert hm.max() == pytest.approx(1.0)


def test_offset_aponta_para_o_centro():
    """O quê: para todo pixel de foreground, pixel + offset == centroide.
    Por quê: é a propriedade que DEFINE o alvo. A decodificação desloca cada pixel pelo seu
              offset e espera cair no centro; se a identidade não valer, nada funciona.
    Como: verifica a igualdade em todos os pixels do objeto.
    """
    labels = disco(64, 25, 40, raio=8)
    alvos = center_offset_targets(labels)
    offsets = alvos["offsets"].numpy()

    ys, xs = np.nonzero(labels)
    cy, cx = ys.mean(), xs.mean()

    destino_y = ys + offsets[0][ys, xs]
    destino_x = xs + offsets[1][ys, xs]

    np.testing.assert_allclose(destino_y, cy, atol=1e-4)
    np.testing.assert_allclose(destino_x, cx, atol=1e-4)


def test_offsets_sao_zero_no_fundo():
    """O quê: offsets valem zero fora do foreground.
    Por quê: um pixel de fundo não pertence a objeto nenhum — não há centro para apontar.
              A perda ignora essas posições, mas o alvo precisa ser inspecionável.
    Como: verifica que todo pixel com label 0 tem offset (0, 0).
    """
    labels = disco(64, 20, 20, raio=5)
    offsets = center_offset_targets(labels)["offsets"].numpy()

    fundo = labels == 0
    assert np.all(offsets[0][fundo] == 0)
    assert np.all(offsets[1][fundo] == 0)


def test_dois_objetos_vizinhos_nao_criam_pico_espurio():
    """O quê: o heatmap de dois objetos encostados tem exatamente DOIS picos.
    Por quê: se as gaussianas fossem somadas, o ponto médio entre dois centros próximos
              poderia superar os próprios centros e virar um terceiro objeto inexistente.
              Combinar por máximo evita isso — este teste protege essa escolha.
    Como: dois discos encostados; conta máximos locais estritos acima de 0,5.
    """
    labels = disco(64, 32, 26, raio=6, label=1)
    labels = disco(64, 32, 38, raio=6, label=2, canvas=labels)

    hm = torch.from_numpy(center_offset_targets(labels)["heatmap"][0].numpy())
    maxpool = torch.nn.functional.max_pool2d(hm[None, None], 3, stride=1, padding=1)[0, 0]
    picos = ((hm == maxpool) & (hm > 0.5)).nonzero()

    assert len(picos) == 2
    # e nenhum pico no meio do caminho entre os dois centros
    assert not any(abs(int(p[1]) - 32) <= 1 and int(p[0]) == 32 for p in picos)


def test_sigma_cresce_com_o_tamanho():
    """O quê: objetos maiores recebem gaussianas mais largas.
    Por quê: os núcleos vão de 2,3 a 16,5 px de raio. Sigma fixo apaga os pequenos ou
              borra os grandes até fundir vizinhos.
    Como: compara sigma de áreas crescentes, respeitando os limites [1, 6].
    """
    pequeno = instance_sigma(np.pi * 3 ** 2)
    medio = instance_sigma(np.pi * 9 ** 2)
    grande = instance_sigma(np.pi * 30 ** 2)

    assert pequeno < medio < grande
    assert pequeno >= 1.0
    assert grande <= 6.0


def test_foreground_bate_com_o_label_map():
    """O quê: o canal de foreground é exatamente `labels != 0`.
    Por quê: é o alvo da cabeça de segmentação, que precisa concordar com o gabarito usado
              pela métrica semântica.
    Como: compara os dois diretamente, com duas instâncias.
    """
    labels = disco(64, 20, 20, raio=5, label=1)
    labels = disco(64, 45, 45, raio=7, label=2, canvas=labels)

    fg = center_offset_targets(labels)["foreground"][0].numpy()
    np.testing.assert_array_equal(fg, (labels != 0).astype(np.float32))


def test_imagem_vazia():
    """O quê: sem objetos, tudo é zero e nada explode.
    Por quê: imagens sem núcleo existem, e um laço sobre `np.unique` vazio precisa ser
              tratado sem exceção.
    Como: label map todo zero.
    """
    alvos = center_offset_targets(np.zeros((32, 32), dtype=np.int64))
    assert alvos["heatmap"].max() == 0
    assert alvos["offsets"].abs().max() == 0
    assert alvos["foreground"].max() == 0


def test_shapes_e_tipos():
    """O quê: os alvos têm as formas que o modelo e a perda esperam.
    Por quê: a saída da rede é (4, H, W) — 1 foreground, 1 heatmap, 2 offsets. Uma forma
              errada só apareceria como erro de broadcast no meio do treino.
    Como: confere shape e dtype dos três.
    """
    alvos = center_offset_targets(disco(48, 24, 24, raio=6))
    assert alvos["foreground"].shape == (1, 48, 48)
    assert alvos["heatmap"].shape == (1, 48, 48)
    assert alvos["offsets"].shape == (2, 48, 48)
    for t in alvos.values():
        assert t.dtype == torch.float32
