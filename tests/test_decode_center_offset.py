"""Testes da decodificação da Trilha C.

O teste que importa é o `test_alvos_perfeitos_reconstroem_o_gabarito`: se a decodificação
alimentada com alvos perfeitos não devolve o gabarito de volta, ela está errada e nenhum
treino vai consertar isso.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest
import torch

from src.data.targets import center_offset_targets
from src.metrics.instance import MeanAveragePrecision
from src.utils.decode_center_offset import (
    assign_to_centers,
    decode_center_offset,
    find_peaks,
)


def discos(size, especificacoes):
    """Desenha discos: especificacoes = [(cy, cx, raio), ...], rotulados de 1 em diante."""
    labels = np.zeros((size, size), dtype=np.int64)
    y, x = np.ogrid[:size, :size]
    for i, (cy, cx, raio) in enumerate(especificacoes, start=1):
        labels[(y - cy) ** 2 + (x - cx) ** 2 <= raio ** 2] = i
    return labels


def inv_sigmoid(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------ find_peaks

def test_encontra_um_pico_por_objeto():
    """O quê: um heatmap com três gaussianas produz três picos.
    Por quê: cada pico vira um objeto. Picos a mais inventam instâncias; a menos, funde.
    Como: três discos bem separados.
    """
    labels = discos(64, [(15, 15, 5), (15, 45, 5), (45, 30, 5)])
    hm = center_offset_targets(labels)["heatmap"][0]

    picos = find_peaks(hm, threshold=0.3)
    assert len(picos) == 3


def test_pico_fica_no_centroide():
    """O quê: a posição do pico coincide com o centro do objeto.
    Por quê: os offsets apontam para o centroide. Se o pico estiver deslocado, os pontos
              deslocados caem longe dele e a atribuição erra.
    Como: um disco em (20, 30).
    """
    hm = center_offset_targets(discos(64, [(20, 30, 6)]))["heatmap"][0]
    picos = find_peaks(hm, threshold=0.3)

    assert len(picos) == 1
    assert tuple(int(v) for v in picos[0]) == (20, 30)


def test_limiar_descarta_picos_fracos():
    """O quê: máximos locais abaixo do limiar não viram objetos.
    Por quê: a rede produz ativações fracas em lugares errados; sem limiar, cada uma delas
              seria um falso positivo no mAP.
    Como: heatmap com um pico de 0,9 e outro de 0,1; limiar 0,5.
    """
    hm = torch.zeros(32, 32)
    hm[10, 10] = 0.9
    hm[20, 20] = 0.1

    assert len(find_peaks(hm, threshold=0.5)) == 1
    assert len(find_peaks(hm, threshold=0.05)) == 2


def test_picos_proximos_sobrevivem_ao_nms():
    """O quê: dois picos separados por 2 px continuam sendo dois.
    Por quê: os núcleos menores do DSB2018 têm raio 2,3 px — centros vizinhos ficam a
              poucos pixels. Um NMS 3x3 agressivo demais fundiria justamente os casos
              difíceis que a Trilha C existe para resolver.
    Como: dois picos isolados a 2 px de distância.
    """
    hm = torch.zeros(32, 32)
    hm[16, 14] = 0.9
    hm[16, 16] = 0.9

    assert len(find_peaks(hm, threshold=0.5, nms_kernel=3)) == 2


def test_heatmap_vazio():
    """O quê: heatmap sem ativação devolve zero picos, sem exceção.
    Por quê: imagens sem núcleo existem, e `.nonzero()` sobre vazio precisa ser tratado.
    Como: heatmap todo zero.
    """
    picos = find_peaks(torch.zeros(32, 32), threshold=0.3)
    assert len(picos) == 0
    assert picos.shape[1] == 2 if len(picos.shape) > 1 else True


# ------------------------------------------------------------------- assign_to_centers

def test_atribui_pelo_ponto_deslocado():
    """O quê: cada pixel vai para o centro apontado pelo seu offset.
    Por quê: é o coração do método. Se a atribuição usasse a distância do PIXEL ao centro
              em vez do PONTO DESLOCADO, dois objetos encostados se dividiriam pela metade
              geométrica em vez de pela fronteira real.
    Como: dois discos; com alvos perfeitos, cada pixel volta para o seu objeto.
    """
    labels = discos(64, [(20, 20, 7), (20, 40, 7)])
    alvos = center_offset_targets(labels)
    fg = torch.from_numpy(labels != 0)
    centros = torch.tensor([[20, 20], [20, 40]])

    resultado = assign_to_centers(fg, alvos["offsets"], centros).numpy()

    # cada objeto do gabarito deve ter recebido um único rótulo
    for v in (1, 2):
        rotulos = np.unique(resultado[labels == v])
        assert len(rotulos) == 1, f"objeto {v} foi dividido em {len(rotulos)} partes"


def test_sem_centros_devolve_vazio():
    """O quê: sem centros detectados, o label map sai todo zero.
    Por quê: caso de borda que quebra `cdist` com matriz vazia.
    Como: fg cheio, lista de centros vazia.
    """
    fg = torch.ones(16, 16, dtype=torch.bool)
    r = assign_to_centers(fg, torch.zeros(2, 16, 16), torch.zeros(0, 2, dtype=torch.long))
    assert r.max() == 0


def test_fundo_permanece_zero():
    """O quê: pixels fora do foreground ficam com rótulo 0.
    Por quê: rotular fundo como objeto criaria instâncias gigantes e destruiria o mAP.
    Como: um disco pequeno num canvas grande.
    """
    labels = discos(48, [(24, 24, 6)])
    alvos = center_offset_targets(labels)
    fg = torch.from_numpy(labels != 0)

    r = assign_to_centers(fg, alvos["offsets"], torch.tensor([[24, 24]])).numpy()
    assert np.all(r[labels == 0] == 0)


# ----------------------------------------------------- O TESTE DECISIVO (fim a fim)

def test_alvos_perfeitos_reconstroem_o_gabarito():
    """O quê: alimentada com os alvos perfeitos, a decodificação devolve o gabarito.
    Por quê: é o análogo do `test_gabarito_como_predicao_da_map_1` das métricas. Se a
              decodificação não consegue reconstruir os objetos a partir de heatmap e
              offsets EXATOS, o erro está nela — nenhum treino corrigiria isso.
    Como: constrói os alvos do gabarito, converte para logits, decodifica, e compara pelo
          mAP que já temos implementado. Tem que dar 1,0.
    """
    labels = discos(96, [(20, 20, 8), (20, 60, 8), (60, 20, 8), (65, 65, 10)])
    alvos = center_offset_targets(labels)

    seg_logits = torch.from_numpy(inv_sigmoid(alvos["foreground"][0].numpy()))
    hm_logits = torch.from_numpy(inv_sigmoid(alvos["heatmap"][0].numpy()))

    pred = decode_center_offset(seg_logits, hm_logits, alvos["offsets"])

    m_ap, _ = MeanAveragePrecision()(
        torch.from_numpy(np.asarray(pred)).long(), torch.from_numpy(labels)
    )
    assert m_ap == pytest.approx(1.0), f"mAP {m_ap:.3f} — a decodificação perde objetos"


def test_dois_objetos_encostados_saem_separados():
    """O quê: dois discos que se tocam viram DUAS instâncias.
    Por quê: é exatamente o caso que derruba a Parte 1 — lá, limiar + componentes conexos
              devolve um blob só. Se este teste passar, a Trilha C está cumprindo o que
              promete, e este é o resultado que a apresentação precisa mostrar.
    Como: dois discos de raio 8 com centros a 15 px — eles se sobrepõem.
    """
    labels = discos(64, [(32, 25, 8), (32, 39, 8)])
    # confirma que o caso é mesmo adversário: na máscara binária é um blob único
    from scipy import ndimage
    _, n_blobs = ndimage.label(labels != 0)
    assert n_blobs == 1, "o caso de teste precisa ter os dois objetos encostados"

    alvos = center_offset_targets(labels)
    pred = decode_center_offset(
        torch.from_numpy(inv_sigmoid(alvos["foreground"][0].numpy())),
        torch.from_numpy(inv_sigmoid(alvos["heatmap"][0].numpy())),
        alvos["offsets"],
    )

    pred = np.asarray(pred)
    assert len(np.unique(pred[pred != 0])) == 2


def test_imagem_sem_objetos():
    """O quê: entrada sem nenhum objeto devolve label map vazio, sem exceção.
    Por quê: acontece de verdade no dataset, e um crash aqui derrubaria a avaliação inteira.
    Como: logits muito negativos nos dois canais.
    """
    pred = decode_center_offset(
        torch.full((32, 32), -10.0), torch.full((32, 32), -10.0), torch.zeros(2, 32, 32)
    )
    assert np.asarray(pred).max() == 0


def test_saida_no_formato_da_metrica():
    """O quê: a saída é inteira, com 0 = fundo e rótulos contíguos a partir de 1.
    Por quê: é o contrato que `src/metrics/instance` assume. Rótulos com buracos ou tipo
              float quebrariam a contagem de objetos.
    Como: confere dtype e o conjunto de rótulos.
    """
    labels = discos(64, [(20, 20, 6), (44, 44, 6)])
    alvos = center_offset_targets(labels)
    pred = np.asarray(decode_center_offset(
        torch.from_numpy(inv_sigmoid(alvos["foreground"][0].numpy())),
        torch.from_numpy(inv_sigmoid(alvos["heatmap"][0].numpy())),
        alvos["offsets"],
    ))

    assert np.issubdtype(pred.dtype, np.integer)
    assert sorted(np.unique(pred)) == [0, 1, 2]
