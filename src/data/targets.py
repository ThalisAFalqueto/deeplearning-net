"""Alvos da Trilha C: centro + offsets.

A Parte 1 mostrou que uma máscara binária não carrega informação suficiente para separar
objetos encostados. A Trilha C troca o que a rede prevê: além do foreground, ela prevê
**onde está o centro** de cada objeto e, para cada pixel, **um vetor apontando para o centro
do seu próprio objeto**.

Isso resolve o problema da permutação de rótulos — o offset é uma grandeza geométrica local,
não um identificador arbitrário. Dois núcleos idênticos em posições diferentes produzem
offsets diferentes, e a identidade das instâncias é reconstruída depois, na decodificação.

Este módulo constrói os alvos a partir do label map do gabarito:

    foreground  (1, H, W)   1 onde há objeto
    heatmap     (1, H, W)   gaussiana centrada no centroide de cada objeto
    offsets     (2, H, W)   (Δy, Δx) do pixel até o centro do seu objeto
"""

import numpy as np
import torch

SIGMA_MIN = 1.0
SIGMA_MAX = 6.0


def instance_sigma(area: float) -> float:
    """Desvio-padrão da gaussiana de um objeto, proporcional ao seu tamanho.

    Os núcleos do DSB2018 têm raio equivalente de 2,3 px (p5) a 16,5 px (p95). Um sigma
    fixo não serve para as duas pontas: pequeno demais faz a gaussiana do núcleo grande
    virar um ponto isolado no meio dele; grande demais borra os pequenos até que dois
    vizinhos se fundam num pico só.

    Args:
        area: área do objeto em pixels.

    Returns:
        Sigma em pixels, limitado a [1, 6].
    """
    raio = np.sqrt(area / np.pi)
    return float(np.clip(raio / 3.0, SIGMA_MIN, SIGMA_MAX))


def center_offset_targets(labels) -> dict:
    """Constrói os alvos de foreground, heatmap e offsets a partir do label map.

    Args:
        labels: label map (H, W) — tensor ou array, 0 = fundo, 1..N = instâncias.

    Returns:
        Dicionário com tensores float32:
            ``foreground`` (1, H, W), ``heatmap`` (1, H, W), ``offsets`` (2, H, W).

        Os offsets valem zero fora do foreground; a perda os ignora lá de qualquer forma,
        mas zero deixa o alvo inspecionável.
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    altura, largura = labels_np.shape
    foreground = (labels_np != 0).astype(np.float32)
    heatmap = np.zeros((altura, largura), dtype=np.float32)
    offsets = np.zeros((2, altura, largura), dtype=np.float32)

    yy, xx = np.mgrid[:altura, :largura].astype(np.float32)

    for label in np.unique(labels_np[labels_np != 0]):
        mask = labels_np == label
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()

        sigma = instance_sigma(mask.sum())
        gaussiana = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))

        # MÁXIMO, não soma: somar as gaussianas de dois núcleos vizinhos criaria um pico
        # espúrio entre eles, e a decodificação leria isso como um terceiro objeto.
        np.maximum(heatmap, gaussiana, out=heatmap)

        # o offset aponta do pixel PARA o centro, de modo que (pixel + offset) == centro
        offsets[0][mask] = cy - ys
        offsets[1][mask] = cx - xs

    return {
        "foreground": torch.from_numpy(foreground).unsqueeze(0),
        "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
        "offsets": torch.from_numpy(offsets),
    }


def batch_center_offset_targets(labels_batch: torch.Tensor) -> dict:
    """Versão em lote de :func:`center_offset_targets`.

    Args:
        labels_batch: (B, H, W) de label maps.

    Returns:
        Dicionário com ``foreground`` (B, 1, H, W), ``heatmap`` (B, 1, H, W) e
        ``offsets`` (B, 2, H, W).
    """
    alvos = [center_offset_targets(labels) for labels in labels_batch]
    return {
        chave: torch.stack([a[chave] for a in alvos])
        for chave in ("foreground", "heatmap", "offsets")
    }
