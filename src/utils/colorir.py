"""Pinta um label map de instâncias com uma cor por objeto.

Estava duplicada em três módulos (avaliação, mosaico e galeria de falhas); virou uma função
só quando o notebook de inferência precisou da quarta cópia.

A paleta é sorteada com semente fixa, então a mesma imagem sai sempre com as mesmas cores —
o que importa para comparar figuras do antes e do depois lado a lado.
"""

import numpy as np


def colorir(rotulos, seed: int = 0) -> np.ndarray:
    """Label map (H, W) → imagem RGB (H, W, 3) em [0, 1], com o fundo preto.

    Args:
        rotulos: 0 = fundo, 1..N = instâncias. Os números podem ser esparsos.
        seed: semente da paleta.

    Returns:
        Array (H, W, 3) de floats. O rótulo 0, se existir, fica preto; os demais recebem
        cores aleatórias claras (componentes em [0,2; 1,0]), para contrastarem com o fundo.
    """
    rotulos = np.asarray(rotulos)
    ids, compacto = np.unique(rotulos, return_inverse=True)
    compacto = compacto.reshape(rotulos.shape)

    rng = np.random.default_rng(seed)
    paleta = rng.random((len(ids), 3)) * 0.8 + 0.2
    paleta[ids == 0] = 0.0          # fundo preto — só age se houver rótulo 0 no recorte
    return paleta[compacto]
