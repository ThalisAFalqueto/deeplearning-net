"""Decodificação ingênua: limiar + componentes conexos.

É o método da Parte 1 do enunciado — o que fracassa quando os objetos se tocam, e cujo
fracasso a Parte 2 vai corrigir trocando a representação de saída da rede.

A rede prevê UMA probabilidade por pixel ("isto é objeto?"). Essa probabilidade não
carrega nenhuma informação sobre QUAL objeto — então tudo que dá para fazer é cortar num
limiar e chamar de "um objeto" cada mancha de pixels grudados.
"""

import numpy as np
from scipy import ndimage


def labels_from_probability(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converte um mapa de probabilidade em label map de instâncias.

    Args:
        prob: array (H, W) de floats em [0, 1] — a saída da rede após a sigmoide.
        threshold: corte para considerar um pixel como objeto.

    Returns:
        Array (H, W) de inteiros: 0 = fundo, 1..N = instâncias. É o mesmo formato que
        `src/metrics/instance.py` consome.
    """
    # `ndimage.label` recebe uma máscara binária e numera cada grupo de pixels
    # conectados. Devolve uma tupla: (label map, quantidade de grupos).
    raise NotImplementedError


def remove_small_objects(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove instâncias com menos de `min_area` pixels e renumera de 1 a N.

    Ruído no mapa de probabilidade vira instâncias minúsculas depois do limiar, e cada
    uma delas conta como falso positivo. Filtrar por área é o pós-processamento mais
    barato que existe — e o efeito dele no mAP vale ser medido.

    Args:
        labels: label map de instâncias.
        min_area: área mínima, em pixels, para uma instância sobreviver.

    Returns:
        Label map com as instâncias pequenas removidas e as restantes renumeradas
        de forma contígua.
    """
    raise NotImplementedError
