"""Monta imagens grandes a partir de imagens do dataset.

O DSB2018 não tem imagens grandes o bastante para exigir tiles: a rede treina em 256² e o
split de validação chega nesse tamanho depois do resize. O enunciado sugere a saída — juntar
várias imagens do dataset numa só. Um mosaico 4×4 de imagens 256² é uma imagem 1024² com
gabarito conhecido.

Os rótulos de cada bloco são deslocados para não colidirem: o núcleo 3 da primeira imagem e
o núcleo 3 da segunda são objetos diferentes e precisam de números diferentes no mosaico.
"""

import numpy as np
import torch


def montar_mosaico(amostras: list) -> tuple[torch.Tensor, np.ndarray]:
    """Junta ``lado²`` amostras numa grade ``lado × lado``, em ordem de leitura.

    Args:
        amostras: lista de pares ``(imagem (1, s, s), rótulos (s, s))`` — o que o dataset
            devolve. O tamanho da lista tem que ser um quadrado perfeito.

    Returns:
        ``imagem`` (1, lado·s, lado·s) e ``rotulos`` (lado·s, lado·s) com 0 = fundo e IDs
        únicos no mosaico inteiro.
    """
    lado = int(round(len(amostras) ** 0.5))
    if lado * lado != len(amostras):
        raise ValueError(f"{len(amostras)} amostras não formam uma grade quadrada")

    s = amostras[0][0].shape[-1]
    imagem = torch.zeros(1, lado * s, lado * s)
    rotulos = np.zeros((lado * s, lado * s), dtype=np.int64)

    proximo = 0
    for j, (img, rot) in enumerate(amostras):
        r, c = divmod(j, lado)
        rot = rot.numpy() if isinstance(rot, torch.Tensor) else np.asarray(rot)
        bloco = np.where(rot > 0, rot.astype(np.int64) + proximo, 0)

        imagem[:, r * s:(r + 1) * s, c * s:(c + 1) * s] = img
        rotulos[r * s:(r + 1) * s, c * s:(c + 1) * s] = bloco
        proximo = max(proximo, int(bloco.max()))

    return imagem, rotulos


def montar_mosaicos(
    dataset, lado: int = 4, ordem: list[int] | None = None
) -> list[tuple[torch.Tensor, np.ndarray, list[int]]]:
    """Agrupa o dataset em mosaicos de ``lado²`` imagens consecutivas.

    Args:
        ordem: índices do dataset na ordem em que entram nos mosaicos (por exemplo,
            agrupados por modalidade). ``None`` = a ordem do dataset, que é determinística
            pela seed do split. As imagens que sobram no fim (menos de ``lado²``) ficam de
            fora.

    Returns:
        Lista de ``(imagem, rotulos, indices_das_imagens_fonte)``.
    """
    ordem = list(range(len(dataset))) if ordem is None else list(ordem)
    n = lado * lado
    mosaicos = []
    for inicio in range(0, len(ordem) - n + 1, n):
        indices = ordem[inicio:inicio + n]
        imagem, rotulos = montar_mosaico([dataset[i] for i in indices])
        mosaicos.append((imagem, rotulos, indices))
    return mosaicos
