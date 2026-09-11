"""Grade de tiles sobrepostos, pesos de interior e o tile "dono" de cada pixel.

O slide 83 da aula descreve a prática: imagem grande em tiles, tiles sobrepostos,
"consider the inner part" e "average the results". A borda de um tile é onde a rede tem
menos contexto — metade da vizinhança de um pixel da borda está fora do recorte. As duas
receitas do slide atacam isso de formas diferentes:

    parte interna   cada pixel vem só do tile em que ele está mais para o meio
                    (``mapa_de_donos``) — a divisa fica no meio da faixa de sobreposição;
    média           cada pixel é a média das previsões de todos os tiles que o cobrem,
                    opcionalmente com peso maior para quem o vê mais para o meio
                    (``peso_interior``).

Exemplo com imagem 1024, tile 256 e passo 192: tiles começam em 0, 192, 384, 576, 768;
cada par vizinho se sobrepõe em 64 px, e as divisas entre donos caem em 224, 416, 608, 800.
"""

import numpy as np


def posicoes(comprimento: int, tile: int, passo: int) -> list[int]:
    """Onde os tiles começam ao longo de um eixo.

    O último tile é encostado na borda (``comprimento - tile``) mesmo que isso deixe o
    último passo menor — sem isso, a faixa final da imagem ficaria sem cobertura.
    """
    if comprimento < tile:
        raise ValueError(f"imagem ({comprimento}) menor que o tile ({tile})")
    inicios = list(range(0, comprimento - tile + 1, passo))
    if inicios[-1] != comprimento - tile:
        inicios.append(comprimento - tile)
    return inicios


def grade_tiles(altura: int, largura: int, tile: int, passo: int) -> list[tuple[int, int]]:
    """Canto superior esquerdo ``(y, x)`` de cada tile, em ordem de leitura (linha a linha)."""
    return [(y, x) for y in posicoes(altura, tile, passo) for x in posicoes(largura, tile, passo)]


def _dono_no_eixo(comprimento: int, inicios: list[int], tile: int) -> np.ndarray:
    """Para cada coordenada, o índice do tile cujo centro está mais perto."""
    centros = np.asarray(inicios, dtype=np.float64) + tile / 2
    pixels = np.arange(comprimento) + 0.5          # centro geométrico do pixel
    return np.abs(pixels[:, None] - centros[None, :]).argmin(axis=1)


def mapa_de_donos(altura: int, largura: int, tile: int, passo: int) -> np.ndarray:
    """Índice (na ordem de ``grade_tiles``) do tile em que cada pixel está mais para o meio.

    É a "parte interna" do slide: os donos particionam a imagem, e a divisa entre dois tiles
    vizinhos cai no meio da faixa em que eles se sobrepõem.

    Returns:
        Array (altura, largura) de inteiros.
    """
    ys = posicoes(altura, tile, passo)
    xs = posicoes(largura, tile, passo)
    dono_y = _dono_no_eixo(altura, ys, tile)
    dono_x = _dono_no_eixo(largura, xs, tile)
    return dono_y[:, None] * len(xs) + dono_x[None, :]


def peso_interior(tile: int, rampa: int) -> np.ndarray:
    """Peso de cada pixel do tile na média: sobe da borda até o interior.

    Em cada eixo, o peso cresce linearmente com a distância à borda mais próxima e satura em
    1 a ``rampa`` pixels dela; o peso 2D é o produto dos dois eixos. Nunca é zero, então a
    média continua definida na borda da imagem, onde só um tile cobre o pixel.

    Usar ``rampa`` = largura da sobreposição faz o tile vizinho assumir gradualmente ao longo
    da faixa: no meio dela os dois pesam igual.
    """
    i = np.arange(tile)
    distancia = np.minimum(i + 1, tile - i)          # 1 na borda, tile/2 no meio
    eixo = np.clip(distancia / rampa, 0.0, 1.0)
    return np.outer(eixo, eixo).astype(np.float32)
