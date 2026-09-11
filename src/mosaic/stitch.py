"""Inferência em tiles e as formas de costurar o resultado.

Dois tipos de saída passam por aqui, e a diferença entre eles é a Parte 4 inteira:

**Mapas densos** — probabilidade de foreground, heatmap de centros, offsets. São grandezas
por pixel que não dependem de qual tile as calculou: o mesmo pixel, com o mesmo contexto,
recebe o mesmo valor. Por isso dá para fazer a média entre tiles (``media_densa``), como o
slide 83 manda.

**Rótulos de instância** — o número de cada objeto é arbitrário. O objeto 7 do tile da
esquerda não tem relação nenhuma com o objeto 7 do tile da direita, e a média de 7 com 12 não
significa nada. Costurar rótulos decodificados em cada tile (``costura_ingenua``) parte em
dois todo objeto que cruza a divisa entre tiles. ``fusao_por_iou`` conserta isso depois;
``media_densa`` evita o problema, porque decodifica uma vez só, no mosaico inteiro.

Formato das saídas: a rede binária devolve um tensor; a da Trilha C, uma tupla
(seg, heatmap, offsets). Aqui tudo vira um tensor só, com os canais concatenados
(``tamanhos`` guarda como separar de volta), para que a média funcione igual para as duas.
"""

import numpy as np
import torch

from src.metrics.instance import IoUMatrix


def _concatenar(saida) -> tuple[torch.Tensor, list[int] | None]:
    """Saída da rede (tensor ou tupla de tensores (B, c, H, W)) → (B, C, H, W) e tamanhos."""
    if isinstance(saida, tuple):
        return torch.cat(saida, dim=1), [s.shape[1] for s in saida]
    return saida, None


def separar(denso: torch.Tensor, tamanhos: list[int] | None):
    """(C, H, W) → o formato de UMA imagem que ``model.decode`` espera.

    Trilha C: tupla ((1,H,W), (1,H,W), (2,H,W)). Binária: o próprio tensor (1, H, W).
    """
    if tamanhos is None:
        return denso
    return tuple(torch.split(denso, tamanhos, dim=0))


@torch.no_grad()
def inferir_inteira(model, imagem: torch.Tensor) -> tuple[torch.Tensor, list[int] | None]:
    """Uma passada na imagem inteira. A rede é totalmente convolucional, então aceita
    qualquer tamanho — é a referência que os tiles deveriam reproduzir.

    Returns:
        ``denso`` (C, H, W) na CPU e ``tamanhos``.
    """
    dispositivo = next(model.parameters()).device
    denso, tamanhos = _concatenar(model(imagem[None].to(dispositivo)))
    return denso[0].cpu(), tamanhos


@torch.no_grad()
def inferir_tiles(
    model, imagem: torch.Tensor, grade: list[tuple[int, int]], tile: int, lote: int = 8
) -> tuple[torch.Tensor, list[int] | None]:
    """Roda a rede em cada tile, separadamente.

    Em modo eval o BatchNorm usa as estatísticas guardadas, então juntar tiles num lote não
    muda o resultado de nenhum — é só mais rápido.

    Returns:
        ``saidas`` (T, C, tile, tile) na CPU, na ordem de ``grade``, e ``tamanhos``.
    """
    dispositivo = next(model.parameters()).device
    recortes = torch.stack([imagem[:, y:y + tile, x:x + tile] for y, x in grade])
    partes, tamanhos = [], None
    for i in range(0, len(recortes), lote):
        saida, tamanhos = _concatenar(model(recortes[i:i + lote].to(dispositivo)))
        partes.append(saida.cpu())
    return torch.cat(partes), tamanhos


def costura_densa(
    saidas: torch.Tensor, grade: list[tuple[int, int]], donos: np.ndarray, tile: int
) -> torch.Tensor:
    """Mapa denso do mosaico usando só a parte interna de cada tile, sem média.

    Cada pixel vem do seu tile dono (``mapa_de_donos``).

    Returns:
        (C, H, W).
    """
    altura, largura = donos.shape
    denso = torch.zeros(saidas.shape[1], altura, largura)
    for k, (y, x) in enumerate(grade):
        meu = torch.from_numpy(donos[y:y + tile, x:x + tile] == k)
        janela = denso[:, y:y + tile, x:x + tile]
        janela[:, meu] = saidas[k][:, meu]
    return denso


def media_densa(
    saidas: torch.Tensor,
    grade: list[tuple[int, int]],
    altura: int,
    largura: int,
    peso: np.ndarray | None = None,
) -> torch.Tensor:
    """Média, pixel a pixel, das saídas de todos os tiles que cobrem cada pixel.

    A média é feita sobre as saídas cruas da rede (logits e offsets em pixels) — é o que a
    decodificação consome depois.

    Args:
        peso: (tile, tile). ``None`` = média simples (todos os tiles pesam igual);
            ``peso_interior(...)`` = quem vê o pixel mais para o meio pesa mais.

    Returns:
        (C, H, W).
    """
    tile = saidas.shape[-1]
    w = torch.ones(tile, tile) if peso is None else torch.as_tensor(peso, dtype=torch.float32)
    soma = torch.zeros(saidas.shape[1], altura, largura)
    soma_pesos = torch.zeros(altura, largura)
    for k, (y, x) in enumerate(grade):
        soma[:, y:y + tile, x:x + tile] += saidas[k] * w
        soma_pesos[y:y + tile, x:x + tile] += w
    return soma / soma_pesos


def _globalizar(rotulos_tiles: list[np.ndarray]) -> list[np.ndarray]:
    """Desloca os rótulos de cada tile para que nenhum número se repita entre tiles."""
    globais, proximo = [], 0
    for rot in rotulos_tiles:
        rot = np.asarray(rot, dtype=np.int64)
        globais.append(np.where(rot > 0, rot + proximo, 0))
        proximo += int(rot.max()) if rot.size else 0
    return globais


def costura_ingenua(
    rotulos_tiles: list[np.ndarray], grade: list[tuple[int, int]], donos: np.ndarray, tile: int
) -> np.ndarray:
    """Cada tile decodificado sozinho; cada pixel recebe o rótulo do seu tile dono.

    É a "parte interna" do slide aplicada a rótulos. Um objeto que cruza a divisa entre dois
    donos recebe um número de cada lado — vira dois objetos.

    Returns:
        Label map (H, W) do mosaico, 0 = fundo.
    """
    globais = _globalizar(rotulos_tiles)
    saida = np.zeros(donos.shape, dtype=np.int64)
    for k, (y, x) in enumerate(grade):
        meu = donos[y:y + tile, x:x + tile] == k
        saida[y:y + tile, x:x + tile][meu] = globais[k][meu]
    return saida


def _raiz(pai: dict, a: int) -> int:
    while pai.get(a, a) != a:
        pai[a] = pai.get(pai[a], pai[a])     # encurta o caminho
        a = pai[a]
    return a


def fusao_por_iou(
    rotulos_tiles: list[np.ndarray],
    grade: list[tuple[int, int]],
    donos: np.ndarray,
    tile: int,
    limiar: float = 0.5,
) -> np.ndarray:
    """Costura ingênua + fusão dos pedaços que dois tiles vizinhos concordam ser o mesmo objeto.

    Na faixa em que dois tiles se sobrepõem, os dois viram exatamente os mesmos pixels. Se o
    objeto ``a`` do tile da esquerda e o objeto ``b`` do tile da direita ocupam quase a mesma
    região **dentro dessa faixa** (IoU ≥ ``limiar``), são o mesmo núcleo, e os pedaços que a
    costura ingênua separou recebem um número só.

    Medir o IoU só dentro da faixa é o que faz funcionar para objetos cortados: o tile da
    esquerda pode ver só metade do núcleo, mas a metade que está na faixa ele vê inteira.

    As uniões são transitivas (union-find): um núcleo no cruzamento de quatro tiles junta os
    quatro pedaços.

    Returns:
        Label map (H, W) do mosaico, 0 = fundo.
    """
    globais = _globalizar(rotulos_tiles)
    iou_matrix = IoUMatrix()
    pai: dict[int, int] = {}

    for k, (yk, xk) in enumerate(grade):
        for l in range(k + 1, len(grade)):
            yl, xl = grade[l]
            y0, y1 = max(yk, yl), min(yk, yl) + tile
            x0, x1 = max(xk, xl), min(xk, xl) + tile
            if y0 >= y1 or x0 >= x1:
                continue                       # tiles que não se sobrepõem
            a = globais[k][y0 - yk:y1 - yk, x0 - xk:x1 - xk]
            b = globais[l][y0 - yl:y1 - yl, x0 - xl:x1 - xl]
            ids_a, ids_b = np.unique(a[a > 0]), np.unique(b[b > 0])
            if len(ids_a) == 0 or len(ids_b) == 0:
                continue
            iou = iou_matrix(a, b).numpy()     # linhas = ids_a, colunas = ids_b, em ordem
            for i, j in zip(*np.nonzero(iou >= limiar)):
                ra, rb = _raiz(pai, int(ids_a[i])), _raiz(pai, int(ids_b[j]))
                if ra != rb:
                    pai[max(ra, rb)] = min(ra, rb)

    saida = costura_ingenua(rotulos_tiles, grade, donos, tile)
    if pai:
        tabela = np.arange(int(saida.max()) + 1)
        for rotulo in pai:
            if rotulo < len(tabela):
                tabela[rotulo] = _raiz(pai, rotulo)
        saida = tabela[saida]
    return saida
