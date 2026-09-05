"""Decodificação da Trilha C: de heatmap + offsets para objetos numerados.

Esta é a peça que resolve o problema da Parte 1. Lá, a única informação disponível era
"este pixel é objeto?", e dois núcleos encostados viravam um blob porque nada na máscara
binária os separava.

Aqui a rede fornece duas informações a mais: **onde estão os centros** (o heatmap) e, para
cada pixel, **um vetor apontando para o centro do seu próprio objeto** (os offsets). Dois
núcleos encostados têm centros distintos, e os pixels de cada um apontam para o seu — a
separação existe nos dados, basta lê-la.

O enunciado descreve o algoritmo: *"decodificação por picos no heatmap e atribuição de cada
pixel ao centro mais próximo do seu ponto deslocado"*.
"""

import numpy as np
import torch
import torch.nn.functional as F


def find_peaks(
    heatmap: torch.Tensor, threshold: float = 0.3, nms_kernel: int = 3
) -> torch.Tensor:
    """Encontra os centros de objeto como máximos locais do heatmap.

    Args:
        heatmap: (H, W) com valores em [0, 1] — a saída da rede após a sigmoide.
        threshold: valor mínimo para um máximo local contar como objeto. Abaixo dele o
            pico é ruído, não núcleo.
        nms_kernel: tamanho da vizinhança para o máximo local (ímpar).

    Returns:
        Tensor (N, 2) com as posições (y, x) dos picos, em ordem determinística.
        Se não houver pico algum, devolva um tensor de shape (0, 2).

    Por que "máximo local" e não "acima do limiar":
        A gaussiana de um núcleo cobre dezenas de pixels acima de qualquer limiar razoável.
        Se você pegasse todos eles, um único núcleo viraria dezenas de objetos. O que
        distingue o centro é ser o **maior da sua vizinhança** — daí o NMS.

    Como fazer:
        `F.max_pool2d(hm, nms_kernel, stride=1, padding=nms_kernel//2)` devolve, em cada
        posição, o maior valor da vizinhança. Um pixel é máximo local exatamente onde esse
        resultado é igual ao próprio heatmap. Combine com o limiar e use `.nonzero()` para
        obter as coordenadas.

        Cuidado: `max_pool2d` espera (B, C, H, W). Use `[None, None]` para adicionar as
        dimensões e `[0, 0]` para removê-las depois.

    Cuidado com platôs:
        Se dois pixels vizinhos tiverem exatamente o mesmo valor máximo, ambos passam no
        teste de igualdade e viram dois objetos. Com heatmap previsto por rede isso é raro
        (os valores são contínuos), mas com alvos sintéticos perfeitos pode acontecer.
    """
    raise NotImplementedError


def assign_to_centers(
    fg_mask: torch.Tensor, offsets: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    """Atribui cada pixel de foreground ao centro mais próximo do seu ponto deslocado.

    Args:
        fg_mask: (H, W) booleano — quais pixels são objeto.
        offsets: (2, H, W) — canal 0 é Δy, canal 1 é Δx.
        centers: (N, 2) com as posições (y, x) dos centros.

    Returns:
        Label map (H, W) de inteiros: 0 = fundo, 1..N = instâncias. O rótulo `i+1`
        corresponde ao centro da linha `i` de `centers`.

    A ideia:
        O offset de um pixel aponta para o centro do objeto dele. Então o **ponto
        deslocado** — a posição do pixel somada ao seu offset — deveria cair em cima desse
        centro. Basta perguntar de qual centro esse ponto ficou mais perto.

            destino = (y + Δy, x + Δx)
            rótulo do pixel = índice do centro mais próximo de `destino`

        Note que a distância é medida do **ponto deslocado** ao centro, não do pixel ao
        centro. Essa diferença é o que permite separar dois núcleos encostados: dois pixels
        vizinhos na fronteira entre eles estão à mesma distância dos dois centros, mas seus
        offsets apontam para lados opostos.

    Como fazer:
        Monte a matriz de pontos deslocados (M, 2) para os M pixels de foreground, e use
        `torch.cdist` contra os N centros — sai uma matriz (M, N) de distâncias. O
        `argmin(dim=1)` dá o índice do centro mais próximo de cada pixel.

        Com 40 núcleos e 16 mil pixels de foreground, a matriz tem 640 mil entradas: cabe
        tranquilamente em memória e é muito mais rápida que um laço.

    Casos de borda:
        - `centers` vazio: nenhum objeto foi detectado; devolva um label map todo zero.
        - `fg_mask` vazio: idem.
    """
    raise NotImplementedError


def decode_center_offset(
    seg_logits: torch.Tensor,
    heatmap_logits: torch.Tensor,
    offsets: torch.Tensor,
    fg_threshold: float = 0.5,
    peak_threshold: float = 0.3,
    nms_kernel: int = 3,
) -> np.ndarray:
    """Decodifica a saída completa da rede em um label map de instâncias.

    Junta as duas funções acima, aplicando as ativações necessárias.

    Args:
        seg_logits: (H, W) — canal 0 da rede, ainda sem sigmoide.
        heatmap_logits: (H, W) — canal 1 da rede, ainda sem sigmoide.
        offsets: (2, H, W) — canais 2-3 da rede, usados como estão.
        fg_threshold: limiar para considerar um pixel como objeto.
        peak_threshold: limiar para um máximo local contar como centro.
        nms_kernel: vizinhança do NMS.

    Returns:
        Array (H, W) de inteiros no mesmo formato que `src/metrics/instance` consome:
        0 = fundo, 1..N = instâncias.

    Passos:
        1. `sigmoid` nos dois primeiros canais — a rede devolve logits, e tanto o limiar de
           foreground quanto o de pico são definidos em probabilidade.
        2. `fg_mask` = probabilidade de foreground acima de `fg_threshold`.
        3. `find_peaks` sobre o heatmap.
        4. `assign_to_centers`.
        5. Converta para numpy antes de devolver.
    """
    raise NotImplementedError
