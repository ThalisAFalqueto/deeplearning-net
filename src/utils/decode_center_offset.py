"""Decodificação da Trilha C: de heatmap + offsets para objetos numerados.

Esta é a peça que resolve o problema da Parte 1. Lá, a única informação disponível era
"este pixel é objeto?", e dois núcleos encostados viravam um blob porque nada na máscara
binária os separava.

Aqui a rede fornece duas informações a mais: **onde estão os centros** (o heatmap) e, para
cada pixel, **um vetor apontando para o centro do seu próprio objeto** (os offsets). Dois
núcleos encostados têm centros distintos, e os pixels de cada um apontam para o seu — a
separação existe nos dados, basta lê-la.

Num par de objetos colados, dois pixels vizinhos da fronteira se comportam assim:

    pixel (3,3):  Δx = -1  ->  ponto deslocado cai sobre o centro da ESQUERDA
    pixel (3,4):  Δx = +1  ->  ponto deslocado cai sobre o centro da DIREITA

É essa divergência que a atribuição lê. O enunciado descreve o algoritmo: *"decodificação
por picos no heatmap e atribuição de cada pixel ao centro mais próximo do seu ponto
deslocado"*.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def find_peaks(
    heatmap: torch.Tensor, threshold: float = 0.3, nms_kernel: int = 3
) -> torch.Tensor:
    """Encontra os centros de objeto como máximos locais do heatmap.

    Args:
        heatmap: (H, W) com valores em [0, 1] — a saída da rede após a sigmoide.
        threshold: valor mínimo para um máximo local contar como objeto.
        nms_kernel: tamanho da vizinhança para o máximo local (ímpar).

    Returns:
        Tensor float (N, 2) com as posições (y, x) dos centros. Vazio: shape (0, 2).
    """
    hm = heatmap.detach().float()

    # Um pixel é máximo local onde é igual ao maior valor da sua vizinhança. Sem isso,
    # "acima do limiar" pegaria dezenas de pixels por objeto — a gaussiana inteira.
    vizinhanca = F.max_pool2d(
        hm[None, None], nms_kernel, stride=1, padding=nms_kernel // 2
    )[0, 0]
    picos = (hm == vizinhanca) & (hm > threshold)

    if not picos.any():
        return torch.zeros((0, 2), dtype=torch.float32)

    # Platôs: quando o centro do objeto cai entre dois pixels (centroide fracionário,
    # o caso comum), os dois empatam no máximo e ambos passam no teste acima — um
    # objeto viraria dois. Agrupar os picos adjacentes e usar o centroide de cada grupo
    # resolve, e de quebra devolve a posição sub-pixel do centro.
    conectividade_8 = np.ones((3, 3), dtype=int)
    grupos, n = ndimage.label(picos.numpy(), structure=conectividade_8)
    centros = ndimage.center_of_mass(picos.numpy(), grupos, range(1, n + 1))

    return torch.tensor(centros, dtype=torch.float32).reshape(-1, 2)


def assign_to_centers(
    fg_mask: torch.Tensor, offsets: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    """Atribui cada pixel de foreground ao centro mais próximo do seu ponto deslocado.

    Args:
        fg_mask: (H, W) booleano — quais pixels são objeto.
        offsets: (2, H, W) — canal 0 é Δy, canal 1 é Δx.
        centers: (N, 2) com as posições (y, x) dos centros.

    Returns:
        Label map (H, W) de inteiros: 0 = fundo, 1..N contíguos.
    """
    fg_mask = fg_mask.bool()
    labels = torch.zeros(fg_mask.shape, dtype=torch.long)

    if len(centers) == 0 or not fg_mask.any():
        return labels

    # coordenadas (y, x) dos pixels de objeto, em ordem row-major
    coords = fg_mask.nonzero().float()
    # os offsets são indexados pela mesma máscara, então saem na mesma ordem
    deslocamento = torch.stack(
        [offsets[0][fg_mask].float(), offsets[1][fg_mask].float()], dim=1
    )

    # O ponto deslocado é onde o pixel "acha" que está o centro dele. A distância é
    # medida DAQUI até os centros — não da posição original do pixel. É justamente essa
    # troca que separa dois objetos encostados: pixels vizinhos da fronteira têm offsets
    # opostos e viajam para centros diferentes.
    destino = coords + deslocamento

    # (M pixels, N centros) de uma vez; com 16 mil pixels e 40 centros são 640 mil
    # distâncias, muito mais rápido que um laço
    distancias = torch.cdist(destino, centers.float())
    labels[fg_mask] = distancias.argmin(dim=1) + 1   # +1 porque 0 é o fundo

    # Um centro pode não receber pixel nenhum, o que deixaria buracos na numeração.
    # A métrica aceita rótulos esparsos, mas contíguo é o contrato do resto do código.
    presentes = torch.unique(labels[labels != 0])
    if len(presentes) and int(presentes.max()) != len(presentes):
        renumerado = torch.zeros_like(labels)
        for novo, antigo in enumerate(presentes.tolist(), start=1):
            renumerado[labels == antigo] = novo
        labels = renumerado

    return labels


def decode_center_offset(
    seg_logits: torch.Tensor,
    heatmap_logits: torch.Tensor,
    offsets: torch.Tensor,
    fg_threshold: float = 0.5,
    peak_threshold: float = 0.3,
    nms_kernel: int = 3,
) -> np.ndarray:
    """Decodifica a saída completa da rede em um label map de instâncias.

    Args:
        seg_logits: (H, W) — canal 0 da rede, sem sigmoide.
        heatmap_logits: (H, W) — canal 1 da rede, sem sigmoide.
        offsets: (2, H, W) — canais 2-3 da rede.
        fg_threshold: limiar de foreground.
        peak_threshold: altura mínima de um máximo local para contar como centro.
        nms_kernel: vizinhança do NMS.

    Returns:
        Array (H, W) de inteiros: 0 = fundo, 1..N = instâncias. Mesmo formato que
        `src/metrics/instance` consome.
    """
    # a rede devolve logits; os dois limiares são definidos em probabilidade
    fg_mask = torch.sigmoid(seg_logits.detach()) > fg_threshold
    heatmap = torch.sigmoid(heatmap_logits.detach())

    centros = find_peaks(heatmap, peak_threshold, nms_kernel)
    labels = assign_to_centers(fg_mask, offsets.detach(), centros)
    return labels.numpy()
