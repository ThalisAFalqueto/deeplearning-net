"""Perda combinada da Trilha C: segmentação + heatmap + offsets.

A rede prevê três coisas de naturezas diferentes, e cada uma pede a sua perda — as três
citadas nos slides:

    canal 0     foreground      cross-entropy binária    (slides 73-79)
    canal 1     heatmap         L2                       (slide 80)
    canais 2-3  offsets         L1                       (slide 80)

Duas escolhas neste arquivo merecem explicação, porque são o que separa um treino que
converge de um que não sai do lugar. Ambas estão comentadas nos métodos correspondentes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterOffsetLoss:
    """Soma ponderada das três perdas da Trilha C.

    Args:
        w_seg: peso da segmentação.
        w_heatmap: peso do heatmap.
        w_offset: peso dos offsets.
        pos_weight: reforço da penalidade nos pixels próximos aos centros. Ver
            :meth:`heatmap_loss`. Com 0, a perda do heatmap é L2 pura.
    """

    def __init__(
        self,
        w_seg: float = 1.0,
        w_heatmap: float = 1.0,
        w_offset: float = 1.0,
        pos_weight: float = 0.0,
    ):
        self.w_seg = w_seg
        self.w_heatmap = w_heatmap
        self.w_offset = w_offset
        self.pos_weight = pos_weight
        self._bce = nn.BCEWithLogitsLoss()

    def seg_loss(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        """Cross-entropy binária sobre o canal de foreground."""
        return self._bce(logits, alvo)

    def heatmap_loss(self, pred: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        """L2 ponderada sobre o heatmap de centros.

        O pico exato de um núcleo é **um pixel em 65.536** — 0,06% da imagem. Sob L2 pura,
        prever zero em toda a imagem já é uma solução quase ótima: o erro médio fica
        minúsculo e a rede não tem incentivo para aprender centro nenhum. É o mesmo
        desbalanceamento que o Eixo 2 da Parte 3 estuda, e o enunciado o menciona
        explicitamente ("os pixels de centro são uma fração minúscula da imagem").

        A correção é dar mais peso onde o alvo é alto:

            w = 1 + pos_weight · alvo
            loss = média(w · (pred - alvo)²)

        Como o alvo é a própria gaussiana, o peso decai suavemente do centro para a borda —
        não há um corte artificial entre "pixel de centro" e "pixel de fundo".

        Com ``pos_weight = 0`` isto vira L2 pura, o que permite medir o efeito do
        desbalanceamento na ablação.
        """
        peso = 1.0 + self.pos_weight * alvo
        return (peso * (pred - alvo) ** 2).mean()

    def offset_loss(
        self, pred: torch.Tensor, alvo: torch.Tensor, fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 sobre os offsets, restrita ao foreground.

        Um pixel de fundo não pertence a objeto nenhum — não existe centro para ele apontar,
        e o alvo ali é zero por convenção, não porque seja a resposta certa. Incluir o fundo
        faria a rede gastar capacidade aprendendo a prever zero em 75% da imagem, e diluiria
        o gradiente dos pixels que realmente importam pelo mesmo fator.

        A média é sobre os pixels de foreground, não sobre a imagem toda: assim o valor da
        perda não muda de escala quando a imagem tem mais ou menos objetos.
        """
        # fg_mask é (B, 1, H, W) e pred é (B, 2, H, W): o broadcast aplica a mesma máscara
        # aos dois canais de offset
        erro = (pred - alvo).abs() * fg_mask
        n = fg_mask.sum() * pred.shape[1]
        if n == 0:
            return pred.sum() * 0.0    # imagem sem foreground: zero, mas mantém o grafo
        return erro.sum() / n

    def __call__(self, logits: torch.Tensor, alvos: dict) -> tuple[torch.Tensor, dict]:
        """Calcula a perda total e as componentes.

        Args:
            logits: saída bruta da rede, (B, 4, H, W). Canal 0 é logit de foreground,
                canal 1 é o heatmap, canais 2-3 são os offsets.
            alvos: dicionário de :func:`src.data.targets.batch_center_offset_targets`.

        Returns:
            Tupla (perda total, dicionário com as três componentes já ponderadas). O
            dicionário vai para o log por época — é como se percebe cedo que uma das
            perdas está dominando ou estagnada.
        """
        seg = self.seg_loss(logits[:, 0:1], alvos["foreground"])

        # sigmoid no heatmap: o alvo é uma gaussiana em [0, 1], então a saída precisa
        # estar no mesmo intervalo. Sem isso a rede teria que aprender a se limitar sozinha.
        heatmap = self.heatmap_loss(torch.sigmoid(logits[:, 1:2]), alvos["heatmap"])

        offset = self.offset_loss(logits[:, 2:4], alvos["offsets"], alvos["foreground"])

        total = self.w_seg * seg + self.w_heatmap * heatmap + self.w_offset * offset
        return total, {
            "seg": float(seg.detach()),
            "heatmap": float(heatmap.detach()),
            "offset": float(offset.detach()),
        }
