"""Gráfico de mAP em função da densidade de objetos — Parte 1, item 5.

O enunciado pede: *"Quantifiquem o fracasso: gráfico do mAP (ou do erro de contagem) em
função da densidade de objetos na imagem. A tendência tem que ficar visível."*

É o produto central da Parte 1 e o gancho da apresentação. A tese que ele demonstra:

    o IoU semântico quase não se move conforme a imagem fica mais cheia,
    enquanto o mAP de instância desaba.

Isso porque a métrica semântica pergunta "estes pixels são núcleo?" — e a resposta continua
certa quando dois núcleos se fundem. Já a de instância pergunta "quantos núcleos, e quais
pixels de qual?", e aí a fusão é fatal. Quanto mais denso, mais fusão.

A entrada é o ``per_image.json`` que ``EvalEngine`` já salva, com uma linha por imagem
contendo ``n_gt``, ``n_pred``, ``iou``, ``dice``, ``map`` e ``count_error``.
"""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def gt_fusion_rate(labels: np.ndarray) -> float:
    """Fração de objetos do gabarito que dividem componente conexo com outro.

    A densidade que o enunciado sugere (número de objetos por imagem) é uma **proxy
    imperfeita** do que realmente quebra o baseline. Uma imagem pode ter 300 núcleos
    bem espaçados e ir muito bem; o que arruína o método ingênuo é o **toque** entre
    objetos, não a quantidade deles.

    Esta função mede o toque diretamente: conta quantos componentes conexos a máscara
    binária do gabarito tem e compara com quantos objetos existem de fato.

        0,0  todos os objetos separados — o baseline consegue ir bem
        0,5  metade dos objetos está grudada em algum outro

    Medido no baseline da Parte 1 (136 imagens de validação), agrupando por esta taxa:

        fusão 0,0-0,1   IoU 0,833   mAP 0,589
        fusão 0,1-0,3   IoU 0,794   mAP 0,359
        fusão 0,3-0,5   IoU 0,787   mAP 0,198
        fusão 0,5-0,7   IoU 0,781   mAP 0,064

    O IoU cai 6% e o mAP cai 89% ao longo da mesma faixa. As correlações:
    mAP x densidade = -0,240, mAP x fusão = -0,577, IoU x fusão = -0,116.

    Vale plotar os dois gráficos: o de densidade porque é o que o enunciado pede
    literalmente, e o de fusão porque é o que explica o fenômeno.

    Args:
        labels: label map do gabarito (H, W), 0 = fundo.

    Returns:
        Valor entre 0 e 1.
    """
    n_objetos = len(np.unique(labels)) - 1
    if n_objetos == 0:
        return 0.0
    _, n_componentes = ndimage.label(labels > 0)
    return 1.0 - n_componentes / n_objetos


def load_per_image(path: Path) -> list[dict]:
    """Carrega o ``per_image.json`` produzido pela avaliação.

    Args:
        path: caminho do arquivo (normalmente ``outputs/<nome>/per_image.json``).

    Returns:
        Lista de dicionários, um por imagem.
    """
    return json.loads(Path(path).read_text())


def group_by_density(
    registros: list[dict], n_bins: int = 6, mode: str = "width"
) -> list[dict]:
    """Agrupa as imagens por densidade de objetos e resume cada faixa.

    Uma imagem com 3 núcleos e outra com 4 não são casos diferentes o bastante para
    aparecerem separadas: plotar um ponto por imagem vira uma nuvem sem tendência
    legível. Agrupar em faixas de densidade e reportar a média de cada faixa é o que
    torna a tendência visível — que é exatamente o que o enunciado exige.

    Args:
        registros: saída de ``load_per_image``.
        n_bins: quantas faixas de densidade.

    Returns:
        Lista de dicionários ordenada por densidade crescente, um por faixa não vazia,
        com ``densidade`` (centro da faixa), ``map``, ``iou``, ``dice``, ``count_error``
        (médias) e ``n_imagens``.
    """
    if not registros:
        return []

    densidades = np.array([r["n_gt"] for r in registros], dtype=float)
    minimo, maximo = densidades.min(), densidades.max()

    # Todas as imagens com a mesma densidade: uma faixa só, sem dividir por zero.
    if maximo == minimo:
        return [_resumir(registros, minimo)]

    # A distribuição de densidades é muito enviesada (mediana 24, máximo 375 no DSB2018).
    # Com faixas de largura igual, uma acaba com 113 imagens e outra com 2 — e a média
    # de 2 imagens é ruído puro, o que faz a curva subir e descer sem significado.
    # O modo "quantile" usa faixas de tamanho comparável em número de imagens.
    if mode == "quantile":
        ordenados = sorted(registros, key=lambda r: r["n_gt"])
        faixas = []
        for grupo in np.array_split(np.array(ordenados, dtype=object), n_bins):
            if len(grupo):
                grupo = list(grupo)
                faixas.append(_resumir(grupo, np.median([r["n_gt"] for r in grupo])))
        return faixas

    largura = (maximo - minimo) / n_bins
    faixas = []
    for k in range(n_bins):
        inicio = minimo + k * largura
        fim = inicio + largura
        # a última faixa inclui o limite superior, senão a imagem mais densa fica de fora
        if k == n_bins - 1:
            grupo = [r for r in registros if inicio <= r["n_gt"] <= fim]
        else:
            grupo = [r for r in registros if inicio <= r["n_gt"] < fim]

        if grupo:
            faixas.append(_resumir(grupo, (inicio + fim) / 2))

    return faixas


def _resumir(grupo: list[dict], densidade: float) -> dict:
    """Média das métricas de um grupo de imagens."""
    return {
        "densidade": float(densidade),
        "map": float(np.mean([r["map"] for r in grupo])),
        "iou": float(np.mean([r["iou"] for r in grupo])),
        "dice": float(np.mean([r["dice"] for r in grupo])),
        "count_error": float(np.mean([r["count_error"] for r in grupo])),
        "n_imagens": len(grupo),
    }


def plot_density(
    registros: list[dict], output_path: Path, n_bins: int = 6, titulo: str = ""
) -> None:
    """Desenha e salva o gráfico de mAP x densidade.

    Args:
        registros: saída de ``load_per_image``.
        output_path: onde salvar a figura (PNG).
        n_bins: quantas faixas de densidade.
        titulo: título da figura.

    A figura tem dois painéis. No de cima, IoU e mAP na mesma escala 0-1: é o contraste
    entre as duas curvas que constitui o argumento — o IoU mal se move enquanto o mAP cai.
    No de baixo, o erro absoluto de contagem, que o enunciado aceita como alternativa e
    que mostra a tendência de forma ainda mais direta.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faixas = group_by_density(registros, n_bins, mode="quantile")
    x = [f["densidade"] for f in faixas]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # nuvem de pontos ao fundo: mostra a dispersão sem esconder a tendência
    ax_top.scatter([r["n_gt"] for r in registros], [r["iou"] for r in registros],
                   s=12, alpha=0.25, color="tab:blue")
    ax_top.scatter([r["n_gt"] for r in registros], [r["map"] for r in registros],
                   s=12, alpha=0.25, color="tab:red")

    ax_top.plot(x, [f["iou"] for f in faixas], "o-", color="tab:blue", lw=2.5,
                label="IoU semântico (a máscara binária)")
    ax_top.plot(x, [f["map"] for f in faixas], "s-", color="tab:red", lw=2.5,
                label="mAP de instância (os objetos)")

    ax_top.set_ylabel("métrica")
    ax_top.set_ylim(0, 1)
    ax_top.legend(loc="lower left", fontsize=9)
    ax_top.grid(alpha=0.3)
    if titulo:
        ax_top.set_title(titulo)

    ax_bot.plot(x, [f["count_error"] for f in faixas], "^-", color="tab:orange", lw=2.5)
    ax_bot.scatter([r["n_gt"] for r in registros], [r["count_error"] for r in registros],
                   s=12, alpha=0.25, color="tab:orange")
    ax_bot.set_xlabel("densidade — número de objetos por imagem")
    ax_bot.set_ylabel("erro absoluto\nde contagem")
    ax_bot.set_yscale("symlog")     # o erro cresce ordens de grandeza entre as faixas
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=110)
    plt.close(fig)


def plot_fusion(
    registros: list[dict], output_path: Path,
    n_bins: int = 5, titulo: str = "", fusao: list[float] | None = None
) -> None:
    """Mesmo gráfico, mas contra a taxa de fusão do gabarito em vez da densidade.

    Ver ``gt_fusion_rate``: a densidade é o que o enunciado pede, mas o toque entre
    objetos é o que de fato explica a queda do mAP (correlação -0,577 contra -0,240).

    Args:
        registros: saída de ``load_per_image``.
        output_path: onde salvar a figura.
        n_bins: quantas faixas.
        titulo: título da figura.
        fusao: taxa de fusão por imagem. Se None, lê o campo ``gt_fusion`` que a
            avaliação grava em cada registro.
    """
    if fusao is None:
        fusao = [r["gt_fusion"] for r in registros]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reaproveita o agrupamento trocando a variável do eixo x
    com_fusao = [dict(r, n_gt=f) for r, f in zip(registros, fusao)]
    faixas = group_by_density(com_fusao, n_bins, mode="quantile")
    x = [f["densidade"] for f in faixas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fusao, [r["iou"] for r in registros], s=12, alpha=0.25, color="tab:blue")
    ax.scatter(fusao, [r["map"] for r in registros], s=12, alpha=0.25, color="tab:red")
    ax.plot(x, [f["iou"] for f in faixas], "o-", color="tab:blue", lw=2.5,
            label="IoU semântico")
    ax.plot(x, [f["map"] for f in faixas], "s-", color="tab:red", lw=2.5,
            label="mAP de instância")

    ax.set_xlabel("fração de objetos do gabarito que se tocam")
    ax.set_ylabel("métrica")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    if titulo:
        ax.set_title(titulo)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=110)
    plt.close(fig)
