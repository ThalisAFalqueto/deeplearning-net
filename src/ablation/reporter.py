"""Utilitários de visualização para ablação."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_ablation_bars(resultados: dict, metric: str, output_path: Path):
    """Gera gráfico de barras comparando configs para uma métrica."""
    nomes = list(resultados.keys())
    means = [resultados[n][metric]["mean"] for n in nomes]
    stds = [resultados[n][metric]["std"] for n in nomes]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(len(nomes) * 1.5, 4))
    ax.bar(nomes, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel(metric)
    ax.set_title(f"Ablação — {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_ablation_table(resultados: dict, output_path: Path):
    """Gera tabela visual (matplotlib table) com todas as métricas."""
    metricas = ["iou", "dice", "map", "count_error"]
    nomes = list(resultados.keys())

    celulas = []
    for metrica in metricas:
        linha = [metrica]
        for nome in nomes:
            if metrica in resultados[nome]:
                m = resultados[nome][metrica]
                linha.append(f"{m['mean']:.4f} ± {m['std']:.4f}")
            else:
                linha.append("-")
        celulas.append(linha)

    fig, ax = plt.subplots(figsize=(len(nomes) * 2.5, len(metricas) * 0.6 + 1))
    ax.axis("off")
    tabela = ax.table(
        cellText=celulas,
        colLabels=["Métrica"] + nomes,
        loc="center",
        cellLoc="center",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
