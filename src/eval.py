"""Avaliação. O "um comando que avalia" exigido pelo enunciado.

    python -m src.eval --config configs/p0_synthetic.yaml

Carrega um checkpoint, decodifica as predições em objetos numerados e reporta as
métricas semânticas (IoU, Dice) lado a lado com as de instância (mAP, erro de contagem).

O contraste entre as duas colunas é o argumento central do trabalho: a máscara binária
pode estar quase perfeita enquanto os objetos estão todos errados.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.synthetic import SyntheticEllipses
from src.decode.connected import labels_from_probability
from src.metrics.instance import count_error, mean_average_precision
from src.metrics.semantic import dice, iou, to_binary
from src.models.unet import UNet


def colorize(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Pinta cada instância de uma cor aleatória, com fundo preto.

    Uma máscara de instâncias em escala de cinza é ilegível: rótulos vizinhos têm
    valores parecidos. Cor aleatória por rótulo é o que torna visível se dois objetos
    encostados foram separados ou fundidos.
    """
    rng = np.random.default_rng(seed)
    palette = np.vstack([[0, 0, 0], rng.random((int(labels.max()) + 1, 3))])
    return palette[labels]


@torch.no_grad()
def run(cfg: dict, checkpoint: Path, out_dir: Path, n_figuras: int = 6):
    """Avalia o checkpoint no conjunto de validação."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]
    threshold = cfg["decode"]["threshold"]

    val = SyntheticEllipses(
        n_samples=d["n_val"], size=d["size"], seed=cfg["seed"] + 777,
        min_obj=d["min_obj"], max_obj=d["max_obj"],
    )
    loader = DataLoader(val, batch_size=cfg["train"]["batch_size"])

    model = UNet(
        in_channels=1, out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base"], depth=cfg["model"]["depth"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    model.eval()

    por_imagem, exemplos = [], []
    for images, gts in loader:
        probs = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
        for prob, gt in zip(probs, gts.numpy()):
            pred_labels = labels_from_probability(prob, threshold)
            m_ap, _ = mean_average_precision(pred_labels, gt)
            por_imagem.append({
                "n_gt": int(len(np.unique(gt)) - 1),
                "n_pred": int(len(np.unique(pred_labels)) - 1),
                "iou": iou(prob > threshold, to_binary(gt)),
                "dice": dice(prob > threshold, to_binary(gt)),
                "map": float(m_ap),
                "count_error": int(count_error(pred_labels, gt)),
            })
            if len(exemplos) < n_figuras:
                exemplos.append((prob, gt, pred_labels))

    resumo = {k: float(np.mean([p[k] for p in por_imagem]))
              for k in ("iou", "dice", "map", "count_error")}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image.json").write_text(json.dumps(por_imagem, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(resumo, indent=2))

    # figura: imagem, gabarito, predição — lado a lado
    fig, axes = plt.subplots(3, len(exemplos), figsize=(2.4 * len(exemplos), 7.4))
    for col, (prob, gt, pred_labels) in enumerate(exemplos):
        axes[0, col].imshow(prob, cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(colorize(gt))
        axes[2, col].imshow(colorize(pred_labels))
        axes[0, col].set_title("probabilidade", fontsize=8)
        axes[1, col].set_title(f"gabarito: {len(np.unique(gt))-1} obj", fontsize=8)
        axes[2, col].set_title(f"predição: {len(np.unique(pred_labels))-1} obj", fontsize=8)
        for row in range(3):
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "predicoes.png", dpi=90)

    return resumo, por_imagem


def main():
    parser = argparse.ArgumentParser(description="Avalia um checkpoint treinado.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="padrão: <output_dir>/best.pth do config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output_dir"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pth"

    resumo, por_imagem = run(cfg, checkpoint, out_dir)

    print(f"\n{len(por_imagem)} imagens de validação\n")
    print("  SEMÂNTICO (a máscara binária)")
    print(f"    IoU                 {resumo['iou']:.4f}")
    print(f"    Dice                {resumo['dice']:.4f}")
    print("\n  INSTÂNCIA (os objetos)")
    print(f"    mAP                 {resumo['map']:.4f}")
    print(f"    erro de contagem    {resumo['count_error']:.2f} objetos por imagem")
    print(f"\n  figura:  {out_dir/'predicoes.png'}")
    print(f"  por imagem: {out_dir/'per_image.json'}")


if __name__ == "__main__":
    main()
