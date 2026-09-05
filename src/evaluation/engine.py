"""Motor de avaliação: carrega checkpoint, computa métricas e salva figuras."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SyntheticEllipses
from src.utils import labels_from_probability
from src.metrics.instance import MeanAveragePrecision, CountError
from src.metrics.semantic import IoU, Dice
from src.models.unet import UNet
from src.utils import to_binary
from src.evaluation.config import EvalConfig
from src.core.config import AppConfig


class EvalEngine:
    def __init__(self, app_config: AppConfig, checkpoint: Path):
        self.cfg = app_config.get_eval_config()
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def run(self) -> None:
        cfg = self.cfg
        d = cfg.data
        threshold = cfg.decode["threshold"]

        val_ds = self._build_val_dataset()
        loader = DataLoader(val_ds, batch_size=cfg.train["batch_size"])

        model = UNet(
            in_channels=1, out_channels=cfg.model["out_channels"],
            base=cfg.model["base"], depth=cfg.model["depth"],
        ).to(self.device)
        model.load_state_dict(torch.load(self.checkpoint, map_location=self.device, weights_only=False)["model"])
        model.eval()

        iou_metric = IoU()
        dice_metric = Dice()
        map_metric = MeanAveragePrecision()
        count_error_metric = CountError()

        por_imagem, exemplos = [], []
        for images, gts in loader:
            probs = torch.sigmoid(model(images.to(self.device))).cpu()[:, 0]
            for prob, gt in zip(probs, gts):
                prob_np = prob.numpy()
                gt_np = gt.numpy()

                pred_labels_np = labels_from_probability(prob_np, threshold)
                pred_labels = torch.tensor(pred_labels_np)
                gt_tensor = gt.long()

                m_ap, _ = map_metric(pred_labels, gt_tensor)
                binary_gt = to_binary(gt_tensor)
                iou_val = iou_metric(prob > threshold, binary_gt)
                dice_val = dice_metric(prob > threshold, binary_gt)
                count_err = count_error_metric(pred_labels, gt_tensor)

                por_imagem.append({
                    "n_gt": int(len(torch.unique(gt_tensor)) - 1),
                    "n_pred": int(len(torch.unique(pred_labels)) - 1),
                    "iou": iou_val,
                    "dice": dice_val,
                    "map": float(m_ap),
                    "count_error": int(count_err),
                })
                if len(exemplos) < 6:
                    exemplos.append((prob_np, gt_np, pred_labels_np))

        resumo = {k: float(np.mean([p[k] for p in por_imagem]))
                  for k in ("iou", "dice", "map", "count_error")}

        out_dir = cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "per_image.json").write_text(json.dumps(por_imagem, indent=2))
        (out_dir / "summary.json").write_text(json.dumps(resumo, indent=2))

        self._save_figures(exemplos, out_dir)
        self._print_summary(resumo, out_dir, len(por_imagem))

    def _colorize(self, labels: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        palette = np.vstack([[0, 0, 0], rng.random((int(labels.max()) + 1, 3))])
        return palette[labels]

    def _save_figures(self, exemplos, out_dir: Path) -> None:
        fig, axes = plt.subplots(3, len(exemplos), figsize=(2.4 * len(exemplos), 7.4))
        for col, (prob, gt, pred_labels) in enumerate(exemplos):
            axes[0, col].imshow(prob, cmap="gray", vmin=0, vmax=1)
            axes[1, col].imshow(self._colorize(gt))
            axes[2, col].imshow(self._colorize(pred_labels))
            axes[0, col].set_title("probabilidade", fontsize=8)
            axes[1, col].set_title(f"gabarito: {len(np.unique(gt))-1} obj", fontsize=8)
            axes[2, col].set_title(f"predição: {len(np.unique(pred_labels))-1} obj", fontsize=8)
            for row in range(3):
                axes[row, col].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "predicoes.png", dpi=90)

    def _build_val_dataset(self):
        """Constrói o dataset de validação a partir da config."""
        d = self.cfg.data
        if d["kind"] == "synthetic":
            return SyntheticEllipses(
                n_samples=d["n_val"], size=d["size"], seed=self.cfg.seed + 777,
                min_obj=d["min_obj"], max_obj=d["max_obj"],
            )
        raise ValueError(f"data.kind desconhecido: {d['kind']}")

    def _print_summary(self, resumo, out_dir, n_imagens) -> None:
        print(f"\n{n_imagens} imagens de validação\n")
        print("  SEMÂNTICO (a máscara binária)")
        print(f"    IoU                 {resumo['iou']:.4f}")
        print(f"    Dice                {resumo['dice']:.4f}")
        print("\n  INSTÂNCIA (os objetos)")
        print(f"    mAP                 {resumo['map']:.4f}")
        print(f"    erro de contagem    {resumo['count_error']:.2f} objetos por imagem")
        print(f"\n  figura:  {out_dir/'predicoes.png'}")
        print(f"  por imagem: {out_dir/'per_image.json'}")
