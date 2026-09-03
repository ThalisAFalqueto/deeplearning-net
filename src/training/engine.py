"""Motor de treino: loop, validação e checkpoint."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.synthetic import SyntheticEllipses
from src.data.dsb2018 import DSB2018
from src.metrics.semantic import IoU, Dice
from src.models.unet import UNet
from src.utils import to_binary
from src.training.config import TrainConfig


class TrainEngine:
    def __init__(self, config: TrainConfig):
        self.cfg = config  # A engine recebe a configuração carregada na inicialização

        # Detecto o device utilizado (cuda, rocm ou cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fix_seeds(self):
        """Ajusta a seed aleatória para manter os resultados reprodutíveis
        """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def run(self) -> None:
        cfg = self.cfg
        t = cfg.train

        self._fix_seeds()

        train_ds, val_ds = self._build_datasets()
        train_loader = DataLoader(
            train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"]
        )
        val_loader = DataLoader(
            val_ds, batch_size=t["batch_size"], num_workers=t["num_workers"]
        )

        model = UNet(
            in_channels=1,
            out_channels=cfg.model["out_channels"],
            base=cfg.model["base"],
            depth=cfg.model["depth"],
        ).to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])

        n_params = sum(p.numel() for p in model.parameters())
        print(f"dispositivo: {self.device} | parâmetros: {n_params/1e3:.1f}k "
              f"| campo receptivo: {model.receptive_field()} px")
        print(f"treino: {len(train_ds)} imagens | validação: {len(val_ds)} imagens\n")

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        history, best_iou = [], -1.0
        t_start = time.perf_counter()

        for epoch in range(1, t["epochs"] + 1):
            model.train()
            epoch_loss, t_epoch = 0.0, time.perf_counter()

            for images, labels in train_loader:
                images = images.to(self.device)
                target = (labels > 0).float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                loss = criterion(model(images), target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)

            train_loss = epoch_loss / len(train_ds)
            val_loss, val_iou, val_dice = self._evaluate(model, val_loader, criterion)
            dt = time.perf_counter() - t_epoch
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_iou": val_iou, "val_dice": val_dice, "seconds": dt,
            })
            print(f"época {epoch:3d}/{t['epochs']} | treino {train_loss:.4f} "
                  f"| val {val_loss:.4f} | IoU {val_iou:.4f} | Dice {val_dice:.4f} "
                  f"| {dt:.1f}s")

            if val_iou > best_iou:
                best_iou = val_iou
            torch.save(
                {"model": model.state_dict(), "config": {**cfg.__dict__, "output_dir": str(cfg.output_dir)},
                 "epoch": epoch, "val_iou": val_iou},
                cfg.output_dir / "best.pth",
            )

        total = time.perf_counter() - t_start
        (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\ntempo total: {total/60:.1f} min ({total/t['epochs']:.1f}s por época)")
        print(f"melhor IoU de validação: {best_iou:.4f}")
        print(f"checkpoint: {cfg.output_dir/'best.pth'}")

    def _build_datasets(self):
        d = self.cfg.data
        if d["kind"] == "synthetic":
            train = SyntheticEllipses(
                n_samples=d["n_train"], size=d["size"], seed=self.cfg.seed,
                min_obj=d["min_obj"], max_obj=d["max_obj"],
            )
            val = SyntheticEllipses(
                n_samples=d["n_val"], size=d["size"], seed=self.cfg.seed + 777,
                min_obj=d["min_obj"], max_obj=d["max_obj"],
            )
            return train, val
        if d["kind"] == "dsb2018":
            train = DSB2018(data_dir=d["train_dir"], size=d["size"])
            val = DSB2018(data_dir=d["val_dir"], size=d["size"])
            return train, val
        raise ValueError(f"data.kind desconhecido: {d['kind']}")

    @torch.no_grad()
    def _evaluate(self, model, loader, criterion):
        model.eval()
        iou_metric = IoU()
        dice_metric = Dice()

        total_loss, ious, dices = 0.0, [], []
        for images, labels in loader:
            images = images.to(self.device)
            target = (labels > 0).float().unsqueeze(1).to(self.device)

            logits = model(images)
            total_loss += criterion(logits, target).item() * images.size(0)

            prob = torch.sigmoid(logits).cpu()[:, 0]
            for p, g in zip(prob, labels):
                pred_mask = p > self.cfg.decode["threshold"]
                binary_gt = to_binary(g)
                ious.append(iou_metric(pred_mask, binary_gt))
                dices.append(dice_metric(pred_mask, binary_gt))

        n = len(loader.dataset)
        return total_loss / n, float(torch.tensor(ious).mean()), float(torch.tensor(dices).mean())
