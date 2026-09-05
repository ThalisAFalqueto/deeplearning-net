"""Motor de treino: loop, validação e checkpoint."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.metrics.semantic import IoU, Dice
from src.models.unet import UNet
from src.utils import to_binary
from src.data import DataPipeline
from src.core.config import AppConfig


class TrainEngine:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.cfg = app_config.get_train_config()

        # Detecto o device utilizado (cuda, rocm ou cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fixed_seeds(self):
        """Ajusta a seed aleatória para manter os resultados reprodutíveis
        """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def run(self) -> None:
        cfg = self.cfg
        t = cfg.train

        self._fixed_seeds()

        # Crio a pipeline que gera os dataloaders
        data_pipeline = DataPipeline(self.app_config)

        # Carrego os loaders via pipeline (treino e validação)
        train_loader, val_loader = data_pipeline.build_dataloaders()

        # Instancio o modelo que vai ser utilizado
        model = UNet(
            in_channels=1,
            out_channels=cfg.model["out_channels"],
            base=cfg.model["base"],
            depth=cfg.model["depth"],
        ).to(self.device)

        # Instancio o critério de otimização (função de perca)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Instancio o método de otimização (Adam, SGD, etc)
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])

        # Conto o número de parâmetros do modelo e imprimo algumas informações
        n_params = sum(p.numel() for p in model.parameters())
        print(f"dispositivo: {self.device} | parâmetros: {n_params/1e3:.1f}k "
              f"| campo receptivo: {model.receptive_field()} px")
        print(f"treino: {len(train_loader.dataset)} imagens | validação: {len(val_loader.dataset)} imagens\n")

        # =================== LOOP DE TREINO ===================
        history, best_iou = [], -1.0  # Inicio o histórico de treino e o melhor IoU de validação
        t_start = time.perf_counter()  # Inicio a contagem do tempo total de treino

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

            train_loss = epoch_loss / len(train_loader.dataset)
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
