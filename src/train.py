"""Loop de treino. O "um comando que treina" exigido pelo enunciado.

    python -m src.train --config configs/p0_synthetic.yaml

A configuração vive num YAML para que treino local (CPU), Colab e Kaggle rodem o mesmo
código sem alteração — muda só o arquivo de config.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.synthetic import SyntheticEllipses
from src.metrics.semantic import IoU, Dice, ToBinary
from src.models.unet import UNet


def set_seed(seed: int):
    """Fixa as fontes de aleatoriedade para o treino ser reproduzível.

    O enunciado pede ablações com 2 seeds e média ± desvio, então isto precisa
    funcionar de verdade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg: dict):
    """Cria os conjuntos de treino e validação a partir da config."""
    d = cfg["data"]
    if d["kind"] == "synthetic":
        train = SyntheticEllipses(
            n_samples=d["n_train"], size=d["size"], seed=cfg["seed"],
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )
        # seed diferente: validação tem que ser um conjunto que o modelo nunca viu
        val = SyntheticEllipses(
            n_samples=d["n_val"], size=d["size"], seed=cfg["seed"] + 777,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )
        return train, val
    raise ValueError(f"data.kind desconhecido: {d['kind']}")


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float):
    """Roda a validação: perda média, IoU e Dice semânticos.

    Só métricas semânticas aqui — as de instância exigem decodificar em objetos, o que
    é trabalho do eval.py. Durante o treino queremos um sinal barato e por época.
    """
    model.eval()
    to_binary = ToBinary()
    iou_metric = IoU()
    dice_metric = Dice()

    total_loss, ious, dices = 0.0, [], []
    for images, labels in loader:
        images = images.to(device)
        target = (labels > 0).float().unsqueeze(1).to(device)

        logits = model(images)
        total_loss += criterion(logits, target).item() * images.size(0)

        prob = torch.sigmoid(logits).cpu()[:, 0]
        for p, g in zip(prob, labels):
            pred_mask = p > threshold
            binary_gt = to_binary(g)
            ious.append(iou_metric(pred_mask, binary_gt))
            dices.append(dice_metric(pred_mask, binary_gt))

    n = len(loader.dataset)
    return total_loss / n, float(torch.tensor(ious).mean()), float(torch.tensor(dices).mean())


def main():
    parser = argparse.ArgumentParser(description="Treina o modelo de segmentação.")
    parser.add_argument("--config", required=True, help="caminho do YAML de configuração")
    parser.add_argument("--device", default=None, help="cpu ou cuda (padrão: automático)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(cfg["seed"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(cfg)
    t = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"]
    )
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], num_workers=t["num_workers"])

    model = UNet(
        in_channels=1,
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base"],
        depth=cfg["model"]["depth"],
    ).to(device)

    # binário: uma saída por pixel. BCEWithLogitsLoss junta sigmoid e binary cross
    # entropy num passo só, o que é numericamente mais estável que fazer separado.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"dispositivo: {device} | parâmetros: {n_params/1e3:.1f}k "
          f"| campo receptivo: {model.receptive_field()} px")
    print(f"treino: {len(train_ds)} imagens | validação: {len(val_ds)} imagens\n")

    history, best_iou = [], -1.0
    t_start = time.perf_counter()

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        epoch_loss, t_epoch = 0.0, time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device)
            target = (labels > 0).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        train_loss = epoch_loss / len(train_ds)
        val_loss, val_iou, val_dice = evaluate(
            model, val_loader, criterion, device, cfg["decode"]["threshold"]
        )
        dt = time.perf_counter() - t_epoch
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "val_iou": val_iou, "val_dice": val_dice, "seconds": dt}
        )
        print(f"época {epoch:3d}/{t['epochs']} | treino {train_loss:.4f} "
              f"| val {val_loss:.4f} | IoU {val_iou:.4f} | Dice {val_dice:.4f} "
              f"| {dt:.1f}s")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {"model": model.state_dict(), "config": cfg,
                 "epoch": epoch, "val_iou": val_iou},
                out_dir / "best.pth",
            )

    total = time.perf_counter() - t_start
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\ntempo total: {total/60:.1f} min ({total/t['epochs']:.1f}s por época)")
    print(f"melhor IoU de validação: {best_iou:.4f}")
    print(f"checkpoint: {out_dir/'best.pth'}")


if __name__ == "__main__":
    main()
