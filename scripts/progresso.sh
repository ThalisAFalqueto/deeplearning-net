#!/usr/bin/env bash
# Progresso dos treinos, lido dos checkpoints (o log fica bufferizado).
cd "$(dirname "$0")/.."
./.venv/bin/python - <<'PY'
import glob, os, time, torch
agora = time.strftime("%H:%M")
etapas = [("P1 binária", "outputs/p1_binario"), ("P2 pareada", "outputs/p2_pareado")]
etapas += [(f"ablação {p.split('/')[-3]}/{p.split('/')[-2]}", os.path.dirname(p))
           for p in sorted(glob.glob("outputs/ablation/run_2026091[2-9]*/*/seed_*/best.pth"))]
print(f"[{agora}]")
for nome, pasta in etapas:
    arq = os.path.join(pasta, "best.pth")
    if not os.path.exists(arq):
        print(f"  {nome:34s} — não começou"); continue
    try:
        ck = torch.load(arq, map_location="cpu", weights_only=False)
        h = ck.get("history", [])
        total = (ck.get("config") or {}).get("train", {}).get("epochs", "?")
        s = sum(e["seconds"] for e in h) / 60 if h else 0
        print(f"  {nome:34s} época {ck['epoch']:2d}/{total}  IoU {ck.get('best_iou', 0):.4f}  ({s:.0f} min)")
    except Exception as exc:
        print(f"  {nome:34s} — {exc}")
PY
