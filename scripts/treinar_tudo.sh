#!/usr/bin/env bash
# Parte 1 binária, Parte 2 pareada e a ablação do Eixo 3 — 30 épocas cada, em série.
# Em série de propósito: a máquina tem 8 núcleos físicos e o torch já usa os 8;
# rodar dois treinos ao mesmo tempo deixa os dois ~2x mais lentos.
set -o pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python          # -u = sem buffer, para as linhas aparecerem na hora

echo "=== [1/3] Parte 1 binária — 30 épocas — $(date +%H:%M) ==="
$PY -u -m main --config configs/p1_binario_dsb2018.yaml || exit 1

echo "=== [2/3] Parte 2 pareada — 30 épocas — $(date +%H:%M) ==="
$PY -u -m main --config configs/p2_dsb2018.yaml || exit 1

echo "=== [3/3] Ablação do Eixo 3: baseline x unet_ppm, seeds 0 e 42 — $(date +%H:%M) ==="
$PY -u -m main --ablation configs/baseline_pareado.yaml configs/unet_ppm_dsb2018.yaml \
    --ablation-seeds 0 42 || exit 1

echo "=== TUDO PRONTO $(date +%H:%M) ==="
