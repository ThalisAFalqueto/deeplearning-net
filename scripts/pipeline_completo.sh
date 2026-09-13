#!/usr/bin/env bash
# Roda o PA1 inteiro, do zero: Partes 0 a 6, em sequência, num comando só.
#
# Não usa --resume em nenhuma etapa: cada treino começa do zero e sobrescreve os artefatos
# do seu próprio output_dir (best.pth, last.pth, history.json, summary.json, figuras), mesmo
# que já exista algo lá — inclusive os checkpoints versionados no git. Se quiser manter os
# resultados atuais, faça `git stash` (ou copie outputs/ para outro lugar) antes de rodar.
#
# Tempo esperado: ~480 épocas de treino somadas (Parte 1 + Parte 2 final + Parte 2 pareada +
# 2 eixos de ablação × 2 configs × 2 seeds), fora as etapas só-de-avaliação (mosaico, falhas,
# correção, estresse). Pela média já medida neste projeto (~55-100s/época em GPU), a soma fica
# entre 8 e 13 horas; em CPU pode ser bem mais lento. Rode em background, ex.:
#   nohup bash scripts/pipeline_completo.sh > pipeline.log 2>&1 &
# e acompanhe com `tail -f pipeline.log` ou `scripts/progresso.sh`.
set -o pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python          # -u = sem buffer, para as linhas aparecerem na hora
CKPT=outputs/p2/best.pth       # regerado no passo 4; é o que a Parte 4/5/6 usam

etapa() { echo "=== [$1/11] $2 — $(date '+%d/%m %H:%M') ==="; }

etapa 1 "testes unitários (Parte 0)"
$PY -m pytest tests/ -v || exit 1

etapa 2 "Parte 0 — treino + avaliação sintética"
$PY -u -m main --synthetic || exit 1

etapa 3 "Parte 1 — baseline binária no DSB2018"
$PY -u -m main --config configs/p1_binario_dsb2018.yaml || exit 1

etapa 4 "Parte 2 — modelo final (Trilha C, 100 épocas)"
$PY -u -m main --config configs/p2_final.yaml || exit 1

etapa 5 "Parte 2 — regime pareado com a Parte 1"
$PY -u -m main --config configs/p2_dsb2018.yaml || exit 1

etapa 6 "Parte 3 — Eixo 1: U-Net x SegNet"
$PY -u -m main --ablation configs/default.yaml configs/segnet_dsb2018.yaml \
    --ablation-seeds 0 42 || exit 1

etapa 7 "Parte 3 — Eixo 3: baseline x U-Net+PPM"
$PY -u -m main --ablation configs/baseline_pareado.yaml configs/unet_ppm_dsb2018.yaml \
    --ablation-seeds 0 42 || exit 1

etapa 8 "Parte 4 — inferência em mosaico"
$PY -u -m main --mode mosaic --config configs/p2_dsb2018.yaml --checkpoint "$CKPT" || exit 1

etapa 9 "Parte 5 — galeria de falhas"
$PY -u -m main --mode fails --config configs/p2_dsb2018.yaml --checkpoint "$CKPT" || exit 1

etapa 10 "Parte 5 — correção, antes/depois"
$PY -u -m main --mode fix --config configs/p2_dsb2018.yaml --checkpoint "$CKPT" || exit 1

etapa 11 "Parte 6 — teste de estresse por corrupções"
$PY -u -m main --mode stress --config configs/p2_dsb2018.yaml --checkpoint "$CKPT" || exit 1

echo "=== TUDO PRONTO — $(date '+%d/%m %H:%M') ==="
