# Segmentação de instâncias com arquiteturas de segmentação semântica

Programming Assignment 1 — Aprendizado Profundo (FGV).

Fazer arquiteturas de segmentação **semântica** (U-Net / ResUNet / DeepLab) produzirem
rótulos **instance-aware**, sem detectores com proposta de região. Dataset: **DSB2018 /
BBBC038v1** (núcleos de microscopia).

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate

# torch CPU (evita baixar ~2.5 GB de CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Dados

```bash
bash scripts/download_data.sh      # baixa e organiza o DSB2018 em data/
```

A Parte 0 usa um dataset **sintético** gerado em memória — não precisa de download.

## Treinar

```bash
python -m src.train --config configs/p0_synthetic.yaml
```

## Avaliar

```bash
python -m src.eval --config configs/p0_synthetic.yaml --checkpoint outputs/p0/best.pth
```

Reporta IoU, Dice, mAP de instância (0,50:0,05:0,95) e erro absoluto de contagem.

## Testes

```bash
pytest tests/ -v
```

Os testes da métrica de instância rodam sem treinar nada e validam o matching antes de
qualquer número ser reportado.

## Estrutura

```
src/
├── data/       geração sintética (P0) e DSB2018 (P1)
├── models/     U-Net e variantes
├── losses/     funções de perda
├── decode/     pós-processamento: previsão -> objetos numerados
├── metrics/    IoU/Dice semânticos e mAP de instância (implementado à mão)
├── train.py
└── eval.py
```

## Métrica de instância

Implementada à mão em `src/metrics/instance.py` (bibliotecas de AP de instância são proibidas
pelo enunciado). Definição usada:

- IoU calculado entre **cada par** (objeto previsto, objeto real);
- matching **1-para-1**, permitido apenas quando `IoU >= t`;
- `AP(t) = TP / (TP + FP + FN)`, contando **objetos** (convenção DSB2018);
- `mAP` = média de `AP(t)` sobre `t = 0,50; 0,55; ...; 0,95`.

A regra de matching (gulosa por IoU decrescente ou Hungarian) é selecionável — ver
`src/metrics/instance.py`.
