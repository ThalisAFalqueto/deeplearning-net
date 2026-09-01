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

A Parte 0 usa um dataset **sintético**, gerado em memória — não precisa de download nenhum.

O DSB2018 (Parte 1 em diante) é baixado com:

```bash
bash scripts/download_data.sh      # baixa e organiza o DSB2018 em data/
```

## Treinar

```bash
python -m src.train --config configs/p0_synthetic.yaml
```

## Avaliar

```bash
python -m src.eval --config configs/p0_synthetic.yaml
```

Reporta IoU, Dice, mAP de instância (0,50:0,05:0,95) e erro absoluto de contagem, e salva
`outputs/<nome>/predicoes.png` com imagem, gabarito e predição lado a lado.

O checkpoint padrão é `<output_dir>/best.pth`; use `--checkpoint` para apontar outro.

## Testes

```bash
pytest tests/ -v
```

Os testes da métrica de instância rodam sem treinar nada e validam o matching antes de
qualquer número ser reportado.

## Estrutura

```
configs/        hiperparâmetros por experimento (YAML)
src/
├── data/       geração sintética (P0) e DSB2018 (P1)
├── models/     U-Net com out_channels configurável
├── losses/     funções de perda
├── decode/     pós-processamento: previsão -> objetos numerados
├── metrics/    IoU/Dice semânticos e mAP de instância (implementado à mão)
├── train.py    o "um comando que treina"
└── eval.py     o "um comando que avalia"
tests/          testes das métricas e da decodificação
outputs/        checkpoints, figuras e logs (fora do git)
```

A mesma `UNet` serve as três Partes; muda apenas `out_channels`: 1 canal nas Partes 0 e 1
(logit de foreground), `1 + D` na Parte 2 (foreground + embedding por pixel).

## Parte 0 — teste unitário sintético

Imagens 128×128 com 5 a 20 elipses de tamanhos variados, ruído e contraste sorteados por
imagem, e aglomerados de elipses que se tocam. As máscaras de instância saem de graça, já
que as elipses são desenhadas uma a uma.

O quanto as elipses se tocam varia **por imagem**, de forma que o conjunto cobre desde cenas
esparsas até aglomerados densos. No conjunto gerado, 60% dos objetos se fundem: 469 objetos
formam apenas 143 componentes conexos.

| | |
|---|---|
| Treino | 1,9 min em CPU (7,7 s/época × 15 épocas) |
| IoU semântico | 0,9956 |

O IoU quase perfeito com um mAP de instância baixo é o resultado esperado, não um defeito: a
segmentação semântica responde "onde tem objeto?" e acerta, mas a máscara binária não
carrega nenhuma informação capaz de separar dois objetos encostados.

## Métrica de instância

Implementada à mão em `src/metrics/instance.py` (bibliotecas de AP de instância são proibidas
pelo enunciado). Definição usada:

- IoU calculado entre **cada par** (objeto previsto, objeto real);
- matching **1-para-1**, permitido apenas quando `IoU >= t`;
- `AP(t) = TP / (TP + FP + FN)`, contando **objetos** (convenção DSB2018);
- `mAP` = média de `AP(t)` sobre `t = 0,50; 0,55; ...; 0,95`.

A regra de matching (gulosa por IoU decrescente ou Hungarian) é selecionável — ver
`src/metrics/instance.py`.
