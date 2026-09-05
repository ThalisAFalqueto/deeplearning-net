# Segmentação de instâncias com arquiteturas de segmentação semântica

Programming Assignment 1 — Aprendizado Profundo (FGV).

Fazer arquiteturas de segmentação **semântica** (U-Net / ResUNet / DeepLab) produzirem
rótulos **instance-aware**, sem detectores com proposta de região. Dataset: **DSB2018 /
BBBC038v1** (núcleos de microscopia).

## Ambiente

```bash
make setup  # Instalação

make setup-cpu  # Instalação sem utilizar pacote de placa de vídeo
```

## Dados

A Parte 0 usa um dataset **sintético**, gerado em memória — não precisa de download nenhum.

O DSB2018 (Parte 1 em diante) é baixado com:

```bash
bash scripts/download_data.sh      # baixa e organiza o DSB2018 em data/
```

## Treinar

```bash
python -m main --config configs/synthetic.yaml --mode train
```

## Avaliar

```bash
python -m main --config configs/synthetic.yaml --mode eval
```

Reporta IoU, Dice, mAP de instância (0,50:0,05:0,95) e erro absoluto de contagem, e salva
`outputs/<nome>/predicoes.png` com imagem, gabarito e predição lado a lado.

O checkpoint padrão é `<output_dir>/best.pth`; use `--checkpoint` para apontar outro.

Sem `--mode`, executa treino e avaliação em sequência. `--synthetic` é atalho para
`--config configs/synthetic.yaml`:

```bash
python -m main --synthetic          # treina e avalia a Parte 0
```

## Testes

```bash
pytest tests/ -v
```

Os testes da métrica de instância rodam sem treinar nada e validam o matching antes de
qualquer número ser reportado.

## Estrutura

```
main.py         ponto de entrada único (treino e avaliação)
configs/        hiperparâmetros por experimento (YAML)
src/
├── core/       carregamento de configuração
├── data/       geração sintética (P0), DSB2018 (P1) e as factories
├── models/     U-Net com out_channels configurável
├── losses/     funções de perda
├── metrics/    IoU/Dice semânticos e mAP de instância (implementado à mão)
├── training/   loop de treino
├── evaluation/ avaliação e figuras
└── utils/      pós-processamento: previsão -> objetos numerados
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

Resultado em 60 imagens de validação, treinando 15 épocas em CPU:

| | Semântico | | Instância | |
|---|---|---|---|---|
| **Treino: 1,4 min** | IoU | **0,9994** | mAP | **0,2253** |
| | Dice | **0,9997** | erro de contagem | **7,00** objetos/imagem |

O IoU quase perfeito com um mAP baixo é o resultado esperado, não um defeito. A segmentação
semântica responde "onde tem objeto?" e acerta praticamente todos os pixels — mas a máscara
binária não carrega nenhuma informação capaz de separar dois objetos encostados. Em
`outputs/p0/predicoes.png` é possível ver um gabarito de 20 objetos virando 4 na predição.

É esse contraste que a Parte 2 ataca, mudando o que a rede prevê.

## Métrica de instância

Implementada à mão em `src/metrics/instance/` (bibliotecas de AP de instância são proibidas
pelo enunciado). Definição usada:

- IoU calculado entre **cada par** (objeto previsto, objeto real);
- matching **1-para-1**, permitido apenas quando `IoU >= t`;
- `AP(t) = TP / (TP + FP + FN)`, contando **objetos** (convenção DSB2018);
- `mAP` = média de `AP(t)` sobre `t = 0,50; 0,55; ...; 0,95`.

A regra de matching (gulosa por IoU decrescente ou Hungarian) é selecionável — ver
`src/metrics/instance.py`.
