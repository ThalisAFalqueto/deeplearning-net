# Segmentação de instâncias com arquiteturas de segmentação semântica

Programming Assignment 1 — Aprendizado Profundo (FGV).

Fazer arquiteturas de segmentação **semântica** (U-Net / ResUNet / DeepLab) produzirem
rótulos **instance-aware**, sem detectores com proposta de região. Dataset: **DSB2018 /
BBBC038v1** (núcleos de microscopia).

## Quick Start

### Setup

```bash
# Instalação completa (CPU + CUDA)
make setup

# Instalação sem pacote de placa de vídeo
make setup-cpu
```

### Rodar cada parte

```bash
# Parte 0 — teste unitário sintético (treino + avaliação em CPU, ~2 min)
pytest tests/ -v
python -m main --synthetic

# Parte 1 — baseline binário no DSB2018 (treino + avaliação)
python -m main --config configs/p1_binario_dsb2018.yaml

# Parte 2 — Trilha C no DSB2018
python -m main --config configs/p2_dsb2018.yaml

# Parte 3 — ablação (compara arquiteturas em múltiplas seeds)
python -m main --ablation configs/default.yaml configs/segnet_dsb2018.yaml --ablation-seeds 0 42

# Parte 4 — inferência em mosaico
python -m main --mode mosaic --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth

# Parte 5 — galeria de falhas e a correção (antes/depois)
python -m main --mode fails --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
python -m main --mode fix   --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

### Modos de execução

| Comando | O que faz |
|---|---|
| `python -m main` | Treino + avaliação em sequência |
| `python -m main --mode train` | Apenas treino |
| `python -m main --mode eval` | Apenas avaliação |
| `python -m main --mode mosaic` | Parte 4 — inferência em mosaico |
| `python -m main --mode fails` | Parte 5 — galeria de falhas |
| `python -m main --mode fix` | Parte 5 — correção, com antes/depois |
| `python -m main --synthetic` | Atalho para `--config configs/synthetic.yaml` |
| `python -m main --resume` | Retoma treino de `outputs/<dir>/last.pth` |

## Ambiente

```bash
make setup  # Instalação

make setup-cpu  # Instalação sem utilizar pacote de placa de vídeo
```

## Dados

A Parte 0 usa um dataset **sintético**, gerado em memória — não precisa de download nenhum.

O DSB2018 (Parte 1 em diante) vem do Data Science Bowl 2018 (Kaggle / BBBC038v1). Baixe o
`stage1_train` e descompacte em `src/data/datasets/dsb2018/`:

```bash
# precisa do CLI do Kaggle autenticado (https://www.kaggle.com/docs/api)
kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip
unzip stage1_train.zip -d src/data/datasets/dsb2018/
```

**O layout interno não importa.** O carregador (`src/data/factory.py`) faz busca recursiva e
considera amostra qualquer pasta que contenha `images/`; subpastas como `train/` e
`validation/` são ignoradas, porque a divisão é sempre recalculada por modalidade a partir da
seed (ver "Split estratificado por modalidade"). O que importa é a forma de cada amostra:

```
src/data/datasets/dsb2018/<qualquer coisa>/<id da amostra>/
├── images/<id da amostra>.png     imagem
└── masks/*.png                    uma máscara por núcleo
```

São 670 amostras. Os dados não vão para o git (`.gitignore`).

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
├── models/     backbones (unet, segnet, resunet, pspnet, unet_ppm) + cabeças
├── losses/     funções de perda
├── metrics/    IoU/Dice semânticos e mAP de instância (implementado à mão)
├── training/   loop de treino
├── evaluation/ avaliação e figuras
├── ablation/   Parte 3 — várias configs × várias seeds
├── mosaic/     Parte 4 — mosaico, tiles e costura
├── fails/      Parte 5 — galeria de falhas e a correção
├── stress/     Parte 6 — corrupções e curva de degradação
├── inference/  inferência avulsa: caminho de imagem -> instâncias + contagem
└── utils/      pós-processamento: previsão -> objetos numerados
tests/          testes das métricas, da decodificação e de cada parte (175)
notebooks/      inferencia.ipynb — o entregável que roda sem retreinar
scripts/        treinar_tudo.sh (os treinos em série) e progresso.sh
outputs/        figuras, métricas e checkpoints (tudo versionado)
```

O mesmo backbone `UNet` serve as três Partes; o que muda é a **cabeça**, escolhida pela perda
(`src/models/factory.py`): 1 logit de foreground nas Partes 0 e 1 (`bce`), e três saídas na
Parte 2 (`center_offset` — segmentação, heatmap de centros e offsets).

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

### Regra de matching

**Os resultados reportados usam matching guloso por IoU decrescente.** O Hungarian
(atribuição ótima, via `scipy.optimize.linear_sum_assignment`) também está implementado e é
selecionável pelo parâmetro `matcher` de `MeanAveragePrecision`.

O enunciado observa que regras diferentes dão números diferentes. Medimos na primeira run da
Parte 1 (05/09, `outputs/p1`, 136 imagens de validação): **as duas produzem exatamente o mesmo
mAP (0,4439)** — não há uma única imagem em que divirjam.

Isso não é coincidência do dataset, é consequência do limiar. Se um objeto previsto `P`
tivesse IoU > 0,5 com dois objetos reais disjuntos `A` e `B`, teríamos `|P∩A| > 0,5·|P|` e
`|P∩B| > 0,5·|P|`; somando, `|P∩A| + |P∩B| > |P|`, impossível para conjuntos disjuntos
contidos em `P`. Ou seja, **com limiar ≥ 0,50 nenhum objeto previsto tem mais de um
candidato**, e sem ambiguidade a escolha gulosa já é a ótima. Verificado empiricamente:
0 casos ambíguos em 4.469 objetos previstos.

A divergência só apareceria com limiares abaixo de 0,50, que a métrica definida pelo
enunciado não utiliza. O teste `test_guloso_e_hungaro_divergem` constrói uma matriz de IoU
artificial em que elas divergem, para garantir que as duas implementações são de fato
diferentes.

## Parte 1 — baseline de segmentação semântica

U-Net binária (1 logit de foreground), instâncias extraídas pelo método ingênuo: limiar 0,5 +
componentes conexos. 568 imagens de treino e 102 de validação, 30 épocas, lr 0,006.

```bash
python -m main --config configs/p1_binario_dsb2018.yaml
```

| | Semântico | | Instância | |
|---|---|---|---|---|
| **Treino: 48,9 min (CPU)** | IoU | **0,7828** | mAP | **0,4053** |
| | Dice | **0,8666** | erro de contagem | **12,81** objetos/imagem |

Este é o regime **pareado com a Parte 2** — mesma seed, mesmas imagens de validação, mesmo lr
e mesmas épocas. É ele que sustenta a comparação da seção seguinte. Uma primeira versão desta
parte, de 05/09, está em `outputs/p1` (IoU 0,8098 / mAP 0,4439), mas usou outro split (136
imagens), lr 0,001 e 20 épocas, então não serve para comparar com nada.

### O fracasso quantificado (item 5)

`outputs/figures/p1_binario_densidade.png` traz o gráfico que o enunciado pede: conforme a
densidade de objetos cresce, o IoU semântico quase não se move enquanto o mAP de instância cai.

`outputs/figures/p1_binario_fusao.png` mostra o mesmo contra a **fração de objetos que se
tocam**, que é a variável causal — a densidade é só uma proxy dela. Uma imagem com 300 núcleos
bem espaçados não quebra o baseline; o que quebra é o toque:

| objetos que se tocam | imagens | IoU | mAP |
|---|---|---|---|
| 0 – 10% | 43 | 0,815 | 0,549 |
| 10 – 30% | 51 | 0,764 | 0,321 |
| 30 – 50% | 7 | 0,736 | 0,187 |
| 50 – 70% | 1 | 0,693 | 0,046 |

**O IoU cai 15% e o mAP cai 92% na mesma faixa.** Correlação do mAP com a taxa de fusão:
−0,559; com a densidade: −0,233. A máscara binária continua boa; o que ela não carrega é
qualquer informação capaz de separar dois núcleos encostados.

É esse buraco que a Parte 2 ataca.

## Parte 2 — cabeça de instâncias (Trilha C)

```bash
python -m main --config configs/p2_dsb2018.yaml
```

O encoder-decoder é **o mesmo** da Parte 1; o que muda é o que ele prevê. Em vez de um logit,
três saídas (`src/models/heads.py`):

- **segmentação** — foreground, como antes (perda BCE);
- **heatmap de centros** — uma gaussiana em cada centroide, com sigma proporcional ao tamanho
  do objeto (perda L2 ponderada, `pos_weight` compensa o pico ser ~0,06% da imagem);
- **offsets (Δy, Δx)** — para cada pixel de foreground, o vetor que aponta para o centro do
  seu próprio objeto (perda L1, medida só no foreground).

A decodificação (`src/utils/decode_center_offset.py`) acha os centros como máximos locais do
heatmap e atribui cada pixel ao centro mais próximo do seu **ponto deslocado**. É isso que
separa objetos colados: dois pixels vizinhos na fronteira têm offsets opostos e viajam para
centros diferentes.

### O resultado, lado a lado com a baseline

| métrica | Parte 1 (binária) | Parte 2 (Trilha C) | variação |
|---|---|---|---|
| IoU | 0,7828 | 0,7994 | +2,1% |
| Dice | 0,8666 | 0,8785 | +1,4% |
| **mAP** | 0,4053 | **0,4566** | **+12,7%** |
| **erro de contagem** | 12,81 | **9,85** | **−23,1%** |

Mesma seed, mesmas 102 imagens de validação, mesmo lr 0,006, mesmas 30 épocas. A única
diferença é o que a rede prevê.

**A forma do resultado é o argumento.** As métricas semânticas quase não se mexem: a máscara
já era boa e continua parecida. As de instância saltam. Mudar a representação não melhorou a
segmentação — melhorou a *separação*, que era exatamente o que faltava.

**Ressalva honesta:** o lr 0,006 foi escolhido para a Trilha C e herdado pela binária. No
treino da binária a perda de treino estagnou enquanto o IoU de validação oscilava entre 0,60 e
0,78, sintoma de taxa alta demais para uma cabeça mais simples. Parte dos +12,7% pode ser
baseline subajustada; um treino da binária com lr 0,001 está pendente para fechar essa dúvida.

## Parte 3 — ablações

Duas configurações por eixo, **2 seeds cada**, média ± desvio (desvio populacional, `np.std`
com ddof=0 — com 2 seeds, é metade da diferença entre elas).

### Eixo 1 — como recuperar resolução

U-Net (skip connections) contra SegNet (pool indices), mesmo encoder, 50 épocas
(`outputs/ablation/run_20260910_115413`):

| | IoU | mAP |
|---|---|---|
| U-Net (skips) | 0,8135 ± 0,0013 | **0,4897 ± 0,0009** |
| SegNet (pool indices) | 0,7897 ± 0,0018 | **0,4023 ± 0,0114** |

**−2,9% de IoU contra −17,8% de mAP.** Trocar skip connections por pool indices custa pouco
para a máscara e muito para as instâncias: as skips trazem de volta a localização precisa das
bordas, e é a borda que decide se dois núcleos encostados viram um ou dois objetos.

### Eixo 3 — contexto global

A `PSPNet` do projeto não responde à pergunta do enunciado: ela remove o decoder e as skips, e
o Eixo 1 acabou de mostrar que as skips são decisivas — a queda misturaria dois efeitos. Por
isso o eixo usa a **U-Net + Pyramid Pooling Module no bottleneck, com o decoder mantido**
(`src/models/unet_ppm.py`), em que a *única* diferença para a baseline é o PPM.

```bash
python -m main --ablation configs/baseline_pareado.yaml configs/unet_ppm_dsb2018.yaml --ablation-seeds 0 42
```

30 épocas, 2 seeds (`outputs/ablation/run_20260912_143042`):

| | IoU | mAP | erro de contagem |
|---|---|---|---|
| U-Net (baseline) | 0,8017 ± 0,0092 | **0,4722 ± 0,0151** | 8,13 ± 0,06 |
| U-Net + PPM | 0,7865 ± 0,0008 | **0,3417 ± 0,0680** | 24,59 ± 10,66 |

**A resposta à pergunta do enunciado é não: contexto global não ajudou a separar instâncias —
atrapalhou.** O mAP cai 27,6% e o erro de contagem triplica, enquanto o IoU quase não se move
(−1,9%). O efeito no mAP (0,131) é **4,3× o espalhamento entre as duas seeds da baseline**
(0,030), então não é ruído.

### O que exatamente piorou

O contraste entre os desvios é o achado mais informativo. O **IoU do PPM é o mais estável de
todo o experimento** (± 0,0008), enquanto o **mAP dele é o mais instável** (± 0,0680, quatro
vezes o da baseline). A máscara semântica sai bem e sai igual nas duas seeds; o que oscila
violentamente é a decodificação em instâncias.

A causa é super-segmentação:

| | razão previsto/real | imagens fragmentadas | objetos previstos (4233 reais) |
|---|---|---|---|
| baseline seed 0 / 42 | 1,00 / 1,00 | 13% / 15% | 3931 / 3912 |
| **PPM seed 0 / 42** | 1,00 / **1,17** | **26% / 47%** | **4720 / 6622** |

Na segunda seed o PPM inventa 56% mais objetos do que existem.

### Por que o contexto global não ajudou aqui

Três medições independentes explicam:

1. **Não havia déficit de contexto para corrigir.** A Parte 5 mediu que apenas **0,26% dos
   núcleos passam do campo receptivo de 68 px**. O PPM foi resolver um problema que este
   modelo não tinha.
2. **A falha real era o oposto.** A Parte 5 mostrou que o modo de falha dominante é
   fragmentação por picos espúrios no heatmap, não fusão por falta de contexto. Suavizar as
   features com contexto global piora exatamente esse mecanismo.
3. **Não é subtreino.** Nas últimas 10 épocas a perda de treino do PPM caiu 8,1%, contra 13,7%
   da baseline — o modelo maior desacelera *antes*, não depois. Se estivesse faltando treino,
   ele estaria caindo mais rápido no fim.

Um detalhe que vale a pena: nas épocas 4 a 7 o PPM estava **à frente** (IoU 0,724 contra
0,681). Contexto global ajuda enquanto o encoder ainda não aprendeu a olhar longe, e deixa de
ajudar quando ele já dá conta. Não é que o pyramid pooling seja inútil — é que este problema,
com estes objetos, não tinha o déficit que ele resolve.

### Ressalvas

- **Capacidade:** U-Net 482,5k parâmetros contra U-Net+PPM 794,3k (+65%), quase tudo na conv
  3×3 de projeção do PPM. Como o PPM *perdeu*, o confundimento joga contra ele — mas a
  evidência de que não é subtreino (item 3 acima) responde a isso.
- **Não testamos 100 épocas.** Não dá para afirmar que o PPM nunca alcançaria a baseline, só
  que no orçamento em que foram comparados ele fica atrás e sem sinal de recuperação.
- **Os dois eixos estão em regimes diferentes** — Eixo 1 com 50 épocas na máquina do João,
  Eixo 3 com 30 na do Thalis. Cada eixo compara dentro de si, que é o que a ablação exige.
  Para dimensionar: as duas baselines ficam a **0,017** de mAP uma da outra (0,4897 com 50
  épocas contra 0,4722 com 30), **menos que os 0,030 de espalhamento entre seeds** do
  experimento de 30 épocas. O efeito do regime é menor que o ruído do próprio experimento.
  O `configs/unet_ppm_50ep.yaml` está pronto para rodar o Eixo 3 em 50 épocas e eliminar essa
  ressalva.

## Checkpoint e qual resultado é qual

O repositório tem quatro resultados de Parte 1 / Parte 2, e eles têm papéis diferentes:

| pasta | IoU | mAP | papel |
|---|---|---|---|
| `outputs/p2/best.pth` | 0,8192 | 0,4960 | **modelo final** — Trilha C, 100 épocas. É o checkpoint entregue, e o usado nas Partes 4, 5 e 6 |
| `outputs/p1_binario` | 0,7828 | 0,4053 | baseline **pareada** da Parte 1 |
| `outputs/p2_pareado` | 0,7994 | 0,4566 | Trilha C **pareada**, o par da linha acima |
| `outputs/p1` | 0,8098 | 0,4439 | primeira versão da Parte 1 (05/09), outro split — histórico |

O par do meio é o que prova a tese da Parte 2. O `outputs/p2` é o modelo final porque treinou
mais (100 épocas contra 30) e é o mais forte; ele não entra na tabela pareada justamente
porque não teria par.

Os pesos estão versionados no próprio git — não há link externo.

## Split estratificado por modalidade

O DSB2018 mistura três tipos de imagem, detectados automaticamente
(`src/data/modality.py`) pela saturação de cor e pelo brilho mediano:

| Modalidade | Amostras | |
|---|---|---|
| Fluorescência | 546 | 81,5% |
| Histologia H&E | 108 | 16,1% |
| Brightfield | 16 | 2,4% |

A divisão treino/validação é feita **dentro de cada modalidade** (`src/data/split.py`),
preservando a proporção nos dois conjuntos. Com apenas 16 imagens brightfield, um sorteio
simples deixaria entre 2 e 6 delas na validação conforme a seed; estratificando, são sempre
4. Isso importa porque as ablações da Parte 3 usam 2 seeds e reportam média ± desvio — sem
estratificar, parte do desvio seria a variação de composição do split, e não o efeito que se
quer medir.

A divisão é sempre estratificada — não há chave para desligar. Ela é lógica: nenhum arquivo é
movido, e a classificação de modalidade fica em cache (`.modality_cache.json`). O config
informa só `data_dir` e `val_fraction`; a seed decide o sorteio dentro de cada modalidade.

## Parte 4 — inferência em mosaico

```bash
python -m main --mode mosaic --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

Monta mosaicos 4×4 com imagens de validação da mesma modalidade (1024², com gabarito
conhecido). Depois roda a rede em tiles 256² com passo 192 (sobreposição de 64 px; só 0,4% dos
núcleos passam de 64 px) e compara formas de costurar o resultado. Resultados em
`outputs/p4/summary.json` e `per_mosaic.json`. As figuras são `p4_mosaico.png` (grade de tiles
e divisas), `p4_fronteira.png` (um núcleo na divisa em cada estratégia) e `p4_barras.png`.

A **divisa** é onde termina a parte interna de um tile e começa a do vizinho: o meio da faixa
de sobreposição (224, 416, 608 e 800 px).

| estratégia | mAP | erro de contagem | recall dos núcleos na divisa |
|---|---|---|---|
| imagem inteira numa passada (referência) | 0,415 | 87,0 | 0,851 |
| tiles decodificados um a um + parte interna (**antes**) | 0,352 | 126,7 | 0,761 |
| correção A: fusão por IoU na faixa de sobreposição | 0,413 | 92,2 | 0,857 |
| mapas densos pela parte interna, decodificados uma vez | 0,414 | 89,5 | 0,851 |
| média simples dos mapas densos | 0,387 | 96,7 | 0,844 |
| correção B: média dos mapas ponderada pelo interior | 0,417 | 87,8 | 0,850 |

Checkpoint `outputs/p2/best.pth` (Trilha C, 100 épocas), avaliado em CPU. São 6 mosaicos com
3909 núcleos, dos quais 427 cruzam uma divisa. O IoU semântico fica entre 0,833 e 0,836 em
todas as estratégias.

- **Para a segmentação semântica, a receita do slide funciona.** A máscara de foreground
  costurada difere da passada inteira em 38 a 583 pixels por mosaico de 1 milhão (no máximo
  0,06%).
- **Para instâncias, costurar rótulos quebra.** O número de um objeto é arbitrário em cada
  tile, então um núcleo que cruza a divisa vira dois (ou quatro, no cruzamento de divisas). O
  mAP cai 15% e o erro de contagem sobe 46%.
- **O problema é decodificar tile a tile, e não o contexto perdido na borda.** Costurando os
  mapas densos pela parte interna e decodificando uma vez só, o resultado já é o da passada
  inteira.
- **Correção A:** junta os pedaços que dois tiles vizinhos veem na mesma região da faixa de
  sobreposição. Serve para qualquer representação, inclusive a binária.
- **Correção B:** é o que a Trilha C permite. Offsets e heatmap são grandezas por pixel, não
  números de objeto, então dá para fazer a média entre tiles. A média precisa pesar o
  interior: a média simples mistura previsões feitas na borda do tile, com pouco contexto, e
  perde 7%.

Os números absolutos são menores que os da Parte 2 (0,496) por dois motivos, e os dois atingem
todas as estratégias igualmente:
1. **O mAP é calculado por mosaico** (~650 núcleos), e não como média por imagem.
2. **As emendas entre imagens criam bordas que não existem na imagem sozinha.** As mesmas 96
   imagens dão 0,512 avaliadas sozinhas e 0,458 recortadas do mosaico.

## Parte 5 — galeria de falhas e correção

```bash
python -m main --mode fails --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
python -m main --mode fix   --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

### Campo receptivo × tamanho dos objetos (item obrigatório)

O campo receptivo teórico do encoder é **68 px**, pela recorrência dos slides 35-38 em
`UNet.receptive_field()`. Nenhuma arquitetura do projeto usa atrous/dilated convolution, então
a comparação "com e sem atrous" do enunciado não se aplica; o mecanismo equivalente que temos
é o Pyramid Pooling Module (`pspnet`/`unet_ppm`).

Comparando com os 4.233 núcleos da validação (`outputs/p5/p5_distribuicao.png`), o "tamanho"
precisa ser definido, porque a conclusão muda com a definição:

| medida | p50 | p95 | máx | acima do campo receptivo |
|---|---|---|---|---|
| maior extensão do núcleo | 13 px | 43 px | 117 px | **0,26%** (11 núcleos) |
| diâmetro equivalente (círculo de mesma área) | 11 px | 35 px | 63 px | 0,00% |

A extensão é a medida que importa para o argumento do enunciado — "o pixel central nunca
enxerga as duas bordas". Pelo diâmetro equivalente nenhum núcleo alcança o campo receptivo;
pela extensão, 11 alcançam. São os núcleos alongados, que o círculo de mesma área encolhe.

### O diagnóstico

Campo receptivo curto faz o modelo **fundir** objetos; ele nunca divide um objeto em vários.
Por isso a galeria classifica o modo de falha antes de explicar, contando quantos rótulos a
predição colocou dentro do maior aglomerado do gabarito. Nas 5 piores imagens:

| falha | modalidade | gabarito → predição | modo | causa |
|---|---|---|---|---|
| idx 62 | fluorescência | 375 → 115 | fundiu | densidade: 375 núcleos, contra mediana de 24 |
| idx 56 | histologia | 13 → 49 | **fragmentou** | picos espúrios no heatmap |
| idx 41 | histologia | 9 → 39 | **fragmentou** | picos espúrios no heatmap |
| idx 4 | histologia | 21 → 46 | **fragmentou** | picos espúrios no heatmap |
| idx 10 | brightfield | 46 → 34 | fundiu | núcleos pequenos demais para separar |

Três das cinco piores falhas são **super-segmentação em histologia**, e nelas o campo receptivo
não explica nada: no idx 41 um aglomerado de 119 px — maior que os 68 px do campo receptivo —
foi dividido em 13 rótulos para 2 núcleos, o oposto do que um campo receptivo curto produziria.
A razão mediana predição/gabarito confirma o padrão: histologia 1,34 · fluorescência 1,00 ·
brightfield 0,84.

### A correção, com antes e depois

O diagnóstico aponta para a decodificação, não para os pesos: a textura do tecido gera muitos
máximos locais fracos no heatmap, e cada um vira um objeto. A correção aperta a seleção de
picos — `peak_threshold`, `nms_kernel` e área mínima (`remove_small_objects`). O modelo é o
mesmo; só a decodificação muda.

| decodificação | mAP | erro de contagem |
|---|---|---|
| **antes** — pico 0,5, nms 3, sem filtro de área | 0,4984 | 7,04 |
| **depois** — pico 0,6, nms 7, área ≥ 25 px | **0,5294** | 8,63 |
| alternativa — pico 0,6, nms 7, sem filtro de área | 0,5142 | **6,96** |

O mAP sobe **+0,0310** (+6,2%) e a fragmentação em histologia cai de 1,34 para 1,13 rótulos
por núcleo real — exatamente o que o diagnóstico previa.

**O que a correção custa.** O erro de contagem **piora** (7,04 → 8,63), porque o filtro de
área também apaga núcleos pequenos verdadeiros. Em brightfield, que já sub-contava, a razão
cai de 0,84 para 0,43. A correção conserta a falha diagnosticada e agrava a falha oposta. A
linha "alternativa", sem filtro de área, melhora as duas métricas com um ganho de mAP menor —
é a escolha defensável se a contagem importar tanto quanto o mAP.
`outputs/p5/p5_antes_depois.png` mostra as duas coisas na mesma figura: a histologia
desfragmenta e a imagem densa de fluorescência perde ainda mais núcleos.

**Ressalva.** Os parâmetros foram escolhidos varrendo o mesmo conjunto de validação em que o
resultado é reportado — o trabalho não tem conjunto de teste separado —, então o ganho medido
é otimista. A varredura das 36 combinações está em `outputs/p5/p5_varredura.png` e em
`outputs/p5/correcao.json`.

## Parte 6 — teste de estresse por corrupções

```bash
python -m main --mode stress --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

Das três opções do enunciado, escolhemos **corrupções**: borrão, ruído e brilho/contraste, em
3 intensidades cada. São 10 condições (a imagem limpa mais 3 × 3) sobre as 102 imagens de
validação, com o mesmo modelo e os mesmos pesos — muda só a imagem que entra. Os rótulos nunca
são corrompidos. As corrupções são aplicadas na imagem **já em 256²**, o espaço de entrada do
modelo: o que se testa aqui é o modelo, não o pipeline de dados.

| família | fórmula | nível 1 | nível 2 | nível 3 |
|---|---|---|---|---|
| borrão | `gaussian_filter(img, σ)` | σ = 1 px | σ = 2 px | σ = 4 px |
| ruído | `img + N(0, σ)`, cortado em [0,1] | σ = 0,02 | σ = 0,05 | σ = 0,10 |
| brilho/contraste | `(img − 0,5)·(1−f) + 0,5 + f/2` | f = 0,2 | f = 0,4 | f = 0,6 |

**O modelo foi treinado sem augmentation nenhuma** — não há transform, flip ou rotação em
lugar nenhum do `src/`. Ele nunca viu estas corrupções, então uma curva íngreme é o resultado
esperado, não uma descoberta. A severidade 0 é a imagem limpa, e o mAP nesse ponto reproduz
exatamente o da Parte 2 (0,4984), o que serve de checagem de que o caminho novo não mexeu em
nada.

### Curva de degradação (`outputs/p6/p6_degradacao.png`)

| condição | mAP | IoU | erro de contagem |
|---|---|---|---|
| limpa | 0,4984 | 0,8193 | 7,0 |
| borrão σ=1 px | 0,4656 | 0,7947 | 7,9 |
| borrão σ=2 px | 0,3234 | 0,7022 | 9,8 |
| borrão σ=4 px | 0,1341 | 0,5187 | 16,6 |
| ruído σ=0,02 | 0,2563 | 0,7274 | 35,0 |
| ruído σ=0,05 | 0,0368 | 0,4646 | **509,7** |
| ruído σ=0,10 | 0,0051 | 0,2518 | **1240,3** |
| brilho/contraste f=0,2 | 0,1607 | 0,4276 | 14,6 |
| brilho/contraste f=0,4 | 0,0395 | 0,1260 | 29,2 |
| brilho/contraste f=0,6 | 0,0090 | 0,0352 | 37,6 |

### O que a curva esconde: são três falhas diferentes

O mAP cai nos três casos, mas **pelos três motivos opostos**, e isso só aparece olhando a
contagem e a figura `p6_exemplos.png` (numa imagem com 66 núcleos no gabarito):

| corrupção | núcleos previstos, do limpo ao nível 3 | o que o modelo faz |
|---|---|---|
| borrão | 65 → 65 → 65 → 59 | **mantém a contagem** e perde precisão de borda |
| ruído | 65 → 75 → 385 → **1363** | **alucina objetos**: cada grão de ruído vira um pico no heatmap |
| brilho/contraste | 65 → 61 → 9 → **1** | **fica cego**: o foreground desaparece |

- **Borrão é o menos grave.** Mesmo com σ=4, que borra cerca de um terço de um núcleo típico
  (13 px de extensão, medido na Parte 5), o modelo ainda encontra quase todos os núcleos; o que
  ele perde é a borda exata, e é isso que derruba o mAP nos limiares altos de IoU.
- **Ruído é o mais perigoso**, e pela mesma razão que a Parte 5 encontrou: a decodificação
  transforma **todo máximo local do heatmap** num objeto. Ruído fabrica máximos locais aos
  milhares. É a mesma falha da histologia fragmentada, agora provocada de propósito — e sugere
  que a correção da Parte 5 (exigir picos mais altos) também ajudaria aqui.
- **Brilho/contraste é o único em que o IoU desaba mais que o mAP** (0,819 → 0,035). Nos outros
  dois a máscara semântica resiste enquanto as instâncias se perdem, que é o padrão da Parte 1.
  Aqui não: comprimir o contraste e clarear o fundo empurra a imagem inteira para fora da faixa
  que a rede viu no treino, e ela para de responder.

Ou seja: a hipótese herdada da Parte 1 — "o mAP cai mais rápido que o IoU" — **vale para borrão
e ruído, e não vale para brilho/contraste**.

### Por modalidade

Sob borrão de σ=4, a histologia degrada proporcionalmente **menos** que a fluorescência
(0,240 → 0,147, contra 0,565 → 0,136). É coerente com a distribuição de tamanhos da Parte 5:
núcleos de histologia são maiores, então um borrão de raio fixo os afeta relativamente menos.
Sob ruído e sob brilho/contraste nenhuma modalidade se salva.

Todos os números saem de CPU, com o checkpoint `outputs/p2/best.pth`. Rodar duas vezes dá
`summary.json` idêntico, inclusive o ruído — a semente de cada imagem é fixa.
