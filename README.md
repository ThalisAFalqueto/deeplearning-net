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
├── models/     backbones (unet, segnet, resunet, pspnet, unet_ppm) + cabeças
├── losses/     funções de perda
├── metrics/    IoU/Dice semânticos e mAP de instância (implementado à mão)
├── training/   loop de treino
├── evaluation/ avaliação e figuras
├── ablation/   Parte 3 — várias configs × várias seeds
├── mosaic/     Parte 4 — mosaico, tiles e costura
├── fails/      Parte 5 — galeria de falhas e a correção
├── stress/     Parte 6 — corrupções e curva de degradação
└── utils/      pós-processamento: previsão -> objetos numerados
tests/          testes das métricas, da decodificação e de cada parte
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

O enunciado observa que regras diferentes dão números diferentes. Medimos: **nas 136 imagens
de validação as duas produzem exatamente o mesmo mAP (0,4439)** — não há uma única imagem em
que divirjam.

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

## Parte 1 — baseline no DSB2018

Split estratificado por modalidade (ver adiante), 534 imagens de treino e 136 de validação,
U-Net binária, 20 épocas.

| | Semântico | | Instância | |
|---|---|---|---|---|
| **Treino: 33,1 min (CPU)** | IoU | **0,8098** | mAP | **0,4439** |
| | Dice | **0,8870** | erro de contagem | **10,42** objetos/imagem |

O gráfico exigido pelo item 5 está em `outputs/figures/p1_densidade.png`: conforme a densidade
de objetos cresce, o IoU semântico praticamente não se move enquanto o mAP de instância cai
e o erro de contagem cresce uma ordem de grandeza.

`outputs/figures/p1_fusao.png` mostra o mesmo contra a **fração de objetos que se tocam**, que é
a variável causal — a densidade é apenas uma proxy dela. Uma imagem com 300 núcleos bem
espaçados não quebra o baseline; o que quebra é o toque:

| objetos que se tocam | IoU | mAP |
|---|---|---|
| 0 – 10% | 0,833 | 0,589 |
| 10 – 30% | 0,794 | 0,359 |
| 30 – 50% | 0,787 | 0,198 |
| 50 – 70% | 0,781 | 0,064 |

O IoU cai 6% e o mAP cai 89% na mesma faixa. Correlação do mAP com a densidade: −0,240;
com a taxa de fusão: −0,577.

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
| imagem inteira numa passada (referência) | 0,410 | 86,5 | 0,851 |
| tiles decodificados um a um + parte interna (**antes**) | 0,346 | 127,5 | 0,759 |
| correção A: fusão por IoU na faixa de sobreposição | 0,407 | 91,5 | 0,857 |
| mapas densos pela parte interna, decodificados uma vez | 0,409 | 89,0 | 0,851 |
| média simples dos mapas densos | 0,383 | 96,5 | 0,843 |
| correção B: média dos mapas ponderada pelo interior | 0,412 | 87,2 | 0,850 |

Checkpoint `outputs/p2/best.pth` (Trilha C, 100 épocas), avaliado em CPU. São 6 mosaicos com
3852 núcleos, dos quais 428 cruzam uma divisa. O IoU semântico fica entre 0,833 e 0,836 em
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
