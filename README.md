# Segmentação de instâncias com arquiteturas de segmentação semântica

Programming Assignment 1 — Aprendizado Profundo (FGV).

Fazer arquiteturas de segmentação **semântica** (U-Net / ResUNet / DeepLab) produzirem
rótulos **instance-aware**, sem detectores com proposta de região. Dataset: **DSB2018 /
BBBC038v1** (núcleos de microscopia).

## Quick Start

### Setup

Duas famílias de comandos, conforme você tem `uv` instalado ou não. Elas fazem a mesma coisa
e instalam nos mesmos lugares (`.venv/`); use uma ou outra, sem misturar.

**Com uv:**

```bash
# Instalação completa (CPU + CUDA)
make setup

# Instalação sem pacote de placa de vídeo
make setup-cpu
```

**Sem uv** (usa `venv` + `pip` e o `requirements.txt` versionado no repositório):

```bash
# Instalação completa (CPU + CUDA)
make setup-pip

# Instalação sem pacote de placa de vídeo
make setup-cpu-pip
```

### Rodar cada parte

**Com uv** (prefixa cada comando com `uv run`, sem precisar ativar o `.venv`):

```bash
# Parte 0 — teste unitário sintético (treino + avaliação em CPU, ~2 min)
uv run pytest tests/ -v
uv run python -m main --synthetic

# Parte 1 — baseline binário no DSB2018 (treino + avaliação)
uv run python -m main --config configs/p1_binario_dsb2018.yaml

# Parte 2 — Trilha C no DSB2018
uv run python -m main --config configs/p2_dsb2018.yaml

# Parte 3 — ablação (compara arquiteturas em múltiplas seeds)
uv run python -m main --ablation configs/default.yaml configs/segnet_dsb2018.yaml --ablation-seeds 0 42

# Parte 4 — inferência em mosaico
uv run python -m main --mode mosaic --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth

# Parte 5 — galeria de falhas e a correção (antes/depois)
uv run python -m main --mode fails --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
uv run python -m main --mode fix   --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

**Sem uv** (com o `.venv` ativado — `source .venv/bin/activate`):

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

Tabela com o comando "sem uv" (`.venv` ativado); com `uv` é o mesmo comando prefixado com
`uv run` (ex.: `uv run python -m main --mode train`).

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

**Com uv:**

```bash
make setup      # Instalação

make setup-cpu  # Instalação sem utilizar pacote de placa de vídeo
```

**Sem uv:**

```bash
make setup-pip      # Instalação

make setup-cpu-pip  # Instalação sem utilizar pacote de placa de vídeo
```

`make setup`/`make setup-cpu` usam [uv](https://docs.astral.sh/uv/) (`uv sync` + `uv pip
install`) — **se você não tiver o `uv` instalado, esses dois comandos não vão rodar**; use
`make setup-pip`/`make setup-cpu-pip`, que fazem o equivalente com `venv` + `pip` puro, a
partir do `requirements.txt` versionado no repositório (gerado com `uv export --no-hashes
--format requirements.txt -o requirements.txt`; só quem for atualizar dependências precisa
rodar isso, e precisa de `uv` para tal). Depois de instalado com qualquer uma das duas
famílias, os comandos do projeto rodam com o `.venv` ativado (`source .venv/bin/activate` e
então `python -m main ...`) ou, só quando a instalação foi feita com `uv`, sem ativar,
prefixando cada comando com `uv run` (`uv run python -m main ...`).

## Dados

A Parte 0 usa um dataset **sintético**, gerado em memória — não precisa de download nenhum.

O DSB2018 (Parte 1 em diante) é o BBBC038v1. Baixe o zip `stage1_train`, sem precisar de
conta, em [bbbc.broadinstitute.org/BBBC038](https://bbbc.broadinstitute.org/BBBC038) e
descompacte em `src/data/datasets/dsb2018/`.

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

**Com uv:**

```bash
uv run python -m main --config configs/synthetic.yaml --mode train
```

**Sem uv:**

```bash
python -m main --config configs/synthetic.yaml --mode train
```

## Avaliar

**Com uv:**

```bash
uv run python -m main --config configs/synthetic.yaml --mode eval
```

**Sem uv:**

```bash
python -m main --config configs/synthetic.yaml --mode eval
```

Reporta IoU, Dice, mAP de instância (0,50:0,05:0,95) e erro absoluto de contagem, e salva
`outputs/<nome>/predicoes.png` com imagem, gabarito e predição lado a lado.

O checkpoint padrão é `<output_dir>/best.pth`; use `--checkpoint` para apontar outro.

Sem `--mode`, executa treino e avaliação em sequência. `--synthetic` é atalho para
`--config configs/synthetic.yaml`:

**Com uv:**

```bash
uv run python -m main --synthetic   # treina e avalia a Parte 0
```

**Sem uv:**

```bash
python -m main --synthetic          # treina e avalia a Parte 0
```

## Testes

**Com uv:**

```bash
uv run pytest tests/ -v
```

**Sem uv:**

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
imagem, e aglomerados de elipses que se tocam. As máscaras de instância saem de graça, já que
as elipses são desenhadas uma a uma. O quanto as elipses se tocam varia **por imagem**, de
forma que o conjunto cobre desde cenas esparsas até aglomerados densos.

Saídas em `outputs/p0/`: `summary.json` (métricas agregadas) e `per_image.json` (métricas por
imagem de validação), `history.json` (curva de treino), `best.pth`/`last.pth` (checkpoints) e
`predicoes.png` (imagem, gabarito e predição lado a lado). Os gráficos de mAP/erro de contagem
em função da densidade de objetos e da fração de objetos que se tocam ficam em
`outputs/figures/p0_densidade.png` e `outputs/figures/p0_fusao.png`.

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

O enunciado observa que regras diferentes dão números diferentes, mas com limiar ≥ 0,50 as
duas são **matematicamente equivalentes**: se um objeto previsto `P` tivesse IoU > 0,5 com
dois objetos reais disjuntos `A` e `B`, teríamos `|P∩A| > 0,5·|P|` e `|P∩B| > 0,5·|P|`;
somando, `|P∩A| + |P∩B| > |P|`, impossível para conjuntos disjuntos contidos em `P`. Ou seja,
com limiar ≥ 0,50 nenhum objeto previsto tem mais de um candidato, e sem ambiguidade a escolha
gulosa já é a ótima — a divergência só apareceria com limiares abaixo de 0,50, que a métrica
definida pelo enunciado não utiliza.

O teste `test_guloso_e_hungaro_divergem` constrói uma matriz de IoU artificial em que elas
divergem, para garantir que as duas implementações são de fato diferentes (a equivalência
acima é uma propriedade do limiar escolhido, não das implementações).

## Parte 1 — baseline de segmentação semântica

U-Net binária (1 logit de foreground), instâncias extraídas pelo método ingênuo: limiar 0,5 +
componentes conexos. 568 imagens de treino e 102 de validação, 30 épocas, lr 0,006 — o mesmo
regime da Parte 2, para que as duas sejam comparáveis.

**Com uv:**

```bash
uv run python -m main --config configs/p1_binario_dsb2018.yaml
```

**Sem uv:**

```bash
python -m main --config configs/p1_binario_dsb2018.yaml
```

Saídas em `outputs/p1_binario/`: `summary.json` e `per_image.json` (métricas agregadas e por
imagem), `history.json` (curva de treino), `best.pth`/`last.pth` (checkpoints) e
`predicoes.png` (imagem, gabarito e predição lado a lado). O item 5 do enunciado — quantificar
o fracasso da abordagem binária conforme a cena fica mais densa — fica em
`outputs/figures/p1_binario_densidade.png` (mAP/IoU em função da densidade de objetos) e
`outputs/figures/p1_binario_fusao.png` (mAP/IoU em função da fração de objetos que se tocam,
a variável causal por trás da densidade).

`outputs/p1/` guarda uma versão histórica desta parte (outro split, outro lr, outras épocas) —
não é comparável com `outputs/p1_binario/`.

É esse baseline que a Parte 2 tenta melhorar, mudando o que a rede prevê.

## Parte 2 — cabeça de instâncias (Trilha C)

**Com uv:**

```bash
uv run python -m main --config configs/p2_dsb2018.yaml
```

**Sem uv:**

```bash
python -m main --config configs/p2_dsb2018.yaml
```

O encoder-decoder é **o mesmo** da Parte 1; o que muda é o que ele prevê. Em vez de um logit,
três saídas (`src/models/heads.py`):

- **segmentação** — foreground, como antes (perda BCE);
- **heatmap de centros** — uma gaussiana em cada centroide, com sigma proporcional ao tamanho
  do objeto (perda L2 ponderada, `pos_weight` compensa o pico ser uma fração minúscula da
  imagem);
- **offsets (Δy, Δx)** — para cada pixel de foreground, o vetor que aponta para o centro do
  seu próprio objeto (perda L1, medida só no foreground).

A decodificação (`src/utils/decode_center_offset.py`) acha os centros como máximos locais do
heatmap e atribui cada pixel ao centro mais próximo do seu **ponto deslocado**. É isso que
separa objetos colados: dois pixels vizinhos na fronteira têm offsets opostos e viajam para
centros diferentes.

Saídas em `outputs/p2/` (modelo final, treinado por mais épocas) e `outputs/p2_pareado/`
(regime pareado com a Parte 1 — mesma seed, mesmas imagens de validação, mesmo lr e mesmas
épocas): `summary.json`, `per_image.json`, `history.json`, `best.pth`/`last.pth` e
`predicoes.png`. As figuras de densidade/fusão equivalentes às da Parte 1 ficam em
`outputs/figures/p2_densidade.png`/`p2_fusao.png` (modelo final) e
`p2_pareado_densidade.png`/`p2_pareado_fusao.png` (regime pareado) — comparar essas com as da
Parte 1 é o argumento da Parte 2.

**Ressalva de metodologia:** o lr foi escolhido para a Trilha C e herdado pela baseline
binária da Parte 1, o que pode favorecer artificialmente a comparação entre as duas; um treino
da binária com lr dedicado ajudaria a descartar essa dúvida.

## Parte 3 — ablações

Duas configurações por eixo, **2 seeds cada**, reportando média ± desvio (desvio populacional,
`np.std` com ddof=0 — com 2 seeds, é metade da diferença entre elas).

### Eixo 1 — como recuperar resolução

U-Net (skip connections) contra SegNet (pool indices), mesmo encoder.

### Eixo 3 — contexto global

A `PSPNet` do projeto não responde à pergunta do enunciado: ela remove o decoder e as skips, e
o Eixo 1 já mede o efeito de removê-las — misturar os dois efeitos tornaria a comparação
inconclusiva. Por isso o eixo usa a **U-Net + Pyramid Pooling Module no bottleneck, com o
decoder mantido** (`src/models/unet_ppm.py`), em que a *única* diferença para a baseline é o
PPM.

**Com uv:**

```bash
uv run python -m main --ablation configs/baseline_pareado.yaml configs/unet_ppm_dsb2018.yaml --ablation-seeds 0 42
```

**Sem uv:**

```bash
python -m main --ablation configs/baseline_pareado.yaml configs/unet_ppm_dsb2018.yaml --ablation-seeds 0 42
```

### Onde os resultados ficam

Cada rodada grava em `outputs/ablation/run_<timestamp>/`:

- `report.json` / `report.md` — métricas agregadas (média ± desvio) por configuração e seed;
- `iou_bars.png`, `map_bars.png`, `dice_bars.png`, `count_error_bars.png`, `table.png` —
  gráficos e tabela comparando as configurações do eixo;
- `<config>/seed_<n>/` — uma subpasta por configuração e seed, com os artefatos de treino
  individuais (checkpoints, histórico, métricas por imagem).

## Checkpoints e o papel de cada pasta

O repositório versiona vários checkpoints de Parte 1 / Parte 2, cada um com um papel
diferente:

| pasta | papel |
|---|---|
| `outputs/p2/` | **modelo final** — Trilha C, treinada por mais épocas. É o checkpoint usado nas Partes 4, 5 e 6 |
| `outputs/p1_binario/` | baseline **pareada** da Parte 1 |
| `outputs/p2_pareado/` | Trilha C **pareada**, o par direto da linha acima (mesma seed, mesmas imagens de validação, mesmo lr, mesmas épocas) |
| `outputs/p1/` | versão histórica da Parte 1, com outro split e outros hiperparâmetros |

O par `p1_binario` / `p2_pareado` é o que sustenta a comparação da Parte 2 com a Parte 1;
`outputs/p2` é o modelo final porque treinou mais e é o mais forte, mas por isso mesmo não
entra nessa comparação pareada.

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
simples deixaria entre 2 e 6 delas na validação conforme a seed; estratificando, esse número
fica fixo. Isso importa porque as ablações da Parte 3 usam 2 seeds e reportam média ± desvio —
sem estratificar, parte do desvio seria a variação de composição do split, e não o efeito que
se quer medir.

A divisão é sempre estratificada — não há chave para desligar. Ela é lógica: nenhum arquivo é
movido, e a classificação de modalidade fica em cache (`.modality_cache.json`). O config
informa só `data_dir` e `val_fraction`; a seed decide o sorteio dentro de cada modalidade.

## Parte 4 — inferência em mosaico

**Com uv:**

```bash
uv run python -m main --mode mosaic --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

**Sem uv:**

```bash
python -m main --mode mosaic --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

Monta mosaicos 4×4 com imagens de validação da mesma modalidade (1024², com gabarito
conhecido), roda a rede em tiles 256² com passo 192 (sobreposição de 64 px) e compara formas
de costurar o resultado: a imagem inteira numa passada só (referência), cada tile decodificado
isoladamente, os mapas densos costurados antes de decodificar uma única vez, e duas correções
— fusão de instâncias por IoU na faixa de sobreposição, e média dos mapas densos ponderada
pelo interior de cada tile.

A **divisa** é onde termina a parte interna de um tile e começa a do vizinho: o meio da faixa
de sobreposição. Para segmentação semântica essa costura tende a funcionar bem; para
instâncias, decodificar tile a tile pode duplicar um objeto que cruza a divisa — daí as duas
correções.

Saídas em `outputs/p4/`: `summary.json` e `per_mosaic.json` com as métricas por estratégia e
por mosaico, e as figuras `p4_mosaico.png` (grade de tiles e divisas), `p4_fronteira.png` (um
núcleo na divisa em cada estratégia) e `p4_barras.png` (comparação entre estratégias).

## Parte 5 — galeria de falhas e correção

**Com uv:**

```bash
uv run python -m main --mode fails --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
uv run python -m main --mode fix   --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

**Sem uv:**

```bash
python -m main --mode fails --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
python -m main --mode fix   --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

### Campo receptivo × tamanho dos objetos (item obrigatório)

Calcula o campo receptivo teórico do encoder (`UNet.receptive_field()`, pela recorrência dos
slides 35-38) e compara com a distribuição de tamanhos dos núcleos de validação, sob duas
definições de "tamanho" — maior extensão do núcleo e diâmetro equivalente (círculo de mesma
área) — já que a conclusão muda dependendo de qual se usa. Nenhuma arquitetura do projeto usa
atrous/dilated convolution, então a comparação "com e sem atrous" do enunciado não se aplica;
o mecanismo equivalente disponível é o Pyramid Pooling Module (`pspnet`/`unet_ppm`).

Resultado em `outputs/p5/p5_distribuicao.png`.

### Diagnóstico e correção

A galeria escolhe as 5 piores imagens de validação e classifica o modo de falha de cada uma —
fusão (por densidade ou campo receptivo curto) ou fragmentação (por picos espúrios no
heatmap) — contando quantos rótulos a predição colocou dentro do maior aglomerado do gabarito.
As figuras ficam em `outputs/p5/falha_<n>_idx<i>.png` e o diagnóstico de cada uma em
`outputs/p5/piores.json`.

A correção aperta a seleção de picos na decodificação (`peak_threshold`, `nms_kernel` e área
mínima via `remove_small_objects`), sem retreinar o modelo. A varredura de parâmetros e a
comparação antes/depois ficam em `outputs/p5/correcao.json`, `outputs/p5/p5_varredura.png` e
`outputs/p5/p5_antes_depois.png`.

**Ressalva de metodologia:** os parâmetros da correção são escolhidos varrendo o mesmo
conjunto de validação em que o resultado é reportado — o trabalho não tem conjunto de teste
separado —, então o ganho medido é otimista.

## Parte 6 — teste de estresse por corrupções

**Com uv:**

```bash
uv run python -m main --mode stress --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

**Sem uv:**

```bash
python -m main --mode stress --config configs/p2_dsb2018.yaml --checkpoint outputs/p2/best.pth
```

Das três opções do enunciado, escolhemos **corrupções**: borrão, ruído e brilho/contraste, em
3 intensidades cada. São 10 condições (a imagem limpa mais 3 × 3) sobre as imagens de
validação, com o mesmo modelo e os mesmos pesos — muda só a imagem que entra. Os rótulos nunca
são corrompidos. As corrupções são aplicadas na imagem **já em 256²**, o espaço de entrada do
modelo: o que se testa aqui é o modelo, não o pipeline de dados.

| família | fórmula | nível 1 | nível 2 | nível 3 |
|---|---|---|---|---|
| borrão | `gaussian_filter(img, σ)` | σ = 1 px | σ = 2 px | σ = 4 px |
| ruído | `img + N(0, σ)`, cortado em [0,1] | σ = 0,02 | σ = 0,05 | σ = 0,10 |
| brilho/contraste | `(img − 0,5)·(1−f) + 0,5 + f/2` | f = 0,2 | f = 0,4 | f = 0,6 |

O modelo foi treinado **sem augmentation nenhum** — não há transform, flip ou rotação em
lugar nenhum do `src/` — então ele nunca viu estas corrupções durante o treino.

Saídas em `outputs/p6/`: `summary.json` e `per_image.json` com as métricas por condição e por
imagem, `p6_degradacao.png` (curva de mAP/IoU/erro de contagem por severidade, por família de
corrupção) e `p6_exemplos.png` (a mesma imagem sob as três famílias, em cada intensidade). A
semente de corrupção é fixa por imagem, então rodar de novo reproduz o mesmo `summary.json`.
