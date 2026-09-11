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
outputs/        figuras e métricas (versionadas); checkpoints (fora do git)
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

Ativado por `stratify: true` em `configs/default.yaml`. A divisão é lógica: nenhum arquivo é
movido, e a classificação de modalidade fica em cache (`.modality_cache.json`).

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
