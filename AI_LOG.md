# AI_LOG

Registro de como usei ferramentas de IA neste assignment, conforme pedido no enunciado.

Ferramenta principal: Claude (Claude Code no terminal, e o app para algumas leituras).

---

## Como usamos

Usamos IA principalmente para **geração de código**, não para decidir o que fazer. Em quase
todas as partes a sequência foi a mesma: nós decidíamos o quê (a representação de saída, a
perda, o que uma função precisava fazer, quais eixos ablacionar), pedíamos para o Claude
gerar uma primeira versão, e então revisávamos — rodando os testes, lendo o diff, e voltando
com pedidos de correção ou de reestruturação quando o resultado não estava do jeito que
queríamos.

O caso mais claro disso foi logo no início (Partes 0 e 1): pedimos uma versão inicial
**inteira** gerada por IA, para validar rápido que a ideia geral funcionava fim-a-fim. Ela
funcionava, mas o código estava tudo meio emaranhado — treino, avaliação, dataset e modelo
mal separados, hiperparâmetros espalhados pelo código. Em vez de continuar empilhando
funcionalidade em cima disso, paramos, analisamos a estrutura e passamos a pedir mudanças
**incrementais de arquitetura**: separar as engines de treino e de avaliação, criar
factories para os dataloaders, mover os hiperparâmetros de cada experimento para arquivos de
configuração (YAML) em vez de constantes no código, transformar as métricas em classes
`torch.nn.Module`. Esse ciclo de "gerar → analisar → pedir reestruturação" se repetiu, em
menor escala, nas partes seguintes.

## Episódios

### Estudo do material antes de codar

Antes de escrever qualquer código, usei o Claude para percorrer o deck de Segmentação
Semântica da disciplina, para entender sobre o conteúdo.
O formato que funcionou foi: ele explicava um bloco com exemplos numéricos concretos e depois
me fazia perguntas de checagem.

### Partes 0 e 1 — versão inicial de IA, depois parar e reestruturar

Pedimos ao Claude uma implementação completa da Parte 0 (dataset sintético + treino +
avaliação) e, na sequência, da Parte 1 (U-Net binária + extração por componentes conexos)
gerada inteiramente por IA, só para ter algo rodando rápido e confirmar que a ideia da
disciplina (segmentação semântica → instâncias por pós-processamento) era viável no nosso
código antes de investir tempo em arquitetura.

Depois disso paramos de pedir feature nova e passamos duas sessões só analisando o que tinha
sido gerado. O diagnóstico: tudo funcionava, mas `main.py` fazia treino, avaliação e parsing
de argumento no mesmo lugar, o dataset sintético e o DSB2018 tinham cada um seu próprio
caminho de carregamento copiado e colado, e os hiperparâmetros (lr, épocas, arquitetura)
estavam hardcoded em vários pontos.

A partir daí, os pedidos passaram a ser específicos de arquitetura, um de cada vez:
"separa o loop de treino do de avaliação em módulos próprios", "cria uma factory que decide
qual dataset instanciar a partir do config", "os hiperparâmetros de cada experimento têm que
vir de um YAML, não do código", "transforma o cálculo de IoU/Dice em `nn.Module` reutilizável
em vez de função solta". Isso é o que virou `src/core` (config), `src/data` (factories),
`src/training`/`src/evaluation` (engines) e `src/metrics`.

### Parte 2 (Trilha C) — implementação depois da decisão de design

A escolha da representação (heatmap de centros + offsets, em vez das trilhas A ou B) e do
formato da perda foi decisão nossa, não sugestão da IA. Depois de decidido, usamos o Claude
para implementar a cabeça de rede (`src/models/heads.py`), as perdas correspondentes e uma
primeira versão da decodificação (máximos locais do heatmap + atribuição por ponto
deslocado), sempre acompanhado dos testes que validam a decodificação com casos sintéticos
pequenos (poucos pixels, resultado computável à mão) antes de rodar no dataset real.

### Parte 3 (ablações) — infraestrutura, não a escolha dos eixos

Quais dois eixos comparar (recuperação de resolução e contexto global) e por que a PSPNet
pura não serve para o Eixo 3 foi decisão nossa, documentada no README. O que pedimos à IA foi
a parte mecânica e repetitiva: o runner que roda N configs × M seeds, agrega média e desvio
populacional, e gera os gráficos de barra e a tabela comparativa (`src/ablation/`).

### Partes 4, 5 e 6 — mesmo padrão

Para inferência em mosaico (Parte 4), galeria de falhas (Parte 5) e teste de estresse por
corrupções (Parte 6), o padrão se repetiu: definimos o que cada parte precisava calcular (as
estratégias de costura a comparar, o critério de classificação do modo de falha, as fórmulas
e intensidades de corrupção), e usamos IA para gerar a implementação, os testes e as figuras
— revisando cada uma antes de aceitar, e voltando com pedido de ajuste quando a leitura do
resultado não batia com o que o diagnóstico esperava (por exemplo, reajustar como a Parte 5
decide se uma imagem "fundiu" ou "fragmentou" objetos).

### Depuração de ambiente e configuração

Vários episódios menores de troubleshooting: erros de configuração do `uv` no projeto, ajuste
de YAMLs de config que quebravam o carregamento, correção de erros ao rodar a Parte 2/3 em
sequência. Nesses casos o uso foi bem mecânico — colar o erro, pedir o diagnóstico e a
correção pontual — sem decisão de design envolvida.

---

## Divisão de trabalho

A IA escreveu a maior parte do código-fonte, mas sempre a partir de uma decisão nossa sobre
**o quê** fazer — em especial as três coisas que o enunciado pede como autoria da dupla em
cada parte: a representação de saída, a perda que otimiza isso, e como decodificar a previsão
em objetos. Isso vale tanto para a decisão de design (qual trilha da Parte 2 escolher, quais
eixos ablacionar na Parte 3, o que testar na Parte 6) quanto para validar que a implementação
gerada está correta: a métrica de instância e a regra de matching, por exemplo, têm testes
escritos e revisados por nós especificamente para pegar erros sutis de implementação (matching
guloso vs. Hungarian, off-by-one no threshold de IoU), e não foram aceitas só porque rodavam
sem erro.
