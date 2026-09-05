"""Split estratificado por modalidade.

O enunciado (seção 2) exige: *"split treino/validação/teste estratificado por modalidade
ou cidade, justificado na apresentação"*.

Estratificar significa dividir de forma que **cada conjunto preserve a proporção de cada
modalidade** do conjunto original, em vez de embaralhar tudo e cortar.

Por que importa aqui: das 670 amostras, apenas **16 são brightfield**. Num sorteio simples
de 20% para validação, essas 16 podem cair quase todas de um lado — a validação pode
terminar com zero exemplos dessa modalidade, ou com o dobro do esperado. Isso tem duas
consequências ruins:

1. O mAP reportado passa a depender da seed, porque a composição do conjunto muda junto.
2. As ablações da Parte 3 pedem 2 seeds com média ± desvio. Se a composição varia com a
   seed, o desvio medido é a variação de *qual dado caiu onde*, e não o efeito da mudança
   que se quer avaliar — que é justamente o que a ablação deveria isolar.

Estratificando, a composição fica fixa e a seed passa a variar só o que interessa.
"""

from pathlib import Path

import numpy as np

from src.data.modality import classify_samples


def stratified_split(
    sample_dirs: list[Path],
    val_fraction: float = 0.2,
    seed: int = 0,
    modalities: dict[Path, str] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Divide as amostras preservando a proporção de cada modalidade.

    Args:
        sample_dirs: todos os diretórios de amostra disponíveis.
        val_fraction: fração destinada à validação.
        seed: semente do embaralhamento; o mesmo valor sempre produz o mesmo split.
        modalities: classificação pronta, para não reclassificar. Se None, classifica.

    Returns:
        Tupla (treino, validação), ambas ordenadas para o resultado ser determinístico.
    """
    if modalities is None:
        modalities = classify_samples(sample_dirs)

    por_modalidade: dict[str, list[Path]] = {}
    for sample_dir in sample_dirs:
        por_modalidade.setdefault(modalities[Path(sample_dir)], []).append(Path(sample_dir))

    train: list[Path] = []
    val: list[Path] = []

    # Embaralha e corta DENTRO de cada modalidade, e só então junta. É isso que garante
    # a mesma proporção nos dois lados.
    for modalidade in sorted(por_modalidade):
        grupo = sorted(por_modalidade[modalidade])
        rng = np.random.default_rng(seed)
        rng.shuffle(grupo)

        # arredonda para cima: com 16 amostras e 20%, garante 4 na validação em vez de 3
        n_val = int(np.ceil(len(grupo) * val_fraction))
        val.extend(grupo[:n_val])
        train.extend(grupo[n_val:])

    return sorted(train), sorted(val)


def split_summary(
    train: list[Path], val: list[Path], modalities: dict[Path, str]
) -> str:
    """Tabela com a composição dos dois conjuntos, para conferir e para a apresentação."""
    todas = sorted({m for m in modalities.values()})
    linhas = [f"{'modalidade':<16}{'treino':>10}{'validação':>12}{'% val':>8}"]
    for modalidade in todas:
        n_train = sum(1 for d in train if modalities[Path(d)] == modalidade)
        n_val = sum(1 for d in val if modalities[Path(d)] == modalidade)
        total = n_train + n_val
        linhas.append(
            f"{modalidade:<16}{n_train:>10}{n_val:>12}{100*n_val/total:>7.1f}%"
        )
    linhas.append(
        f"{'TOTAL':<16}{len(train):>10}{len(val):>12}"
        f"{100*len(val)/(len(train)+len(val)):>7.1f}%"
    )
    return "\n".join(linhas)
