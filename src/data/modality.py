"""Detecção da modalidade de imagem do DSB2018.

O dataset mistura tipos de microscopia visualmente muito diferentes, e essa informação
não vem rotulada — é inferida da própria imagem.

Medido sobre as 670 amostras de ``stage1_train``:

    fluorescência   546  (81,5%)   cinza, fundo escuro
    histologia      108  (16,1%)   colorida (H&E), fundo claro
    brightfield      16  ( 2,4%)   cinza, fundo claro

Duas estatísticas bastam para separar os três grupos, e a separação é limpa — não há
nenhuma amostra em faixa ambígua:

    saturação = média de (max(R,G,B) - min(R,G,B))
        562 amostras têm saturação < 1 (cinza puro) e 108 têm > 10.
        NENHUMA cai entre 1 e 10.

    brilho = mediana dos pixels
        as cinza escuras ficam entre 0 e ~8; as claras acima de 190.

Isto é lido da imagem COLORIDA original. O ``DSB2018.__getitem__`` converte para escala
de cinza para treinar, o que é uma decisão separada e não interfere aqui.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

FLUORESCENCE = "fluorescence"
HISTOLOGY = "histology"
BRIGHTFIELD = "brightfield"

# Limiares no meio dos vazios observados na distribuição: a saturação salta de <1 para
# >10 sem valores intermediários, e o brilho mediano de <10 para >190.
SATURATION_THRESHOLD = 5.0
BRIGHTNESS_THRESHOLD = 100.0


def image_statistics(image_path: Path) -> tuple[float, float]:
    """Calcula saturação média e brilho mediano de uma imagem.

    Args:
        image_path: caminho do PNG original (lido em RGB).

    Returns:
        Tupla (saturação, brilho). Saturação 0 significa cinza puro.
    """
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    saturation = float((img.max(axis=2) - img.min(axis=2)).mean())
    brightness = float(np.median(img))
    return saturation, brightness


def detect_modality(sample_dir: Path) -> str:
    """Classifica uma amostra em uma das três modalidades.

    Args:
        sample_dir: diretório da amostra (contendo ``images/<id>.png``).

    Returns:
        Uma das constantes FLUORESCENCE, HISTOLOGY ou BRIGHTFIELD.
    """
    sample_dir = Path(sample_dir)
    saturation, brightness = image_statistics(
        sample_dir / "images" / f"{sample_dir.name}.png"
    )

    if saturation >= SATURATION_THRESHOLD:
        return HISTOLOGY
    return FLUORESCENCE if brightness < BRIGHTNESS_THRESHOLD else BRIGHTFIELD


def classify_samples(
    sample_dirs: list[Path], cache_path: Path | None = None
) -> dict[Path, str]:
    """Classifica várias amostras de uma vez.

    Percorrer as 670 imagens leva cerca de 10 segundos. Como a modalidade é uma
    propriedade fixa do arquivo, o resultado é guardado em disco e reaproveitado — o que
    importa nas ablações da Parte 3, onde o treino roda uma dezena de vezes.

    Args:
        sample_dirs: diretórios das amostras.
        cache_path: onde guardar o resultado. Se None, usa ``.modality_cache.json`` na
            pasta que contém as amostras.

    Returns:
        Dicionário {diretório: modalidade}.
    """
    sample_dirs = [Path(d) for d in sample_dirs]
    if not sample_dirs:
        return {}

    if cache_path is None:
        cache_path = sample_dirs[0].parent.parent / ".modality_cache.json"
    cache_path = Path(cache_path)

    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}  # cache corrompido: refaz do zero

    resultado, novos = {}, False
    for d in sample_dirs:
        chave = str(d)
        if chave not in cache:
            cache[chave] = detect_modality(d)
            novos = True
        resultado[d] = cache[chave]

    if novos:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))

    return resultado


def modality_counts(modalities: dict[Path, str]) -> dict[str, int]:
    """Conta quantas amostras há de cada modalidade, em ordem decrescente."""
    counts: dict[str, int] = {}
    for modality in modalities.values():
        counts[modality] = counts.get(modality, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
