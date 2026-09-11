"""Parte 4 — inferência em mosaico: imagem grande, tiles sobrepostos e costura."""

from src.mosaic.build import montar_mosaico, montar_mosaicos
from src.mosaic.tiling import grade_tiles, mapa_de_donos, peso_interior, posicoes
from src.mosaic.stitch import (
    costura_densa,
    costura_ingenua,
    fusao_por_iou,
    inferir_inteira,
    inferir_tiles,
    media_densa,
    separar,
)

__all__ = [
    "montar_mosaico",
    "montar_mosaicos",
    "grade_tiles",
    "mapa_de_donos",
    "peso_interior",
    "posicoes",
    "costura_densa",
    "costura_ingenua",
    "fusao_por_iou",
    "inferir_inteira",
    "inferir_tiles",
    "media_densa",
    "separar",
]
