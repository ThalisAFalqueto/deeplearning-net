"""Configuração de avaliação como dataclass."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalConfig:
    seed: int
    data: dict
    model: dict
    train: dict
    decode: dict
    output_dir: Path

    @classmethod
    def from_dict(cls, cfg: dict) -> "EvalConfig":
        return cls(
            seed=cfg["seed"],
            data=cfg["data"],
            model=cfg["model"],
            train=cfg["train"],
            decode=cfg["decode"],
            output_dir=Path(cfg["output_dir"]),
        )
