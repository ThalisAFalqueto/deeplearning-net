"""Configuração de treino como dataclass."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int
    data: dict
    model: dict
    train: dict
    decode: dict
    output_dir: Path
    # onde caem os gráficos densidade/fusão. Runs normais usam o padrão; a ablação
    # sobrescreve para a pasta da seed, mantendo tudo sob outputs/ablation/<timestamp>.
    figures_dir: Path = Path("outputs/figures")

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrainConfig":
        return cls(
            seed=cfg["seed"],
            data=cfg["data"],
            model=cfg["model"],
            train=cfg["train"],
            decode=cfg["decode"],
            output_dir=Path(cfg["output_dir"]),
            figures_dir=Path(cfg.get("figures_dir", "outputs/figures")),
        )
