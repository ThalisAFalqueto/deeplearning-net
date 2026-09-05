"""Constroi dataloaders de treino e validação a partir da configuração."""

from abc import ABC, abstractmethod

from src.training.config import TrainConfig
from src.core.config import AppConfig


class DatasetFactory(ABC):
    @abstractmethod
    def build_train(self, cfg: TrainConfig):
        pass

    @abstractmethod
    def build_val(self, cfg: TrainConfig):
        pass


class SyntheticDatasetFactory(DatasetFactory):
    def build_train(self, cfg: TrainConfig):
        from .synthetic import SyntheticEllipses
        d = cfg.data
        return SyntheticEllipses(
            n_samples=d["n_train"], size=d["size"], seed=cfg.seed,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )

    def build_val(self, cfg: TrainConfig):
        from .synthetic import SyntheticEllipses
        d = cfg.data
        return SyntheticEllipses(
            n_samples=d["n_val"], size=d["size"], seed=cfg.seed + 777,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )


class DSB2018DatasetFactory(DatasetFactory):
    """Constrói os conjuntos do DSB2018.

    Com ``data.stratify: true`` no config, ignora a separação em disco entre ``train_dir``
    e ``val_dir``: junta todas as amostras e refaz a divisão estratificada por modalidade
    (ver ``src/data/split.py``). Os arquivos não são movidos — a divisão é lógica, o que
    a torna reproduzível pela seed e barata de mudar.
    """

    def __init__(self):
        self._cache = {}

    def _all_sample_dirs(self, cfg: TrainConfig) -> list:
        from pathlib import Path
        import os

        root = Path(os.getcwd())
        dirs = []
        for chave in ("train_dir", "val_dir"):
            caminho = cfg.data.get(chave)
            if caminho:
                dirs.extend(d for d in (root / caminho).iterdir() if d.is_dir())
        return sorted(dirs)

    def _split(self, cfg: TrainConfig):
        """Calcula (ou reaproveita) o split estratificado desta configuração."""
        from .modality import classify_samples
        from .split import stratified_split

        chave = (cfg.data.get("train_dir"), cfg.data.get("val_dir"),
                 cfg.data.get("val_fraction", 0.2), cfg.seed)
        if chave in self._cache:
            return self._cache[chave]

        amostras = self._all_sample_dirs(cfg)
        # a classificação lê as 670 imagens uma vez; guardada junto com o split para
        # que build_train e build_val não repitam o trabalho
        modalidades = classify_samples(amostras)
        train, val = stratified_split(
            amostras,
            val_fraction=cfg.data.get("val_fraction", 0.2),
            seed=cfg.seed,
            modalities=modalidades,
        )
        self._cache[chave] = (train, val, modalidades)
        return self._cache[chave]

    def build_train(self, cfg: TrainConfig):
        from .dsb2018 import DSB2018
        if cfg.data.get("stratify", False):
            train, _, _ = self._split(cfg)
            return DSB2018(size=cfg.data["size"], sample_dirs=train)
        return DSB2018(data_dir=cfg.data["train_dir"], size=cfg.data["size"])

    def build_val(self, cfg: TrainConfig):
        from .dsb2018 import DSB2018
        if cfg.data.get("stratify", False):
            _, val, _ = self._split(cfg)
            return DSB2018(size=cfg.data["size"], sample_dirs=val)
        return DSB2018(data_dir=cfg.data["val_dir"], size=cfg.data["size"])

    def split_report(self, cfg: TrainConfig) -> str:
        """Tabela da composição do split — para conferir e para a apresentação."""
        from .split import split_summary
        train, val, modalidades = self._split(cfg)
        return split_summary(train, val, modalidades)


class DatasetFactoryRegistry:
    _factories = {
        "synthetic": SyntheticDatasetFactory(),
        "dsb2018": DSB2018DatasetFactory(),
    }

    @classmethod
    def get(cls, app_config: AppConfig) -> DatasetFactory:
        kind = app_config.get_train_config().data["kind"]
        if kind not in cls._factories:
            raise ValueError(f"data.kind desconhecido: {kind}")
        return cls._factories[kind]