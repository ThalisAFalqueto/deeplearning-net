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
    def build_train(self, cfg: TrainConfig):
        from .dsb2018 import DSB2018
        return DSB2018(data_dir=cfg.data["train_dir"], size=cfg.data["size"])

    def build_val(self, cfg: TrainConfig):
        from .dsb2018 import DSB2018
        return DSB2018(data_dir=cfg.data["val_dir"], size=cfg.data["size"])


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