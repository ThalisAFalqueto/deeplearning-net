"""Constroi dataloaders de treino e validação a partir da configuração."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

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
        from src.data.synthetic import SyntheticEllipses
        d = cfg.data
        return SyntheticEllipses(
            n_samples=d["n_train"], size=d["size"], seed=cfg.seed,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )

    def build_val(self, cfg: TrainConfig):
        from src.data.synthetic import SyntheticEllipses
        d = cfg.data
        return SyntheticEllipses(
            n_samples=d["n_val"], size=d["size"], seed=cfg.seed + 777,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )


class DSB2018DatasetFactory(DatasetFactory):
    def build_train(self, cfg: TrainConfig):
        from src.data.dsb2018 import DSB2018
        return DSB2018(data_dir=cfg.data["train_dir"], size=cfg.data["size"])

    def build_val(self, cfg: TrainConfig):
        from src.data.dsb2018 import DSB2018
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


class DataPipeline:
    def __init__(self, app_config: AppConfig):
        self.cfg = app_config.get_train_config()
        self.factory = DatasetFactoryRegistry.get(app_config)

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        t = self.cfg.train

        train_ds = self.factory.build_train(self.cfg)
        val_ds = self.factory.build_val(self.cfg)

        train_loader = DataLoader(
            train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"]
        )
        val_loader = DataLoader(
            val_ds, batch_size=t["batch_size"], num_workers=t["num_workers"]
        )

        return train_loader, val_loader
