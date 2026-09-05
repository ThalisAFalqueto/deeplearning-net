from typing import Tuple
from torch.utils.data import DataLoader

from ..core.config import AppConfig
from .factory import DatasetFactoryRegistry


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
