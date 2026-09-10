from src.losses.center_offset import CenterOffsetLoss, CenterOffsetTask
from src.losses.bce import BCELoss
from src.losses.factory import (
    LossFactory,
    LossFactoryRegistry,
    resolve_task_name,
)

__all__ = [
    "CenterOffsetLoss",
    "CenterOffsetTask",
    "BCELoss",
    "LossFactory",
    "LossFactoryRegistry",
    "resolve_task_name",
]
