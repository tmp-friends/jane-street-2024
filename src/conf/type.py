from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    data_dir: str


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class TrainConfig:
    dir: DirConfig
    num_epochs: int
    train_batch_size: int
    valid_batch_size: int
    lr: float
    num_accumulates: int
    early_stopping_steps: int


@dataclass
class InferConfig:
    dir: DirConfig
    model: ModelConfig
    num_folds: int
    model_dir: str
    test_batch_size: int
