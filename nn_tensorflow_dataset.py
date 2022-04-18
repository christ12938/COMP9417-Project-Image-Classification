from dataclasses import dataclass
from typing import Any


@dataclass
class Dataset:
    train_set: Any
    valid_set: Any
    class_weights: dict[int, float]
