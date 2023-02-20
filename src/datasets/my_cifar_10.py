from typing import Dict, Union

import pandas as pd
import torch
from torchvision.datasets import CIFAR10

from constants import FloatTensor
from src.enums.enums import DataType


class MyCIFAR10(CIFAR10):
    def __getitem__(self, idx: Union[int, slice, list]) -> Dict[str, Union[FloatTensor, int]]:
        sample = super().__getitem__(idx)
        return {
            "cifar_image": sample[0],
            "label": sample[1],
        }

    def datatypes(self) -> Dict[str, DataType]:
        return {
            "cifar_image": DataType.IMAGE,
            "label": DataType.CATEGORICAL,
        }
