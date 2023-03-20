from typing import Dict, Union

import pandas as pd
import torch
from torchvision.datasets import CIFAR10

from constants import FloatTensor
from src.enums.enums import DataType


class MyCIFAR10(CIFAR10):
    def __getitem__(self, idx: Union[int, slice, list]) -> Dict[str, Union[FloatTensor, int]]:
        """
        Returns an item from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the image and the label.
        """
        sample = super().__getitem__(idx)
        return {
            "cifar_image": sample[0],
            "label": sample[1],
        }

    def datatypes(self) -> Dict[str, DataType]:
        """
        Gives the datatypes of a dataset sample.
        @return: the datatypes of a dataset sample.
        """
        return {
            "cifar_image": DataType.IMAGE,
            "label": DataType.CATEGORICAL,
        }
