import typing
from typing import Dict, Union

from torchvision.datasets import FashionMNIST

from constants import FloatTensor
from src.enums.enums import DataType

class MyFashionMNIST(FashionMNIST):
    def __getitem__(self, idx: int) -> typing.Dict[str, Union[FloatTensor, int]]:
        sample = super().__getitem__(idx)
        return {
            "image": sample[0],
            "label": sample[1],
        }

    def datatypes(self) -> typing.Dict[str, DataType]:
        return {
            "image": DataType.IMAGE,
            "label": DataType.CATEGORICAL,
        }
