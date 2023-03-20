import typing
from typing import Dict, Union

from torchvision.datasets import FashionMNIST

from constants import FloatTensor
from src.enums.enums import DataType

class MyFashionMNIST(FashionMNIST):
    def __getitem__(self, idx: int) -> typing.Dict[str, Union[FloatTensor, int]]:
        """
        Returns an item from the fashionMNIST dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the image and the label.
        """
        sample = super().__getitem__(idx)
        return {
            "image": sample[0],
            "label": sample[1],
        }

    def datatypes(self) -> typing.Dict[str, DataType]:
        """
        Gives the datatypes of a FashionMNIST dataset sample.
        @return: the datatypes of a FashionMNIST dataset sample.
        """
        return {
            "image": DataType.IMAGE,
            "label": DataType.CATEGORICAL,
        }
