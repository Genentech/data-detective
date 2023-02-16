from typing import Dict, Union

import pandas as pd
import torch
from torchvision.datasets import CIFAR10

from constants import FloatTensor
from src.enums.enums import DataType


class MyCIFAR10(CIFAR10):
    def __getitem__(self, idx: Union[int, slice, list]) -> Dict[str, Union[FloatTensor, int]]:
        #todo: figure ouot why this is even necessary.
        if isinstance(idx, list):
            attr_dict = pd.DataFrame([self.__getitem__(index) for index in idx]).to_dict()
            for k, v in attr_dict.items():
                try:
                    attr_dict[k] = torch.stack(tuple(v.values()))
                except TypeError:
                    attr_dict[k] = torch.Tensor(tuple(v.values())).reshape((len(idx), -1))
            return attr_dict

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
