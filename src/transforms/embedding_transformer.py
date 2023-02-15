import types
import typing
from abc import abstractmethod

import torch
import torchvision.transforms
from torchvision.transforms import GaussianBlur

from src.enums.enums import DataType


class Transform(torch.nn.Module):
    def __init__(self, transform_class, new_column_name_fn: typing.Callable[[str], str], new_column_datatype: DataType,
                 in_place: bool = False):
        super().__init__()
        self.transform_class = transform_class
        self.new_column_name_fn = new_column_name_fn
        self.new_column_datatype = new_column_datatype
        self.in_place = in_place

    def initialize_transform(self, transform_kwargs):
        self.transform = self.transform_class(**transform_kwargs)

    #TODO: add a "fit" method to accommodate transforms that need to be fit.

    def forward(self, obj):
        if not self.transform:
            raise Exception("Transform not initialized before use.")

        return self.transform(obj)


class TransformedDataset:
    def __init__(self,
        dataset: torch.utils.data.Dataset,
        transforms: typing.Dict[str, typing.List[Transform]]
    ):
        self.dataset = dataset
        self.transforms = transforms

    def datatypes(self):
        dataset = self.dataset
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        new_datatypes = dataset.datatypes()

        for col_name, transform_list in self.transforms.items():
            if all([transform.in_place for transform in transform_list]):
                new_datatypes.pop(col_name)

            for transform in transform_list:
                new_column_name_fn = transform.new_column_name_fn
                new_column_name = new_column_name_fn(col_name)
                new_datatype = transform.new_column_datatype
                new_datatypes[new_column_name] = new_datatype

        return new_datatypes

    def __getitem__(self, item):
        new_item = self.dataset[item]
        for col_name, transform_list in self.transforms.items():
            for transform in transform_list:
                new_column_name_fn = transform.new_column_name_fn
                new_column_name = new_column_name_fn(col_name)
                new_value = transform(new_item[col_name])
                new_item[new_column_name] = new_value

            if all([transform.in_place for transform in transform_list]):
                new_item.pop(col_name)

        return new_item

    def __getattr__(self, item):
        return getattr(self.dataset, item) or getattr(self, item)

    def __len__(self):
        return self.dataset.__len__()


