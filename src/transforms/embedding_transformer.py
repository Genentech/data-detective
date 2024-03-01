import os
import pickle

import joblib
import numpy as np
import torch
import typing

from src.datasets.data_detective_dataset import DataDetectiveDataset
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
        dataset: DataDetectiveDataset,
        transforms: typing.Dict[str, typing.List[Transform]]
    ):
        self.dataset = dataset
        self.transforms = transforms
        self.cache_statistics_dict = {
            'cache_misses': 0,
            'cache_hits': 0,
        }

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

                hash_value = TransformedDataset.hash_column(col_name, new_item[col_name], new_column_name)

                filepath = f"data/tmp/{hash_value}.pkl"
                if os.path.isfile(filepath):
                    try: 
                        with open(filepath, "rb") as f:
                            new_value = pickle.load(f)
                            new_item[new_column_name] = new_value
                            self.cache_statistics_dict['cache_hits'] += 1
                    except: 
                        if not os.path.isdir("data/tmp"):
                            os.makedirs("data/tmp")

                        new_value = transform(new_item[col_name])
                        with open(filepath, "wb") as f:
                            self.cache_statistics_dict['cache_misses'] += 1
                            pickle.dump(new_value, f)
                            new_item[new_column_name] = new_value
                else:
                    if not os.path.isdir("data/tmp"):
                        os.makedirs("data/tmp")

                    new_value = transform(new_item[col_name])
                    with open(filepath, "wb") as f:
                        self.cache_statistics_dict['cache_misses'] += 1
                        pickle.dump(new_value, f)
                        new_item[new_column_name] = new_value

            if all([transform.in_place for transform in transform_list]):
                new_item.pop(col_name)

        return new_item

    def __len__(self):
        return self.dataset.__len__()

    @staticmethod
    def hash_column(col_name, col_val, transform_name):
        if hasattr(col_val, "numpy"):
            col_val = col_val.numpy()

        return joblib.hash((
            col_name,
            col_val,
            transform_name,
        ))