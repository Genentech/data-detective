import os
import pickle
from collections import defaultdict

import joblib
import numpy as np
import torch
import typing

from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType


class Transform(torch.nn.Module):
    def __init__(self, transform_class, new_column_name_fn: typing.Callable[[str], str], new_column_datatype: DataType,
                 in_place: bool = False, cache_values: bool = True):
        super().__init__()
        self.transform_class = transform_class
        self.new_column_name_fn = new_column_name_fn
        self.new_column_datatype = new_column_datatype
        self.in_place = in_place
        self.cache_values = cache_values

        if self.cache_values:
            self.cache_statistics_dict = defaultdict(lambda: 0)

    def hash_transform_value(self, val):
        if hasattr(val, "numpy"):
            val = val.numpy()

        #todo: add assertion that no two transforms have the same name
        transform_name = self.new_column_name_fn("")

        return joblib.hash((
            val,
            transform_name,
            self.options,
        ))

    def initialize_transform(self, transform_kwargs):
        self.options = transform_kwargs
        self.transform = self.transform_class(**transform_kwargs)

    #TODO: add a "fit" method to accommodate transforms that need to be fit.

    def forward(self, obj):
        if not hasattr(self, "transform"):
            raise Exception("Transform not initialized before use.")

        hash_value = self.hash_transform_value(obj)

        filepath = f"data/tmp/{hash_value}.pkl"
        if os.path.isfile(filepath):
            try: 
                with open(filepath, "rb") as f:
                    transformed_value = pickle.load(f)
                    self.cache_statistics_dict['cache_hits'] += 1
            except Exception: 
                transformed_value = self.transform(obj)
                with open(filepath, "wb") as f:
                    pickle.dump(transformed_value, f)
                    self.cache_statistics_dict['cache_misses'] += 1
        else:
            # import pdb; pdb.set_trace()
            if not os.path.isdir("data/tmp"):
                os.makedirs("data/tmp")

            transformed_value = self.transform(obj)
            with open(filepath, "wb") as f:
                pickle.dump(transformed_value, f)
                self.cache_statistics_dict['cache_misses'] += 1

        return transformed_value

class TransformedDataset(DataDetectiveDataset):
    def __init__(self,
        dataset: DataDetectiveDataset,
        transforms: typing.Dict[str, typing.List[Transform]]
    ):
        self.dataset = dataset
        self.include_subject_id_in_data = self.dataset.include_subject_id_in_data
        self.show_id = self.dataset.show_id
        self.index_df = self.dataset.index_df
        
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
                new_item[new_column_name] = transform(new_item[col_name])

            if all([transform.in_place for transform in transform_list]):
                new_item.pop(col_name)

        return new_item

    def __len__(self):
        return self.dataset.__len__()

    # @staticmethod
    # def hash_column(col_name, col_val, transform_name):
    #     if hasattr(col_val, "numpy"):
    #         col_val = col_val.numpy()
    #
    #     return joblib.hash((
    #         # col_name,
    #         col_val,
    #         transform_name,
    #     ))