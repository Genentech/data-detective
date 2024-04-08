import os
import pickle
from collections import defaultdict
from cachetools import LRUCache
import time

import joblib
import numpy as np
import torch
import typing

from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType


class PickleableResnetTransform(torch.nn.Module):
    cache = LRUCache(maxsize=50000)

    def __init__(self, in_place: bool = False, cache_values: bool = True):
        super().__init__()
        self.new_column_datatype = DataType.MULTIDIMENSIONAL
        self.in_place = in_place
        self.cache_values = cache_values

        # if self.cache_values:
        #     self.cache_statistics_dict = defaultdict(lambda: 0)

    def new_column_name_fn(self, old_name): 
        return f"resnet50_{old_name}"

    def transform(self):
        if len(x.shape) == 2:
            # add channel dimension
            x = torch.unsqueeze(x, 0)
        if len(x.shape) == 3:
            # need a 4th dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[1] == 1:
            # if 1ch need from 1ch to 3ch RGB
            x = x.expand(x.shape[0], 3, *x.shape[2:])
        x = self.backbone(x)
        x = x.squeeze()
        x = x.reshape((-1, 2048))
        x = x.detach().numpy()
        return x

    def hash_transform_value(self, id=None, col_name=None):
        # if hasattr(val, "numpy"):
        #     val = val.numpy()

        #todo: add assertion that no two transforms have the same name
        transform_name = self.new_column_name_fn("")
        
        return joblib.hash((
            id, 
            col_name,
            transform_name,
            {k: v for k, v in self.options.items() if k not in ["column", "data_object"]},
        ))

    def initialize_transform(self, transform_kwargs):
        self.options = transform_kwargs
        import torchvision.models

        transform_kwargs.pop("data_object")
        transform_kwargs.pop("column")

        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2, **transform_kwargs
        )
        modules = list(resnet.children())[:-1]
        self.backbone = torch.nn.Sequential(torch.nn.Upsample((224, 224)), *modules)


    #TODO: add a "fit" method to accommodate transforms that need to be fit.

    def forward(self, dataset, item, col_name):
        ### this takes most of the time
        if not hasattr(self, "transform"):
            raise Exception("Transform not initialized before use.")

        hash_value = self.hash_transform_value(id=dataset.get_sample_id(item), col_name=col_name)

        # start = time.time() 
        # if hash_value in self.cache: 
        #     transformed_value = self.cache.get(hash_value)
        #     print("cache hit")
        #     self.cache_statistics_dict['cache_hits'] += 1
        # else: 
        #     obj = dataset[item][col_name]
        #     transformed_value = self.transform(obj)
        #     self.cache[hash_value] = transformed_value
        #     print("cache miss")
        #     self.cache_statistics_dict['cache_miss'] += 1
        # end = time.time()
        # print(f"transforming took {1000 * (end - start)} ms")

        filepath = f"data/tmp/{hash_value}.pkl"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                # self.cache_statistics_dict['cache_hits'] += 1
                transformed_value = pickle.load(f)
        else:
            # import pdb; pdb.set_trace()
            if not os.path.isdir("data/tmp"):
                os.makedirs("data/tmp")

            obj = dataset[item][col_name]
            transformed_value = self.transform(obj)
            with open(filepath, "wb") as f:
                # self.cache_statistics_dict['cache_misses'] += 1
                pickle.dump(transformed_value, f)

        return transformed_value

class Transform(torch.nn.Module):
    cache = LRUCache(maxsize=50000)

    def __init__(self, transform_class, new_column_name_fn: typing.Callable[[str], str], new_column_datatype: DataType,
                 in_place: bool = False, cache_values: bool = True):
        super().__init__()
        self.transform_class = transform_class
        self.new_column_name_fn = new_column_name_fn
        self.new_column_datatype = new_column_datatype
        self.in_place = in_place
        self.cache_values = cache_values

        # if self.cache_values:
        #     self.cache_statistics_dict = defaultdict(lambda: 0)

    def hash_transform_value(self, id=None, col_name=None):
        # if hasattr(val, "numpy"):
        #     val = val.numpy()

        #todo: add assertion that no two transforms have the same name
        transform_name = self.new_column_name_fn("")
        
        return joblib.hash((
            id, 
            col_name,
            transform_name,
            {k: v for k, v in self.options.items() if k not in ["column", "data_object"]},
        ))

    def initialize_transform(self, transform_kwargs):
        self.options = transform_kwargs
        self.transform = self.transform_class(**transform_kwargs)

    #TODO: add a "fit" method to accommodate transforms that need to be fit.

    def forward(self, dataset, item, col_name):
        ### this takes most of the time
        if not hasattr(self, "transform"):
            raise Exception("Transform not initialized before use.")

        hash_value = self.hash_transform_value(id=dataset.get_sample_id(item), col_name=col_name)

        # start = time.time() 
        # if hash_value in self.cache: 
        #     transformed_value = self.cache.get(hash_value)
        #     print("cache hit")
        #     self.cache_statistics_dict['cache_hits'] += 1
        # else: 
        #     obj = dataset[item][col_name]
        #     transformed_value = self.transform(obj)
        #     self.cache[hash_value] = transformed_value
        #     print("cache miss")
        #     self.cache_statistics_dict['cache_miss'] += 1
        # end = time.time()
        # print(f"transforming took {1000 * (end - start)} ms")

        filepath = f"data/tmp/{hash_value}.pkl"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                # self.cache_statistics_dict['cache_hits'] += 1
                transformed_value = pickle.load(f)
        else:
            # import pdb; pdb.set_trace()
            if not os.path.isdir("data/tmp"):
                os.makedirs("data/tmp")

            obj = dataset[item][col_name]
            transformed_value = self.transform(obj)
            with open(filepath, "wb") as f:
                # self.cache_statistics_dict['cache_misses'] += 1
                pickle.dump(transformed_value, f)

        return transformed_value

class TransformedDataset(DataDetectiveDataset):
    def __init__(self,
        dataset: DataDetectiveDataset,
        transforms: typing.Dict[str, typing.List[Transform]],
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
        start_time = time.time()

        new_item = self.dataset[item]

        for col_name, transform_list in self.transforms.items():
            for transform in transform_list:
                new_column_name_fn = transform.new_column_name_fn
                new_column_name = new_column_name_fn(col_name)
                # new_item[new_column_name] = transform(new_item[col_name])
                new_item[new_column_name] = transform(self.dataset, item, col_name)

            if all([transform.in_place for transform in transform_list]):
                new_item.pop(col_name)

        end_time = time.time()
        # Calculate the duration
        duration_seconds = end_time - start_time
        duration_milliseconds = duration_seconds * 1000

        # print(f"Execution time for transformed idx {item}:", duration_milliseconds, "milliseconds")
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