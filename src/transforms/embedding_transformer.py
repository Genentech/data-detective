import os
import pickle
from cachetools import LRUCache
import time

import joblib
import torch
import typing

from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType


class Transform(torch.nn.Module):
    cache = LRUCache(maxsize=50000)

    def dump_cache_to_disk():
        filepath = 'data/tmp/cache.pkl'
        # Ensure all directories in the path are created
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Open the file and write to it
        with open(filepath, 'wb') as f:
            pickle.dump(Transform.cache, f)
        
        print(f"Cache written to {filepath}")

    def load_cache_from_disk():
        filepath = 'data/tmp/cache.pkl'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                Transform.cache = pickle.load(f)
            print(f"Cache loaded from {filepath}")
        else:
            print(f"File {filepath} does not exist. Cache not loaded.")

    def __init__(self, new_column_datatype: DataType, in_place: bool = False, cache_values: bool = True):
        super().__init__()
        self.new_column_datatype = new_column_datatype
        self.in_place = in_place
        self.cache_values = cache_values

    def hash_transform_value(self, id=None, col_name=None):
        #todo: add assertion that no two transforms have the same name
        transform_name = self.new_column_name("")
        
        return joblib.hash((
            id, 
            col_name,
            transform_name,
            {k: v for k, v in self.options.items() if k not in ["column", "data_object"]},
        ))

    def hash_object(self, obj):
        if hasattr(obj, "numpy"):
            obj = obj.numpy()

        #todo: add assertion that no two transforms have the same name
        transform_name = self.new_column_name("")
        
        return joblib.hash((
            obj,
            transform_name,
            {k: v for k, v in self.options.items() if k not in ["column", "data_object"]},
        ))

    def initialize_transform(self, transform_kwargs):
        self.options = transform_kwargs

    def forward_item(self, obj):
        ### this takes most of the time
        if not hasattr(self, "transform"):
            raise Exception("Transform not initialized before use.")

        hash_value = self.hash_object(obj)

        # start = time.time() 
        # print(item)
        if hash_value in self.cache: 
            transformed_value = self.cache.get(hash_value)
        else: 
            transformed_value = self.transform(obj)
            self.cache[hash_value] = transformed_value

        return transformed_value

    #TODO: add a "fit" method to accommodate transforms that need to be fit.
    def forward(self, dataset, item, col_name):
        ### this takes most of the time
        if not hasattr(self, "transform"):
            raise Exception("Transform not initialized before use.")

        hash_value = self.hash_transform_value(id=dataset.get_sample_id(item), col_name=col_name)

        # start = time.time() 
        # print(item)
        if hash_value in self.cache: 
            transformed_value = self.cache.get(hash_value)
        else: 
            obj = dataset[item][col_name]
            transformed_value = self.transform(obj)
            self.cache[hash_value] = transformed_value
        # # end = time.time()
        # print(f"transforming took {1000 * (end - start)} ms")

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
                # new_column_name_fn = transform.new_column_name_fn
                new_column_name = transform.new_column_name(col_name)
                new_datatype = transform.new_column_datatype
                new_datatypes[new_column_name] = new_datatype

        return new_datatypes

    def __getitem__(self, item):
        start_time = time.time()
        if hasattr(item, "__len__"): 
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(item), shuffle=False)
            new_item = next(iter(loader))
        else: 
            new_item = self.dataset[item]

        end1 = time.time() 
        # print(f"loading data took {(end1 - start_time) * 1000} ms")

        for col_name, transform_list in self.transforms.items():
            for transform in transform_list:
                # new_column_name_fn = transform.new_column_name_fn
                new_column_name = transform.new_column_name(col_name)
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

    def get_matrix(self, column_wise=True, columns=None): 
        if self.transforms == {}:
            return self.dataset.get_matrix(column_wise=column_wise, columns=columns)
        else: 
            if columns is None: 
                columns = [column for column, datatype in self.datatypes().items() if datatype in {DataType.MULTIDIMENSIONAL, DataType.CONTINUOUS, DataType.CATEGORICAL}]

            return DataDetectiveDataset.get_matrix(self, column_wise=column_wise, columns=columns)
