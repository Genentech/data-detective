import re
import typing
from typing import List

import torch
from torch.utils.data import Dataset

import src.utils
from src.enums.enums import DataType


class ColumnFilteredDataset(Dataset):
    """
    Defines a dataset from another PyTorch dataset that filters on the matching regular expressions.
    """
    def __init__(self, unfiltered_dataset: Dataset = None, matching_regexes: List[str] = None, matching_datatypes: List[DataType] = None):
        self.unfiltered_dataset = unfiltered_dataset
        self.matching_regexes = matching_regexes if matching_regexes else ['.*']
        self.matching_datatypes = matching_datatypes if matching_datatypes else [ e.value for e in DataType ]

    def __getitem__(self, index: int):
        return {
            column_name: entry
            for column_name, entry in self.unfiltered_dataset.__getitem__(index).items()
            if self.include_column(column_name)
        }

    def __len__(self) -> int:
        return self.unfiltered_dataset.__len__()

    @property
    def unfiltered_datatypes(self) -> typing.Dict[str, DataType]:
        # fixes a bug where there are multiple dataset splits
        ptr = self.unfiltered_dataset
        while isinstance(ptr, torch.utils.data.Subset):
            ptr = ptr.dataset
        return ptr.datatypes()
        # fixes a bug where sometimes the self.unfiltered dataset is a subset

    def include_column(self, column_name):
        has_matching_datatype = self.unfiltered_datatypes[column_name].value in self.matching_datatypes
        has_matching_name = any([re.compile(reg_exp).match(column_name) for reg_exp in self.matching_regexes])

        return has_matching_name and has_matching_datatype

    def datatypes(self) -> typing.Dict[str, DataType]:
        datatypes = {
            column_name: datatype
            for column_name, datatype in self.unfiltered_datatypes.items()
            if self.include_column(column_name)
        }

        return datatypes

    def to_matrix(self):
        return self.unfiltered_dataset.to_matrix()

    def id(self):
        original_dataset = self.unfiltered_dataset

        while isinstance(original_dataset, ColumnFilteredDataset):
            original_dataset = src.utils.unfilter_dataset(original_dataset)

        return id(original_dataset)
