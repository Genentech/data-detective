import re
import typing
from typing import List

import torch
from torch.utils.data import Dataset

from src.datasets.data_detective_dataset import DataDetectiveDataset, LambdaDictWrapper
from src.enums.enums import DataType


class ColumnFilteredDataset(DataDetectiveDataset):
    """
    Defines a dataset from another PyTorch dataset that filters on the matching regular expressions.
    """
    def __init__(self, unfiltered_dataset: Dataset = None, matching_regexes: List[str] = None, matching_datatypes: List[DataType] = None):
        self.matching_regexes = matching_regexes if matching_regexes else ['.*']
        self.matching_datatypes = matching_datatypes if matching_datatypes else [ e.value for e in DataType ]

        self.unfiltered_dataset = unfiltered_dataset
        self.include_subject_id_in_data = unfiltered_dataset.include_subject_id_in_data
        self.show_id = unfiltered_dataset.show_id
        self.index_df = unfiltered_dataset.index_df

    def __getitem__(self, index: int) -> typing.Dict:
        """
        Returns an item from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the data and the label.
        """
        original_item = self.unfiltered_dataset.__getitem__(index)
        if isinstance(original_item, LambdaDictWrapper):
            original_item.unwrap = False
            new_item = LambdaDictWrapper({
                column_name: entry
                for column_name, entry in original_item.items()
                if self.include_column(column_name)
            })
        else: 
            new_item = {
                column_name: entry
                for column_name, entry in original_item.items()
                if self.include_column(column_name)
            }

        return new_item

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        @return: the length of the dataset
        """
        return self.unfiltered_dataset.__len__()

    @property
    def unfiltered_datatypes(self) -> typing.Dict[str, DataType]:
        """
        Gets the datatypes of the unfiltered dataset
        @return: datatypes of the unfiltered dataset
        """
        # fixes a bug where there are multiple dataset splits
        ptr = self.unfiltered_dataset
        while isinstance(ptr, torch.utils.data.Subset):
            ptr = ptr.dataset
        return ptr.datatypes()
        # fixes a bug where sometimes the self.unfiltered dataset is a subset

    def include_column(self, column_name: str) -> bool:
        """
        Returns whether or not the column should be included in the filtered dataset.

        @param column_name: the name of the column to check
        @return: a boolean indicating whether or not the column should be included.
        """
        has_matching_datatype = self.unfiltered_datatypes[column_name].value in self.matching_datatypes
        has_matching_name = any([re.compile(reg_exp).match(column_name) for reg_exp in self.matching_regexes])

        return has_matching_name and has_matching_datatype

    def datatypes(self) -> typing.Dict[str, DataType]:
        """
        Gives the datatypes of a dataset sample for the column filtered dataset.
        @return: the datatypes of a dataset sample.
        """
        datatypes = {
            column_name: datatype
            for column_name, datatype in self.unfiltered_datatypes.items()
            if self.include_column(column_name)
        }

        return datatypes