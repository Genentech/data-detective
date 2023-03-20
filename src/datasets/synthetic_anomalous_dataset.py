import typing
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticAnomalousDataset(Dataset):
    """
    Constructs a dataset from an anomalous dataset that works well for the test bench
    for situations where oversampling is needed to get some nice anomaly classes.

    @precondition: normal class proportion is higher than current normal class proportion.
    """
    def __init__(self,
                 original_dataset: Dataset,
                 normal_class: int = 0,
                 include_all: bool = True,
                 # new_length: Optional[int] = None,
                 desired_normal_class_proportion = 0.5,
                 ) -> None:

        #todo: support case where include_all isn't true

        self.original_dataset = original_dataset
        self.normal_class = normal_class
        # this is the structure we are using to map from the
        self.class_indices = self._get_class_indices(original_dataset)
        self.added_normal_elements = self.compute_added_normal_elements(include_all, desired_normal_class_proportion)

    def compute_added_normal_elements(self, include_all: bool, desired_normal_class_proportion: float):
        """
        Computes how many added normal elements we need to get to our desired normal class proporotion.

        @param include_all: whether to include all of the existing elements as well
        @return: a number of normal elements that need to be added in order
        """
        # haven't supported partial inclusion yet
        assert include_all
        class_counts = self._get_class_counts(self.original_dataset)

        original_normal_class_proportion = class_counts[self.normal_class] / self.original_dataset.__len__()
        assert original_normal_class_proportion < desired_normal_class_proportion, "desired normal class proportion is smaller than original normal class proportion"

        # compute minimum required length for include_all
        if include_all:
            original_normal_class_count = class_counts[self.normal_class]
            total_count = self.original_dataset.__len__()
            # eqn for oversampling:
                # n + k / (N + k) = p => k = ( N*p - n ) / (1 - p)
            added_normal_elements = int(np.ceil(
                (total_count * desired_normal_class_proportion - original_normal_class_count) / (
                            1 - desired_normal_class_proportion)))

            return added_normal_elements

    @staticmethod
    def _get_class_counts(original_dataset: Dataset) -> typing.Dict[int, int]:
        """
        Gets a class count for an original dataset by using the get_class_indices method.

        @param original_dataset: the dataset to get the class count from
        @return: the class counts for the dataset.
        """
        return {
            cls: len(index_list) \
                for cls, index_list in SyntheticAnomalousDataset._get_class_indices(original_dataset).items()
        }

    @staticmethod
    def _get_class_indices(original_dataset: Dataset) -> typing.Dict[int, List[int]]:
        """
        Gets the class counts for the anomalous dataset.

        @param original_dataset: the original dataset that is getting oversampled
        @return: the a dictionary mapping from each class to a list of the indices in that class
        """
        class_indices: defaultdict[int, List[int]] = defaultdict(lambda: [])
        dataset_length: int = original_dataset.__len__()

        for idx in range(dataset_length):
            sample: typing.Dict[str, Union[int, torch.FloatTensor]] = original_dataset[idx]
            sample_class: int = sample['label']
            class_indices[sample_class].append(idx)

        return class_indices

    def datatypes(self):
        return self.original_dataset.datatypes()

    def __len__(self):
        return self.original_dataset.__len__() + self.added_normal_elements

    def __getitem__(self, index: int):
        """
        Gets an item from the synthetic anomalous dataset. If the index is less than the length of the original dataset,
        we will just use getitem from the original dataset. If the index is larger, we will allocate this index
        to the normal class, and we will pick out an object from the normal_class_index_list to then access the original
        dataset from.

        @param index: the index to get the item from
        @return: the index of the item in the given dataset.`
        """
        if index >= self.original_dataset.__len__():
            normal_class_index_list = self.class_indices[self.normal_class]
            index = normal_class_index_list[index % len(normal_class_index_list)]

        return self.original_dataset[index]
