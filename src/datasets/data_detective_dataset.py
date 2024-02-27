from abc import abstractmethod
import math
from typing import Any, Dict, List, Optional, Union
import copy
import warnings
import joblib
import pandas as pd
import torch
from torch._utils import _accumulate
from torch import default_generator, randperm

from src.enums.enums import DataType

class DatatypesAndGetItemMeta(type):
    def __new__(cls, name, bases, dct):
        # Check if the class has a 'get_data' method
        if 'datatypes' in dct:
            # Wrap the 'get_data' method with additional behavior
            original_datatypes = dct['datatypes']
            def wrapped_datatypes(self, *args, **kwargs):
                datatypes_dict = original_datatypes(self, *args, **kwargs)
                if self.include_subject_id_in_data: 
                    datatypes_dict |= {"subject_id": DataType.CATEGORICAL} 

                return datatypes_dict
            dct['datatypes'] = wrapped_datatypes

        # Check if the class has a 'get_data' method
        if '__getitem__' in dct:
            # Wrap the 'get_data' method with additional behavior
            original_getitem = dct['__getitem__']
            def wrapped_getitem(self, *args, **kwargs):
                data_dict = original_getitem(self, *args, **kwargs)
                data_idx_or_sample_id = args[0] 
                if self.include_subject_id_in_data: 
                    data_dict["subject_id"] = self.get_subject_id(data_idx_or_sample_id=data_idx_or_sample_id)

                if isinstance(data_idx_or_sample_id, int): 
                    data_idx = data_idx_or_sample_id
                    sample_id = self.get_sample_id(data_idx)
                else:
                    sample_id = data_idx_or_sample_id

                if sample_id not in set(self.index_df['sample_id']):
                    raise Exception(f"Sample ID {sample_id} not found in index dataframe.")

                # lets enforce this to be
                return {
                    "id": self.get_id(sample_id),
                    "data": data_dict
                }

            dct['__getitem__'] = wrapped_getitem
        return super().__new__(cls, name, bases, dct)


class DataDetectiveDataset(torch.utils.data.Dataset, metaclass=DatatypesAndGetItemMeta):
    """
    What should be true of every Data Detective Dataset?
    - it should override __getitem__()
    - it should override __len__()
    - it should override .datatypes()
    - it should satisfy the following identification criteria: 
        - optional map to subject ids  
        - optional map to sample ids
        - if no sample id is provided, one will be provided based on
          the data dictionary

    example: 
    dd_test = dataset(indices=indices) # these indices are then used for the data. 
    """
    def __init__(self, sample_ids: list = None, subject_ids: list = None, show_id: bool = True, include_subject_id_in_data=True):
    # def __init__(self, sample_id_key: str = None, subject_ids: list = None, show_id: bool = True):
        self.show_id = show_id
        self.include_subject_id_in_data = include_subject_id_in_data
        
        if sample_ids and subject_ids: 
            assert(len(sample_ids) == len(subject_ids))

        # We need either sample_ids provided, subject_ids provided, or __len__ provided
        # otherwise we cannot register sample IDs for all of the data... 
        assert(
            subject_ids is not None
            or sample_ids is not None
            or self.__len__() is not None
        )

        index_objects = []
        
        for data_idx in range(self.__len__()):
            index_objects.append({
                "data_idx": data_idx, 
                "sample_id": self.get_sample_id(data_idx, sample_ids=sample_ids),
                "subject_id": self.get_subject_id(data_idx, subject_ids=subject_ids, sample_ids=sample_ids),
            })

        self.index_df = pd.DataFrame(index_objects)

    def should_include_subject_id_in_data(self): 
        return self.include_subject_id_in_data 

    # can be overridden
    # let's try to avoid using idx and focus on data_idx, sample_idx, and subject_idx
    # should this map from internal id or 
    def get_subject_id(self, data_idx_or_sample_id: int, subject_ids: list = None, sample_ids: list = None) -> str: 
        # get the value from the index_df first usually
        if hasattr(self, "index_df"): 
            if isinstance(data_idx_or_sample_id, int): 
                data_idx = data_idx_or_sample_id
                return self.index_df[self.index_df['data_idx'] == data_idx]['subject_id'][0]
            else: 
                sample_id = data_idx_or_sample_id
                return self.index_df[self.index_df['sample_id'] == sample_id]['subject_id'][0]
        
        # only called in __init__
        if isinstance(data_idx_or_sample_id, int): 
            data_idx = data_idx_or_sample_id
            if subject_ids is not None: 
                return subject_ids[data_idx]

            sample_id = self.get_sample_id(data_idx, sample_ids=sample_ids)
        else: 
            # this branch is only called in the pre-index-df regime
            # in other words, you never need to check 
            sample_id = data_idx_or_sample_id

        return sample_id

    # can be overridden
    # maps from data index 
    def get_sample_id(self, data_idx: int, sample_ids: list = None) -> str: 
        if hasattr(self, "index_df"): 
            return self.index_df[self.index_df['data_idx'] == data_idx]['sample_id'][0]

        # only called in __init__
        if sample_ids is not None:
            return sample_ids[data_idx]
        else: 
            return joblib.hash(self.get_data(data_idx))

    def get_id(self, sample_idx) -> Dict[int, Union[str, int]]: 
        id_dict = {
            "subject_id": self.get_subject_id(sample_idx),
            "sample_id": sample_idx,
        }

        return id_dict

    @abstractmethod
    def get_data(self, data_idx_or_sample_id: Union[int, str]) -> Any: 
        pass

    # idx here can be sample idx or normal idx, hard to say. 
    def __getitem__(self, data_idx_or_sample_id) -> Dict[str, Any]: 
        if isinstance(data_idx_or_sample_id, int): 
            data_idx = data_idx_or_sample_id
            sample_id = self.get_sample_id(data_idx)
        else:
            sample_id = data_idx_or_sample_id

        if sample_id not in set(self.index_df['sample_id']):
            raise Exception(f"Sample ID {sample_id} not found in index dataframe.")

        data_dict = self.get_data(data_idx_or_sample_id) 
        if self.include_subject_id_in_data: 
            data_dict["subject_id"] = self.get_subject_id(data_idx_or_sample_id=data_idx_or_sample_id)

        # lets enforce this to be
        return {
            "id": self.get_id(sample_id),
            "data": data_dict
        }

    # not necessary to override
    def __len__(self) -> int: 
        pass

    @abstractmethod
    def datatypes(self) -> Dict[str, DataType]: 
        pass

    def find_all_instances(self, sample_id):
        self.index_df[self.index_df["sample_id"] == sample_id]

    def remove_samples(self, sample_ids: set, in_place=False):
        if not in_place: 
            modified_dataset = copy.deepcopy(self)
        else: 
            modified_dataset = self

        modified_dataset.index_df = modified_dataset.index_df[~modified_dataset.index_df['sample_id'].isin(sample_ids)]

        return modified_dataset

    def remove_indices(self, data_idxs: set, in_place=False):
        if not in_place: 
            modified_dataset = copy.deepcopy(self)
        else: 
            modified_dataset = self

        modified_dataset.index_df = modified_dataset.index_df[~modified_dataset.index_df['data_idx'].isin(data_idxs)]

        return modified_dataset

    def remove_all_indices_except(self, data_idxs: set, in_place=False):
        if not in_place: 
            modified_dataset = copy.deepcopy(self)
        else: 
            modified_dataset = self

        modified_dataset.index_df = modified_dataset.index_df[modified_dataset.index_df['data_idx'].isin(data_idxs)]

        return modified_dataset

def dd_random_split(dataset: DataDetectiveDataset, lengths: [Union[int, float]],
                 generator: Optional[torch.Generator] = torch.default_generator) -> List[torch.utils.data.Subset]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    return [
        dataset.remove_all_indices_except(
            indices[offset - length : offset],
            in_place=False
        ) for offset, length in zip(_accumulate(lengths), lengths)
    ]

