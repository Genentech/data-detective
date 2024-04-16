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

from collections.abc import MutableMapping


class LambdaDictWrapper(MutableMapping):
    def __init__(self, dictionary):
        self._data = dictionary
        self.unwrap = True

    def __getitem__(self, key):
        value = self._data[key]
        if callable(value) and self.unwrap:
            return value()
        return value

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]

class DatatypesAndGetItemMeta(type):


    def __new__(cls, name, bases, dct):
        # Check if the class has a 'get_data' method
        if 'datatypes' in dct:
            # Wrap the 'get_data' method with additional behavior
            original_datatypes = dct['datatypes']
            def wrapped_datatypes(self, *args, **kwargs):
                datatypes_dict = original_datatypes(self, *args, **kwargs)
                if hasattr(self, "include_subject_id_in_data") and self.include_subject_id_in_data: 
                    datatypes_dict |= {"subject_id": DataType.CATEGORICAL} 

                return datatypes_dict
            dct['datatypes'] = wrapped_datatypes

        # Check if the class has a 'get_data' method
        if '__getitem__' in dct:
            # Wrap the 'get_data' method with additional behavior
            original_getitem = dct['__getitem__']
            def wrapped_getitem(self, *args, **kwargs):
                from src.datasets.column_filtered_dataset import ColumnFilteredDataset
                from src.datasets.one_hot_encoded_dataset import OneHotEncodedDataset
                from src.transforms.embedding_transformer import TransformedDataset

                assert len(args) == 1, f"You must provide exactly one argument to __getitem__, you passed {args}."
                try: 
                    # handles the case of subsetting
                    if hasattr(self, "index_df") and isinstance(args[0], int) and not any([isinstance(self, klass) for klass in [ColumnFilteredDataset, OneHotEncodedDataset, TransformedDataset]]): 
                        new_args = (self.index_df.iloc[args[0]]['data_idx'], )
                    else: 
                        new_args = args
                    data_dict = original_getitem(self, *new_args, **kwargs)
                except (KeyError, IndexError, TypeError):
                    if isinstance(args[0], int): 
                        idx = list(self.index_df[self.index_df.index == args[0]]['sample_id'])[0]
                    else: 
                        idx = list(self.index_df[self.index_df["sample_id"] == args[0]].index)[0]

                    data_dict = original_getitem(self, *(idx,), **kwargs)

                if not self.show_id and not self.include_subject_id_in_data: 
                    return data_dict 

                data_idx_or_sample_id = args[0] 
                if self.include_subject_id_in_data: 
                    data_dict["subject_id"] = self.get_subject_id(data_idx_or_sample_id=data_idx_or_sample_id)
                if not self.show_id: 
                    return data_dict

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
        if '__len__' in dct:
            # length call works as follows: 
                # if self.index_df works, use that
            # otherwise: 
                # try to call 

            original_len = dct['__len__']
            def wrapped_len(self, *args, **kwargs):
                if hasattr(self, "index_df"):
                    return len(self.index_df)
                
                orig_len = original_len(self, *args, **kwargs)
                if orig_len is None: 
                    raise Exception("Must either provide sample ids, subject ids, or a valid initial length.")
                return orig_len

            dct['__len__'] = wrapped_len
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
    def __init__(self, dataset = None, sample_ids: list = None, subject_ids: list = None, show_id: bool = False, include_subject_id_in_data=False ):
        self.include_subject_id_in_data = include_subject_id_in_data
        self.show_id = False # only for initialization
        
        if sample_ids and subject_ids: 
            assert(len(sample_ids) == len(subject_ids))

        #todo: initial_length => __len__(): 
        if sample_ids is not None: 
            initial_length = len(sample_ids)
        elif subject_ids is not None: 
            initial_length = len(subject_ids)
        else: 
            initial_length = self.__len__()
    
        index_objects = []
        for data_idx in range(initial_length):
            """
            Here is the general idea / design considerations behind this section. 
            - we have accepted that multiple objects can have the same sample IDs, 
            given that they have the same data AND subject ids. so the sample id is 
            contingent on the subject id. 
            - however, we also know that in the case that there is no subject id, the
            subject id is dependent on the sample id (because they should be the same)
            and the sample id is no longer dependent on the subject id
            - SO. we try to fill in the subject id first based on the given list. if we 
            are able to, then we can fill in the sample_id first by either given list or 
            by hashing the data object, turning off any ID 
            """
            optional_sample_id = sample_ids[data_idx] if sample_ids is not None else None
            optional_subject_id = subject_ids[data_idx] if subject_ids is not None else None

            if optional_sample_id is not None: 
                sample_id = optional_sample_id
            else: 
                sample_id = self.get_default_sample_id(data_idx, subject_id = optional_subject_id)

            if optional_subject_id is not None: 
                subject_id = optional_subject_id
            else: 
                subject_id = self.get_default_subject_id(data_idx, sample_id = sample_id)

            index_object = {
                "data_idx": data_idx,
                "sample_id": sample_id,
                "subject_id": subject_id,
            }

            index_objects.append(index_object)

        self.index_df = pd.DataFrame(index_objects)
        self.show_id = show_id
    
    def get_default_sample_id(self, data_idx, subject_id=None):
        saved_show_id = self.show_id
        saved_include_subject_id_in_data = self.include_subject_id_in_data

        self.show_id = False
        self.include_subject_id_in_data = False
        
        data_dict = joblib.hash(self.__getitem__(data_idx))
        if subject_id is not None and saved_include_subject_id_in_data: 
            data_dict['subject_id'] = subject_id

        self.show_id = saved_show_id
        self.include_subject_id_in_data = saved_include_subject_id_in_data
        return joblib.hash(data_dict)
                
    def get_default_subject_id(self, data_idx, sample_id=None):
        if sample_id is None: 
            sample_id = self.default_sample_id(data_idx, subject_id=None)
        
        return sample_id

    # can be overridden
    # let's try to avoid using idx and focus on data_idx, sample_idx, and subject_idx
    # should this map from internal id or 
    # note: all ids must not be ints, to avoid conflict with data indexing.
    def get_subject_id(self, data_idx_or_sample_id: int, subject_ids: list = None, sample_ids: list = None) -> str: 
        if isinstance(data_idx_or_sample_id, int): 
            data_idx = data_idx_or_sample_id
            return self.index_df[self.index_df.index == data_idx]['subject_id'].item()
        else: 
            sample_id = data_idx_or_sample_id
            return self.index_df[self.index_df['sample_id'] == sample_id]['subject_id'][0]

    # can be overridden
    # maps from data index 
    def get_sample_id(self, data_idx: int, sample_ids: list = None) -> str: 
        return self.index_df[self.index_df.index == data_idx]['sample_id'].item()
        # try: 
        #     return self.index_df[self.index_df.index == data_idx]['sample_id'].item()
        # except Exception: 
        #     c=3

    def get_id(self, sample_idx) -> Dict[int, Union[str, int]]: 
        id_dict = {
            "subject_id": self.get_subject_id(sample_idx),
            "sample_id": sample_idx,
        }

        return id_dict

    # idx here can be sample idx or normal idx, hard to say. 
    def __getitem__(self, data_idx_or_sample_id) -> Dict[str, Any]: 
        pass

    # not necessary to override
    def __len__(self) -> int: 
        return len(self.index_df)

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
        modified_dataset.index_df.reset_index(inplace=True, drop=True)

        return modified_dataset

    def remove_indices(self, data_idxs: set, in_place=False):
        if not in_place: 
            modified_dataset = copy.deepcopy(self)
        else: 
            modified_dataset = self

        modified_dataset.index_df = modified_dataset.index_df[~modified_dataset.index_df['data_idx'].index.isin(data_idxs)]
        modified_dataset.index_df.reset_index(inplace=True, drop=True)

        return modified_dataset

    def remove_all_indices_except(self, data_idxs: set, in_place=False):
        if not in_place: 
            modified_dataset = copy.deepcopy(self)
        else: 
            modified_dataset = self

        modified_dataset.index_df = modified_dataset.index_df[modified_dataset.index_df['data_idx'].index.isin(data_idxs)]
        modified_dataset.index_df.reset_index(inplace=True, drop=True)

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

    split_datasets = [
        dataset.remove_all_indices_except(
            indices[offset - length : offset],
            in_place=False
        ) for offset, length in zip(_accumulate(lengths), lengths)
    ]

    assert sum([dataset.__len__() for dataset in split_datasets]) == dataset.__len__()

    return split_datasets


