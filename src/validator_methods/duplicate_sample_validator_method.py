from typing import List, Set, Dict, Tuple
import joblib

import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class DuplicateSampleValidatorMethod(DataValidatorMethod):
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return { DataType.MULTIDIMENSIONAL, DataType.IMAGE, DataType.TEXT, DataType.SEQUENTIAL, DataType.CATEGORICAL, DataType.CONTINUOUS }

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {ValidatorMethodParameter.ENTIRE_SET}

    @staticmethod
    def get_method_kwargs(data_object: Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
        """
        Gets the arguments for each run of the validator_method, and what to store the results under. given data_object
        with include_filtering and the validator kwargs, as given precisely in the schema.

        @param data_object: the datasets object after `include` filtering
        @param validator_kwargs:
        @return:
        """
        entire_set = data_object['entire_set']
        kwargs_dict = {
            "entire_set": entire_set
        }

        return kwargs_dict


    @staticmethod
    def validate( entire_set: torch.utils.data.Dataset ) -> List[Tuple[int, int]]:
        hash_indices_dict = {}

        for idx in entire_set.__len__():
            item = entire_set[idx]
            item_hash = joblib.hash(item)

            if item_hash in hash_indices_dict:
                # If the hash is already in the dictionary, add the current index to the set
                hash_indices_dict[item_hash].add(idx)
            else:
                # If the hash is not in the dictionary, create a new set with the current index
                hash_indices_dict[item_hash] = {idx}
                
        duplicate_index_sets = [index_set for index_set in hash_indices_dict.values() if len(index_set) >= 2]
        return duplicate_index_sets
