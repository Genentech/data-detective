from typing import List, Set, Dict, Tuple
import joblib

import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class NearDuplicateMultidimensionalValidatorMethod(DataValidatorMethod):
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return { DataType.MULTIDIMENSIONAL }

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return { ValidatorMethodParameter.ENTIRE_SET }

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
        kwargs_dict = {}
        def get_matrix(column_key, dataset):
            matrix_lst = []

            for idx in range(dataset.__len__()):
                sample = dataset[idx]
                column_data = sample[column_key]
                matrix_lst.append(column_data)

            matrix_lst = np.vstack(matrix_lst)

            return matrix_lst

        for column_name in entire_set.datatypes().keys():
            kwargs_dict[column_name] = {
                "data": get_matrix(column_name, entire_set),
                "angle_threshold": validator_kwargs.get("angle_threshold", 15)
            }

        return kwargs_dict

    @staticmethod
    def validate( data: np.array, angle_threshold: float) -> List[Tuple[int, int]]:
        angle_threshold = np.radians(angle_threshold)
        cos_sim_threshold = np.cos(angle_threshold)

        # Normalize vectors
        norm_data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

        # Calculate cosine similarities using a single matrix multiplication
        cos_sims = norm_data @ norm_data.T

        # Set lower triangular part to zero to avoid self-comparisons and duplicates
        np.fill_diagonal(cos_sims, 0)
        cos_sims = np.triu(cos_sims)

        # Find indices where cosine similarity exceeds the threshold
        rows, cols = np.where(cos_sims >= cos_sim_threshold)

        # Create a list of pairs
        pairs = list(zip(rows, cols))

        return pairs
