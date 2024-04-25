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
    EPS = 1e-15

    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return {DataType.MULTIDIMENSIONAL}

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {ValidatorMethodParameter.ENTIRE_SET}

    @staticmethod
    def get_method_kwargs(
        data_object: Dict[str, Dataset], validator_kwargs: Dict = None
    ) -> Dict:
        """
        Gets the arguments for each run of the validator_method, and what to store the results under. given data_object
        with include_filtering and the validator kwargs, as given precisely in the schema.

        @param data_object: the datasets object after `include` filtering
        @param validator_kwargs:
        @return:
        """
        entire_set = data_object["entire_set"]
        kwargs_dict = {}
        matrix_dict = entire_set.get_matrix(column_wise=True)

        for column_name in entire_set.datatypes().keys():
            kwargs_dict[column_name] = {
                "data": matrix_dict[column_name],
                "angle_threshold": validator_kwargs.get("angle_threshold", 15),
            }

        return kwargs_dict

    @staticmethod
    def validate(data: np.array, angle_threshold: float) -> List[Tuple[int, int]]:
        angle_threshold = np.radians(angle_threshold)
        cos_sim_threshold = (
            np.cos(angle_threshold) - NearDuplicateMultidimensionalValidatorMethod.EPS
        )

        # Normalize vectors
        norm_data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

        # Calculate cosine similarities using a single matrix multiplication
        cos_sims = np.dot(norm_data, norm_data.T)

        # Set lower triangular part to zero to avoid self-comparisons and duplicates
        np.fill_diagonal(cos_sims, 0)

        # Find indices where cosine similarity exceeds the threshold
        rows, cols = np.where(cos_sims >= cos_sim_threshold)

        # Create a list of pairs
        # todo: add scores in too.
        all_pairs = np.column_stack((rows, cols))

        # Filter pairs where the row index is less than the column index
        pairs = all_pairs[all_pairs[:, 0] < all_pairs[:, 1]]

        return pairs
