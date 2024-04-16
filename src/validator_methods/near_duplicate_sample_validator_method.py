from typing import List, Set, Dict, Tuple
import joblib

import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class NearDuplicateSampleValidatorMethod(DataValidatorMethod):
    EPS = 1e-15
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return { DataType.CATEGORICAL, DataType.CONTINUOUS, DataType.MULTIDIMENSIONAL }

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
        Gets the arguments for each run of the validator_method, and what to store the results under.

        @param data_object: the datasets object containing the datasets (train, test, entire, etc.)
        @param validator_kwargs: the kwargs from the validation schema.
        @return: a dict mapping from the key the result from calling .validate() on the kwargs values.
        """
        entire_dataset: Dataset = data_object["entire_set"]

        matrix = []

        for idx in range(entire_dataset.__len__()):
            sample = entire_dataset[idx]
            matrix.append(
                np.concatenate([k.flatten() for k in sample.values()])
            )

        matrix = np.array(matrix)

        kwargs_dict = {
            f"results": {
                "data": matrix,
                "angle_threshold": validator_kwargs.get("angle_threshold", 15)
            }
        }

        return kwargs_dict

    @staticmethod
    def validate( 
        data: np.array, 
        angle_threshold: float
    ) -> List[Tuple[int, int]]:
        angle_threshold = np.radians(angle_threshold)
        cos_sim_threshold = np.cos(angle_threshold) - NearDuplicateSampleValidatorMethod.EPS

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
        # todo: add scores in too. 
        pairs = list(zip(rows, cols))

        return pairs
