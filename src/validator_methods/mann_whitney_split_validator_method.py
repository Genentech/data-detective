import itertools
from typing import List, Set, Dict, Type, Union, Any

import numpy as np
import pandas as pd
import scipy
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset
from src.validator_methods.data_validator_method import DataValidatorMethod

from scipy import stats

class MannWhitneySplitValidatorMethod(DataValidatorMethod):
    """
    A nonparametric method for determining whether the train/test split has any distribution shift.

    The Mann-Whitney U-test is a nonparametric test of the null hypothesis that, for randomly selected values X and Y
    from two populations, the probability of X being greater than Y is equal to the probability of Y being greater than
    X.
    """
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return {
            DataType.CONTINUOUS
        }

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {
            ValidatorMethodParameter.TRAINING_SET,
            ValidatorMethodParameter.VALIDATION_SET,
            ValidatorMethodParameter.TEST_SET,
        }

    @staticmethod
    def get_method_kwargs(data_object: Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
        """
        Gets the arguments for each run of the validator_method, and what to store the results under.

        @param data_object:
        @param validator_kwargs:
        @return: a dict mapping from the key the result will be stored under to the
        arguments to call the .validate method from. For example:
        """
        kwargs_dict: dict[str, dict[str, Union[ndarray, Any]]] = {}

        # test_dataset = data_object["test_set"]
        # training_dataset = data_object["training_set"]
        # validation_dataset = data_object["validation_set"]
        dataset_keys = sorted([param_key.value for param_key in MannWhitneySplitValidatorMethod.param_keys()])
        # train, val, test order
        dataset_keys[0], dataset_keys[1] = dataset_keys[1], dataset_keys[0]

        # columns = list(test_dataset.datatypes().keys())

        def get_series(column_key, dataset):
            matrix_dict = {
                column: [] for column in dataset.datatypes().keys()
            }

            for idx in range(dataset.__len__()):
                sample = dataset[idx]
                for column, column_data in sample.items():
                    matrix_dict[column].append(column_data)

            for column in dataset.datatypes().keys():
                matrix_dict[column] = np.vstack(matrix_dict[column])

            return matrix_dict[column_key].flatten()

        for dataset_0_key, dataset_1_key in itertools.combinations(dataset_keys, 2):
            dataset_0 = data_object[dataset_0_key]
            dataset_1 = data_object[dataset_1_key]

            columns_0 = sorted(list(dataset_0.datatypes().keys()))
            columns_1 = sorted(list(dataset_1.datatypes().keys()))
            if columns_0 != columns_1:
                raise Exception("Columns in datasets splits are not the same")
            else:
                columns = columns_0

            for column_name in columns:
                series_0 = get_series(column_name, dataset_0)
                series_1 = get_series(column_name, dataset_1)

                # series_0 = np.array(list(dataset_0[:][column_name].values()))
                # series_1 = np.array(list(dataset_1[:][column_name].values()))
                kwargs_dict[f"{dataset_0_key}/{dataset_1_key}/{column_name}"] = {
                    "series_0" : series_0,
                    "series_1" : series_1,
                }

        return kwargs_dict


    @staticmethod
    def validate(series_0: Type[np.array], series_1: Type[np.array]) -> object:
        """
        Runs a kolmogorov-smirnov test against N(0, 1)
        Input dict:
        {
            "": {
                0: 1.1412321,
                ...
                9999: -0.4123643
            }
        }

        @return: the stats object that it needs when it gets back.
        """
        return scipy.stats.mannwhitneyu(series_0, series_1)

