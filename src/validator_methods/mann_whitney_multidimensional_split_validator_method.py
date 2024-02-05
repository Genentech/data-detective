import itertools
from typing import List, Set, Dict, Type, Union, Any

import numpy as np
import pandas as pd
import scipy.stats
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset
from src.validator_methods.data_validator_method import DataValidatorMethod

from scipy import stats

class MannWhitneyMultidimensionalSplitValidatorMethod(DataValidatorMethod):
    """
    A method for determining whether the train/test split has any distribution shift. The Kolmogorov-Smirnov Test
    measures the greatest distance in CDF over a continuous variable.

    https://stats.stackexchange.com/questions/57885/how-to-interpret-p-value-of-kolmogorov-smirnov-test-python
    """
    DEFAULT_ALPHA = 0.05

    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return {
            DataType.MULTIDIMENSIONAL
        }

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {
            ValidatorMethodParameter.SPLIT_GROUP_SET
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

        
        for split_group_name, split_group_data_object in data_object["split_group_set"].items():
            dataset_keys = list(split_group_data_object.keys())
            alpha = validator_kwargs.get("alpha", MannWhitneyMultidimensionalSplitValidatorMethod.DEFAULT_ALPHA)

            # the number of combinations over the splits (used for bonferroni correction)
            num_combinations = len(list(itertools.combinations(dataset_keys, 2)))
            for dataset_0_key, dataset_1_key in itertools.combinations(dataset_keys, 2):
                dataset_0 = split_group_data_object[dataset_0_key]
                dataset_1 = split_group_data_object[dataset_1_key]

                dataset_combination_str = f"{split_group_name}/{dataset_0_key}/{dataset_1_key}"

                columns_0 = sorted(list(dataset_0.datatypes().keys()))
                columns_1 = sorted(list(dataset_1.datatypes().keys()))
                if columns_0 != columns_1:
                    raise Exception("Columns in datasets splits are not the same")
                else:
                    columns = columns_0

                for column_name in columns:
                    def get_matrix(column_key, dataset):
                        matrix_lst = []

                        for idx in range(dataset.__len__()):
                            sample = dataset[idx]
                            column_data = sample[column_key]
                            matrix_lst.append(column_data)

                        matrix_lst = np.vstack(matrix_lst)

                        return matrix_lst

                    matrix_0 = get_matrix(column_name, dataset_0)
                    matrix_1 = get_matrix(column_name, dataset_1)

                    kwargs_dict[f"{dataset_0_key}/{dataset_1_key}/{column_name}"] = {
                        "matrix_0" : matrix_0,
                        "matrix_1" : matrix_1,
                        "alpha": alpha,
                        "num_combinations": num_combinations,
                    }

        return kwargs_dict


    @staticmethod
    def validate(matrix_0: Type[np.array], matrix_1: Type[np.array], alpha: float, num_combinations: int) -> object:
        """
        @param matrix_0: an n x d matrix of features, where n is the number of entries and d is the dimension.
        @param matrix_1: an n x d matrix of features, where n is the number of entries and d is the dimension.
        @param alpha: the significance of the test
        @param num_combinations: the number of combinations over dataset splits (used for bonferroni correction)
        @return: the stats object that it needs when it gets back and a test results (pass/fail)
        """
        assert(len(matrix_0.shape) == 2)
        assert(len(matrix_1.shape) == 2)
        assert(matrix_0.shape[1] == matrix_1.shape[1])

        num_columns = matrix_0.shape[1]
        statistics = []

        for i in range(num_columns):
            series_0 = matrix_0[:,i]
            series_1 = matrix_1[:,i]
            statistics.append(scipy.stats.mannwhitneyu(series_0, series_1))

        # bonferroni correction (3 for number of combinations, 3 choose 2 for val/test train/val train/test)
        alpha /= num_columns * num_combinations
        has_significant_result = any([statistic.pvalue < alpha for statistic in statistics])

        return statistics, {"has_significant_result": has_significant_result}

