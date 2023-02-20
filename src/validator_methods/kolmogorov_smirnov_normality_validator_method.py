from typing import List, Set, Dict, Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset
from src.validator_methods.data_validator_method import DataValidatorMethod

from scipy import stats

class KolmogorovSmirnovNormalityValidatorMethod(DataValidatorMethod):
    """
    A method for determining normality of a continuous feature. The Kolmogorov-Smirnov Test measures the greatest
    distance in CDF
    """
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return {DataType.CONTINUOUS}

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
        Gets the arguments for each run of the validator_method, and what to store the results under.

        @param data_object:
        @param validator_kwargs:
        @return: a dict mapping from the key the result will be stored under to the
        arguments to call the .validate method from. For example:
        """
        kwargs_dict = {}

        entire_dataset = data_object["entire_set"]
        columns = list(entire_dataset.datatypes().keys())

        for column_name in columns:
            column = np.array(list(entire_dataset[:][column_name].values()))
            kwargs_dict[column_name] = {'series': column}

        return kwargs_dict

    @staticmethod
    def validate(series: Type[np.array]) -> object:
        """
        Runs a kolmogorov-smirnov test against N(0, 1)

        @return: the stats object that it needs when it gets back.
        """
        return stats.kstest(series, 'norm')
