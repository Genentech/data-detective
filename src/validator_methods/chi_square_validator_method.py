from typing import Set, Dict

import numpy as np
import pandas as pd
import scipy.stats
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class ChiSquareValidatorMethod(DataValidatorMethod):
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return { DataType.CATEGORICAL }

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
        entire_dataset = data_object['entire_set']
        kwargs_dict = {}

        ci_relations = validator_kwargs['ci_relations']
        for index, ci_relation in enumerate(ci_relations):
            x, y = ci_relation['x'], ci_relation['y']
            kwargs_dict[f"relation between {x} and {y}"] = {
                'x': entire_dataset[:][x],
                'y': entire_dataset[:][y],
            }

        return kwargs_dict


    @staticmethod
    def validate(x: np.array, y: np.array) -> object:
        min_length = min(len(x), len(y))
        df = pd.DataFrame({'x': x[:min_length], 'y': y[:min_length]})
        contingency_table = pd.crosstab(index=df['x'], columns=df['y']).to_numpy()
        return scipy.stats.chi2_contingency(contingency_table)
