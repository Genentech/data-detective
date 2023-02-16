from typing import List, Set, Dict, Type

import numpy as np
import pandas as pd
import torch
from fcit import fcit
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset
from src.validator_methods.data_validator_method import DataValidatorMethod


class FCITValidatorMethod(DataValidatorMethod):
    """
    A method for determining conditionanl independence of two multidimensional vectors given a third.
    """
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return {DataType.CONTINUOUS, DataType.CATEGORICAL, DataType.MULTIDIMENSIONAL}

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
        kwargs_dict = {}

        entire_dataset = data_object["entire_set"]
        ci_relations = validator_kwargs["ci_relations"]
        """
                            "validator_kwargs": {
                        "ci_relations" : [{
                            "independent" : "True",
                            "x": "x",
                            "y": "y",
                            "given": [
                                "z"
                            ]
                        }]
                    }
        """


        for index, ci_relation in enumerate(ci_relations):
            entire_dataset_givens = {key: entire_dataset[:][key] for key in ci_relation['given']}
            givens = np.concatenate(np.array(list(entire_dataset_givens.values())), axis=1)

            kwargs_dict[f"{index}"] = {
                'x': entire_dataset[:][ci_relation['x']],
                'y': entire_dataset[:][ci_relation['y']],
                'z': givens,
            }

        return kwargs_dict

    @staticmethod
    def validate(x: np.array, y: np.array, z: np.array) -> object:
        """
        Runs a Fast Conditional Independence Test to determine if two variables are conditionally independent, given
        a set of other variables.

        Concatenates all the other variables into Z, since X ind Y | W,V is the same when z = cat(w, v)

        {
            "feature_name": {
                0: 1.1412321,
                ...
                9999: -0.4123643
            }
        }

        @return: the stats object that it needs when it gets back.
        """
        result = fcit.test(x, y, z)
        return result
