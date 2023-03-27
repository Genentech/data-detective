from typing import Dict, Type, Set

import numpy as np
from DIFFI.utils import local_diffi_batch
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class DiffiAnomalyExplanationValidatorMethod(DataValidatorMethod):
    """
    A validator method for explainable anomaly detection using the DIFFI feature importance method.
    @src: https://diffi-lrjball.readthedocs.io/en/latest/generated/diffi.TreeExplainer.html
    """
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        Returns the datatype the validators method operates on
        @return: the datatype the validators method operates on
        """
        return {DataType.MULTIDIMENSIONAL}

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters in the datasets object that the validators operates on.
        @return: a list of parameters for the .validate() method.
        """
        return {ValidatorMethodParameter.ENTIRE_SET}


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
                "data_matrix": matrix,
                "data_schema": {data_column_name: data_value.flatten().shape[0] for data_column_name, data_value in entire_dataset[0].items()}
            }
        }

        return kwargs_dict

    @staticmethod
    def validate(
            data_matrix: Type[np.array] = None,  # n x d
            data_schema: Dict[str, int] = None,
    ) -> object:
        """
        Returns the diffi values for the isolation forest that is trained.

        @param data_matrix: an n x d matrix with the datasets needed for the model.
        @param data_schema: a schema mapping from each data column key to its size. needed for aggregation of
        diffi values
        @return: a list of anomaly diffi values.
        """
        iforest = IsolationForest(max_samples=64)
        iforest.fit(data_matrix)
        diffi_values, ord_idx_diffi_te, _ = local_diffi_batch(iforest, data_matrix)

        return diffi_values