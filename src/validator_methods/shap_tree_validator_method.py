from typing import Dict, Type, Set

import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class ShapTreeValidatorMethod(DataValidatorMethod):
    """
    A validator method for explainable anomaly detection using Shapley values.
    @src: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
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
        Returns the shapley values for the isolation forest that is trained.

        @param data_matrix: an n x d matrix with the datasets needed for the model.
        @param data_schema: a schema mapping from each data column key to its size. needed for aggregation of
        shapley values
        @return: a list of anomaly shapley values.
        """
        iforest = IsolationForest()
        iforest.fit(data_matrix)
        explainer = shap.TreeExplainer(iforest, feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(data_matrix)

        # aggregate shap values by data type
        cumulative_indices = np.cumsum(list(data_schema.values()))
        column_shap_value_dict = {}
        for index, data_key in enumerate(list(data_schema.keys())):
            start_index = 0 if index == 0 else cumulative_indices[index - 1]
            end_index = cumulative_indices[index]
            data_column_shap_values = shap_values[:,start_index:end_index].sum(axis=1)
            column_shap_value_dict[data_key] = data_column_shap_values

        return column_shap_value_dict