import itertools
from typing import Set, Dict

import numpy as np
import pandas as pd
import scipy.stats
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod
from src.validator_methods.chi_square_validator_method import ChiSquareValidatorMethod
from src.utils import get_split_group_keys


class ChiSquareSplitValidatorMethod(ChiSquareValidatorMethod):
    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {ValidatorMethodParameter.SPLIT_GROUP_SET}

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

            # needs to be label encoded or it breaks independence testing
            return np.argmax(matrix_dict[column_key], axis=1)


        only_split_groups_data_object = {split_group_name: split_group_data_object 
                            for split_group_name, split_group_data_object in data_object.items()
                            if split_group_name in get_split_group_keys(data_object)}

        for split_group_name, split_group_data_object in only_split_groups_data_object.items():
            dataset_keys = list(split_group_data_object.keys())
            for dataset_0_key, dataset_1_key in itertools.combinations(dataset_keys, 2):
                dataset_0 = split_group_data_object[dataset_0_key]
                dataset_1 = split_group_data_object[dataset_1_key]

                columns_0 = sorted(list(dataset_0.datatypes().keys()))
                columns_1 = sorted(list(dataset_1.datatypes().keys()))
                if columns_0 != columns_1:
                    raise Exception("Columns in datasets splits are not the same")
                else:
                    columns = columns_0

                for column_name in columns: 
                    assert(dataset_0.datatypes()[column_name].value == DataType.CATEGORICAL.value)
                    kwargs_dict[f"{split_group_name}/{dataset_0_key}_vs_{dataset_1_key}/{column_name}"] = {
                        'x': get_series(column_name, dataset_0),
                        'y': get_series(column_name, dataset_1),
                    }

        return kwargs_dict

    @staticmethod
    def validate(x: np.array, y: np.array) -> object:
        def create_split_array(original_data1, original_data2):
            split_labels = np.array(['Split1', 'Split2'])  # You can customize these labels based on your needs
            split_array1 = np.full(len(original_data1), split_labels[0])
            split_array2 = np.full(len(original_data2), split_labels[1])
            return np.concatenate([original_data1, original_data2]), np.concatenate([split_array1, split_array2])

        data, splits = create_split_array(x, y)

        df = pd.DataFrame({'data': data, 'splits': splits})
        contingency_table = pd.crosstab(index=df['data'], columns=df['splits']).to_numpy()
        return scipy.stats.chi2_contingency(contingency_table)