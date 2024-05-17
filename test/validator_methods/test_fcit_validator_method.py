from typing import Dict

import numpy as np
import torch

from src.datasets.data_detective_dataset import dd_random_split
from src.data_detective_engine import DataDetectiveEngine
from src.datasets.synthetic_ci_dataset import SyntheticCIDataset


class TestFCITValidatorMethod:
    def test_positive_example(self):
        """
        Example of CI results on a set with a true conditional independence. (CI is null hypothesis.)
        """
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "conditional_independence_data_validator": {
                    "include": [
                        "^[xyz]$"
                    ],
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
                }
            }
        }

        ci_dataset: SyntheticCIDataset = SyntheticCIDataset(dataset_type='CI')
        train_size: int = int(0.6 * len(ci_dataset))
        val_size: int = int(0.2 * len(ci_dataset))
        test_size: int = len(ci_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset=dd_random_split(ci_dataset, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": ci_dataset
        }

        # results == {'conditional_independence_data_validator': {'fcit_validator_method': {'0': 0.5}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        p_value = results['conditional_independence_data_validator']['fcit_validator_method']['0']

        assert(p_value > 0.05)

    def test_negative_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "conditional_independence_data_validator": {
                    "include": [
                        "^[xyz]$"
                    ],
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
                }
            }
        }

        ci_dataset: SyntheticCIDataset = SyntheticCIDataset(dataset_type='NI')
        train_size: int = int(0.6 * len(ci_dataset))
        val_size: int = int(0.2 * len(ci_dataset))
        test_size: int = len(ci_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset=dd_random_split(ci_dataset, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": ci_dataset
        }

        # results == {'conditional_independence_data_validator': {'fcit_validator_method': {'0': 7.402629383549662e-06}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        p_value = results['conditional_independence_data_validator']['fcit_validator_method']['0']

        assert(p_value < 0.05)