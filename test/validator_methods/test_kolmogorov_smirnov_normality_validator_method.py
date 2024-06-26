from typing import Dict

import numpy as np
import torch

from src.datasets.data_detective_dataset import dd_random_split
from src.data_detective_engine import DataDetectiveEngine
from src.datasets.synthetic_normal_dataset import SyntheticNormalDataset


class TestKolmogorovSmirnovNormalityValidatorMethod:
    def test_positive_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "normality_data_validator": {
                    "include": [
                        r"feature_\d+"
                    ]
                }
            }
        }

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size=10000)
        train_size: int = int(0.6 * len(normal_dataset))
        val_size: int = int(0.2 * len(normal_dataset))
        test_size: int = len(normal_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset=dd_random_split( normal_dataset, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": normal_dataset
        }

        # results == {'normality_data_validator': {'kolmogorov_smirnov_normality_validator_method': {'feature_0': KstestResult(statistic=0.003674476700887941, pvalue=0.9992074462469855)}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        d_statistic, p_value = results['normality_data_validator']['kolmogorov_smirnov_normality_validator_method']['feature_0']

        assert(p_value > 0.05)

    def test_negative_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "normality_data_validator": {
                    "include": [
                        r"feature_\d+"
                    ]
                }
            }
        }

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size=10000)
        normal_dataset.introduce_outliers()

        train_size: int = int(0.6 * len(normal_dataset))
        val_size: int = int(0.2 * len(normal_dataset))
        test_size: int = len(normal_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset=dd_random_split( normal_dataset, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": normal_dataset
        }

        # results == {'normality_data_validator': {'kolmogorov_smirnov_normality_validator_method': {'feature_0': KstestResult(statistic=0.10025403366385621, pvalue=5.984973888392497e-88)}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        d_statistic, p_value = results['normality_data_validator']['kolmogorov_smirnov_normality_validator_method']['feature_0']

        assert(p_value < 0.05)