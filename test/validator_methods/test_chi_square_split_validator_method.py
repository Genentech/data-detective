from typing import Dict

import numpy as np
import torch

from src.datasets.data_detective_dataset import dd_random_split
from src.datasets.synthetic_normal_dataset import SyntheticCategoricalDataset
from src.data_detective_engine import DataDetectiveEngine


class TestChiSquareSplitValidatorMethod:
    def test_positive_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "split_covariate_data_validator": {
                    "include": [
                        r"feature_\d+"
                    ]
                }
            }
        }

        normal_dataset: SyntheticCategoricalDataset = SyntheticCategoricalDataset(p=0.5, num_cols=1, dataset_size=10000)
        train_size: int = int(0.6 * len(normal_dataset))
        val_size: int = int(0.2 * len(normal_dataset))
        test_size: int = len(normal_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset=dd_random_split( normal_dataset, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "standard_split": {
                "training_set": train_dataset,
                "validation_set": val_dataset,
                "test_set": test_dataset,
            }
        }


        # results == {'split_covariate_data_validator': {'kolmogorov_smirnov_split_validator_method': {'training_set/test_set/feature_0': KstestResult(statistic=0.005366666666666672, pvalue=0.7785188817028788), 'training_set/validation_set/feature_0': KstestResult(statistic=0.006116666666666659, pvalue=0.6263395414856405), 'test_set/validation_set/feature_0': KstestResult(statistic=0.00660000000000005, pvalue=0.7737396658858053)}, 'kruskal_wallis_split_validator_method': {'training_set/test_set/feature_0': KruskalResult(statistic=0.06664652898325585, pvalue=0.7962835117722751), 'training_set/validation_set/feature_0': KruskalResult(statistic=0.5260585436481051, pvalue=0.4682686595849018), 'test_set/validation_set/feature_0': KruskalResult(statistic=0.14121920499019325, pvalue=0.707071953434097)}, 'mann_whitney_split_validator_method': {'training_set/test_set/feature_0': MannwhitneyuResult(statistic=600730191.0, pvalue=0.7962836482032007), 'training_set/validation_set/feature_0': MannwhitneyuResult(statistic=602051468.0, pvalue=0.4682687680109038), 'test_set/validation_set/feature_0': MannwhitneyuResult(statistic=200433932.0, pvalue=0.7070722753726866)}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        method_results = results['split_covariate_data_validator']['chi_square_split_validator_method']
        p_value = method_results['standard_split/training_set_vs_validation_set/feature_0'].pvalue
        assert(p_value > 0.05)
        p_value = method_results['standard_split/training_set_vs_test_set/feature_0'].pvalue
        assert(p_value > 0.05)
        p_value = method_results['standard_split/validation_set_vs_test_set/feature_0'].pvalue
        assert(p_value > 0.05)

    def test_negative_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "split_covariate_data_validator": {
                    "include": [
                        r"feature_\d+"
                    ]
                }
            }
        }

        train_dataset: SyntheticCategoricalDataset = SyntheticCategoricalDataset(num_cols=1, dataset_size=10000, p=0.2)
        val_dataset: SyntheticCategoricalDataset = SyntheticCategoricalDataset(num_cols=1, dataset_size=10000, p=0.5)
        test_dataset: SyntheticCategoricalDataset = SyntheticCategoricalDataset(num_cols=1, dataset_size=10000, p=0.8)

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "standard_split": {
                "training_set": train_dataset,
                "validation_set": val_dataset,
                "test_set": test_dataset,
            }
            # "entire_set": normal_dataset
        }

        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        method_results = results['split_covariate_data_validator']['chi_square_split_validator_method']
        p_value = method_results['standard_split/training_set_vs_validation_set/feature_0'].pvalue
        assert(p_value < 0.05)
        p_value = method_results['standard_split/training_set_vs_test_set/feature_0'].pvalue
        assert(p_value < 0.05)
        p_value = method_results['standard_split/validation_set_vs_test_set/feature_0'].pvalue
        assert(p_value < 0.05)