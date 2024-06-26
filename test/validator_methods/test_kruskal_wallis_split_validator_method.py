from typing import Dict

import numpy as np
import torch
from src.datasets.data_detective_dataset import dd_random_split
from src.datasets.synthetic_normal_dataset import SyntheticNormalDatasetContinuous

from src.data_detective_engine import DataDetectiveEngine


class TestKruskalWallisSplitValidatorMethod:
    def test_positive_example(self):
        np.random.seed(43)
        torch.manual_seed(42)

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

        normal_dataset: SyntheticNormalDatasetContinuous = SyntheticNormalDatasetContinuous(num_cols=1, dataset_size=100000)
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
            # "entire_set": normal_dataset
        }

        # results == {'split_covariate_data_validator': {'kolmogorov_smirnov_split_validator_method': {'training_set/test_set/feature_0': KstestResult(statistic=0.004716666666666591, pvalue=0.8907784830726827), 'training_set/validation_set/feature_0': KstestResult(statistic=0.006683333333333263, pvalue=0.5121389366814817), 'test_set/validation_set/feature_0': KstestResult(statistic=0.005699999999999927, pvalue=0.8994451986334527)}, 'kruskal_wallis_split_validator_method': {'training_set/test_set/feature_0': KruskalResult(statistic=0.005694027786375955, pvalue=0.9398496911671448), 'training_set/validation_set/feature_0': KruskalResult(statistic=0.0029069187003187835, pvalue=0.9570022098962008), 'test_set/validation_set/feature_0': KruskalResult(statistic=8.62188171595335e-05, pvalue=0.9925914221129283)}, 'mann_whitney_split_validator_method': {'training_set/test_set/feature_0': MannwhitneyuResult(statistic=599786569.0, pvalue=0.9398498317683777), 'training_set/validation_set/feature_0': MannwhitneyuResult(statistic=599847502.0, pvalue=0.9570023510060811), 'test_set/validation_set/feature_0': MannwhitneyuResult(statistic=200010722.0, pvalue=0.9925917680015839)}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        method_results = results['split_covariate_data_validator']['kruskal_wallis_split_validator_method']
        p_value = method_results['standard_split/training_set_vs_validation_set/feature_0'].pvalue
        assert(p_value > 0.05)
        p_value = method_results['standard_split/training_set_vs_test_set/feature_0'].pvalue
        assert(p_value > 0.05)
        p_value = method_results['standard_split/validation_set_vs_test_set/feature_0'].pvalue
        assert(p_value > 0.05)

    def test_negative_example(self):
        np.random.seed(42)
        torch.manual_seed(42)

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

        train_dataset: SyntheticNormalDatasetContinuous = SyntheticNormalDatasetContinuous(num_cols=1, dataset_size=10000, loc=-1)
        val_dataset: SyntheticNormalDatasetContinuous = SyntheticNormalDatasetContinuous(num_cols=1, dataset_size=10000, loc=0)
        test_dataset: SyntheticNormalDatasetContinuous = SyntheticNormalDatasetContinuous(num_cols=1, dataset_size=10000, loc=1)

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "standard_split": {
                "training_set": train_dataset,
                "validation_set": val_dataset,
                "test_set": test_dataset,
            }
            # "entire_set": normal_dataset
        }

        # results == {'split_covariate_data_validator': {'mann_whitney_split_validator_method': {'training_set/test_set/feature_0': MannwhitneyuResult(statistic=7923532.0, pvalue=0.0), 'training_set/validation_set/feature_0': MannwhitneyuResult(statistic=23611749.0, pvalue=0.0), 'test_set/validation_set/feature_0': MannwhitneyuResult(statistic=75492713.0, pvalue=0.0)}, 'kolmogorov_smirnov_split_validator_method': {'training_set/test_set/feature_0': KstestResult(statistic=0.6772, pvalue=0.0), 'training_set/validation_set/feature_0': KstestResult(statistic=0.3886, pvalue=0.0), 'test_set/validation_set/feature_0': KstestResult(statistic=0.3776, pvalue=0.0)}, 'kruskal_wallis_split_validator_method': {'training_set/test_set/feature_0': KruskalResult(statistic=10622.043853937459, pvalue=0.0), 'training_set/validation_set/feature_0': KruskalResult(statistic=4177.829853541334, pvalue=0.0), 'test_set/validation_set/feature_0': KruskalResult(statistic=3899.0755428250777, pvalue=0.0)}}}
        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        method_results = results['split_covariate_data_validator']['kruskal_wallis_split_validator_method']
        p_value = method_results['standard_split/training_set_vs_validation_set/feature_0'].pvalue
        assert(p_value < 0.05)
        p_value = method_results['standard_split/training_set_vs_test_set/feature_0'].pvalue
        assert(p_value < 0.05)
        p_value = method_results['standard_split/validation_set_vs_test_set/feature_0'].pvalue
        assert(p_value < 0.05)