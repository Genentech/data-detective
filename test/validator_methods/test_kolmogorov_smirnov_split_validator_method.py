from typing import Dict

import numpy as np
import torch

import src.utils

from src.datasets.synthetic_data_generators import SyntheticNormalDataset


class TestKolmogorovSmirnovSplitValidatorMethod:
    def test_positive_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "split_covariate_data_validator": {
                    "include": [
                        "feature_\d+"
                    ]
                }
            }
        }

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size=100000)
        train_size: int = int(0.6 * len(normal_dataset))
        val_size: int = int(0.2 * len(normal_dataset))
        test_size: int = len(normal_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset=torch.utils.data.random_split( normal_dataset, [train_size, val_size, test_size])

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "training_set": train_dataset,
            "validation_set": val_dataset,
            "test_set": test_dataset,
            "entire_set": normal_dataset
        }

        # results == {'split_covariate_data_validator': {'kolmogorov_smirnov_split_validator_method': {'training_set/test_set/feature_0': KstestResult(statistic=0.011366666666666636, pvalue=0.04115018947623861), 'training_set/validation_set/feature_0': KstestResult(statistic=0.006866666666666688, pvalue=0.4770160007255947), 'test_set/validation_set/feature_0': KstestResult(statistic=0.009550000000000058, pvalue=0.3193731089904299)}, 'kruskal_wallis_split_validator_method': {'training_set/test_set/feature_0': KruskalResult(statistic=2.4670793302357197, pvalue=0.11625376658541685), 'training_set/validation_set/feature_0': KruskalResult(statistic=0.8968041600601282, pvalue=0.34364008108600275), 'test_set/validation_set/feature_0': KruskalResult(statistic=0.24699579428124707, pvalue=0.6191984201628177)}, 'mann_whitney_split_validator_method': {'training_set/test_set/feature_0': MannwhitneyuResult(statistic=595557379.0, pvalue=0.11625380766824382), 'training_set/validation_set/feature_0': MannwhitneyuResult(statistic=597321470.0, pvalue=0.34364017116496304), 'test_set/validation_set/feature_0': MannwhitneyuResult(statistic=200573878.0, pvalue=0.6191987255241788)}}}
        results = src.utils.validate_from_schema(test_validation_schema, data_object)
        method_results = results['split_covariate_data_validator']['kolmogorov_smirnov_split_validator_method']
        p_value = method_results['training_set/validation_set/feature_0'].pvalue
        assert(p_value > 0.05)
        p_value = method_results['training_set/test_set/feature_0'].pvalue
        assert(p_value > 0.05)
        p_value = method_results['test_set/validation_set/feature_0'].pvalue
        assert(p_value > 0.05)

    def test_negative_example(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "split_covariate_data_validator": {
                    "include": [
                        "feature_\d+"
                    ]
                }
            }
        }

        train_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size=10000, loc=-1)
        val_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size=10000, loc=0)
        test_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size=10000, loc=1)

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "training_set": train_dataset,
            "validation_set": val_dataset,
            "test_set": test_dataset,
        }

        # {'split_covariate_data_validator': {'kolmogorov_smirnov_split_validator_method': {'training_set/test_set/feature_0': KstestResult(statistic=0.6772, pvalue=0.0), 'training_set/validation_set/feature_0': KstestResult(statistic=0.3886, pvalue=0.0), 'test_set/validation_set/feature_0': KstestResult(statistic=0.3776, pvalue=0.0)}, 'kruskal_wallis_split_validator_method': {'training_set/test_set/feature_0': KruskalResult(statistic=10622.043853937459, pvalue=0.0), 'training_set/validation_set/feature_0': KruskalResult(statistic=4177.829853541334, pvalue=0.0), 'test_set/validation_set/feature_0': KruskalResult(statistic=3899.0755428250777, pvalue=0.0)}, 'mann_whitney_split_validator_method': {'training_set/test_set/feature_0': MannwhitneyuResult(statistic=7923532.0, pvalue=0.0), 'training_set/validation_set/feature_0': MannwhitneyuResult(statistic=23611749.0, pvalue=0.0), 'test_set/validation_set/feature_0': MannwhitneyuResult(statistic=75492713.0, pvalue=0.0)}}}
        results = src.utils.validate_from_schema(test_validation_schema, data_object)
        method_results = results['split_covariate_data_validator']['kolmogorov_smirnov_split_validator_method']
        p_value = method_results['training_set/validation_set/feature_0'].pvalue
        assert(p_value < 0.05)
        p_value = method_results['training_set/test_set/feature_0'].pvalue
        assert(p_value < 0.05)
        p_value = method_results['test_set/validation_set/feature_0'].pvalue
        assert(p_value < 0.05)