from typing import Dict

import numpy as np
import torch

from src.datasets.synthetic_normal_dataset import SyntheticNormalDataset
from src.utils import validate_from_schema


class TestIsolationForestAnomalyValidatorMethod:
    def test_validator_method(self):
        np.random.seed(42)

        test_validation_schema : dict = {
            "default_inclusion": False,
            "validators": {
                "unsupervised_anomaly_data_validator": {
                    "include": [
                        "feature_\d+"
                    ],
                    "validator_kwargs": {

                        # "contamination": "auto",
                        # "max_features": 10,
                        # "max_samples": 10000,
                        # "n_estimators": 5000,
                    }
                }
            }
        }

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=10, dataset_size=10000)
        normal_dataset.introduce_outliers(num_outliers=100)

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

        # results are 0 if sample is anomalous, 1 if not
        # results == {'unsupervised_anomaly_data_validator': {'isolation_forest_anomaly_validator_method': {'results': array([1, 1, 1, ..., 1, 1, 1])}}}
        results = validate_from_schema(test_validation_schema, data_object)



        # _, predictions = results['unsupervised_anomaly_data_validator']['isolation_forest_anomaly_validator_method']['results']
        # outlier_indices = { i for i in range(len(predictions)) if predictions[i] == -1 }
        #
        # false_negatives = normal_dataset.outlier_index_set - outlier_indices
        # false_positives = outlier_indices - normal_dataset.outlier_index_set
        # error_rate = (len(false_negatives) + len(false_positives)) / len(normal_dataset)
        #
        # assert(error_rate < 0.01)
        # # very few examples should get through
        # assert(len(false_negatives) < 3)

    def test_negative_example(self):
        np.random.seed(42)

        test_validation_schema: dict = {
            "default_inclusion": False,
            "validators": {
                "unsupervised_anomaly_data_validator": {
                    "include": [
                        "feature_\d+"
                    ],
                    "validator_kwargs": {
                        "contamination": "auto",
                        "max_features": 10,
                        "max_samples": 10000,
                        "n_estimators": 5000,
                    }
                }
            }
        }

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=10, dataset_size=10000)
        train_size: int = int(0.6 * len(normal_dataset))
        val_size: int = int(0.2 * len(normal_dataset))
        test_size: int = len(normal_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(normal_dataset,
                                                                                 [train_size, val_size, test_size])

        # TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "training_set": train_dataset,
            "validation_set": val_dataset,
            "test_set": test_dataset,
            "entire_set": normal_dataset
        }

        # results are 0 if sample is anomalous, 1 if not
        # results == {'unsupervised_anomaly_data_validator': {'isolation_forest_anomaly_validator_method': {'results': array([1, 1, 1, ..., 1, 1, 1])}}}
        results = validate_from_schema(test_validation_schema, data_object)
        _, predictions = results['unsupervised_anomaly_data_validator']['isolation_forest_anomaly_validator_method']['results']
        outlier_indices = {i for i in range(len(predictions)) if predictions[i] == -1}

        false_negatives = normal_dataset.outlier_index_set - outlier_indices
        false_positives = outlier_indices - normal_dataset.outlier_index_set
        error_rate = (len(false_negatives) + len(false_positives)) / len(normal_dataset)

        assert (error_rate < 0.01)
        # very few examples should get through
        assert(len(false_negatives) < 3)
