from typing import Dict

import numpy as np
import torch

from src.aggregation.rankings import ResultAggregator, RankingAggregationMethod
from src.datasets.adbench_dataset import ADBenchDataset
from src.data_detective_engine import DataDetectiveEngine
from src.enums.enums import DataType

INFERENCE_SIZE = 20

"""
Tests to add for coverage:
- unit test for both score and non score type aggregations
- unit test for both modal and multimodal aggregation
"""

class TestADBenchIntegration:
    SEED = 142

    def test_example(self):
        seed = TestADBenchIntegration.SEED
        finished = False

        np.random.seed(seed)
        torch.manual_seed(seed)

        npz_files = [
            # "4_breastw.npz",
            # "6_cardio.npz",
            # # "16_http.npz",
            # "21_Lymphography.npz",
            "25_musk.npz",
            # # "31_satimage-2.npz",
            # "38_thyroid.npz",
            # "42_WBC.npz",
            # "43_WDBC.npz",
        ]

        results_for_table = []

        for npz_filename in npz_files:
            print(npz_filename)
            # TODO: add proper datasets augmentation strategy
            adbench_dataset: ADBenchDataset = ADBenchDataset(
                # npz_filename="16_http.npz",
                npz_filename=npz_filename,
                input_data_type=DataType.MULTIDIMENSIONAL,
                output_data_type=DataType.CATEGORICAL,
            )

            test_validation_schema: dict = {
                "default_inclusion": False,
                "validators": {
                    "unsupervised_anomaly_data_validator": {
                        "include": [
                            adbench_dataset.input_data_name,
                            "label",
                        ],
                    },
                    "split_covariate_data_validator": {
                        "include": [
                            adbench_dataset.input_data_name,
                            "label"
                        ]
                    },
                    "ood_inference_data_validator": {
                        "include": [
                            adbench_dataset.input_data_name,
                            "label"
                        ]
                    }
                }
            }

            inference_dataset, everything_but_inference_dataset = torch.utils.data.random_split(adbench_dataset,
                                                                                                [INFERENCE_SIZE,
                                                                                                 adbench_dataset.__len__() - INFERENCE_SIZE])
            true_results = []
            for idx in range(inference_dataset.__len__()):
                sample = inference_dataset[idx]
                true_results.append(sample['label'])
            true_results = np.array(true_results)

            while len(np.unique(true_results)) < 2:
                inference_dataset, everything_but_inference_dataset = torch.utils.data.random_split(adbench_dataset,
                                                                                                    [INFERENCE_SIZE,
                                                                                                     adbench_dataset.__len__() - INFERENCE_SIZE])
                true_results = []
                for idx in range(inference_dataset.__len__()):
                    sample = inference_dataset[idx]
                    true_results.append(sample['label'])
                true_results = np.array(true_results)

            train_size: int = int(0.6 * len(everything_but_inference_dataset))
            val_size: int = int(0.2 * len(everything_but_inference_dataset))
            test_size: int = len(everything_but_inference_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(everything_but_inference_dataset,
                                                                                     [train_size, val_size, test_size])

            # TODO: lists for validation sets and test sets.
            data_object: Dict[str, torch.utils.data.Dataset] = {
                "standard_split": {
                    "training_set": train_dataset,
                    "validation_set": val_dataset,
                    "test_set": test_dataset,
                },
                "entire_set": adbench_dataset,
                "everything_but_inference_set": everything_but_inference_dataset,
                "inference_set": inference_dataset
            }

            
            results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)

            aggregator = ResultAggregator(results)

            aggregated_results = aggregator.aggregate_results_multimodally(
                validator_name="unsupervised_anomaly_data_validator",
                aggregation_methods=[RankingAggregationMethod.ROUND_ROBIN]
            )

            aggregated_results = aggregator.aggregate_results_modally(
                validator_name="unsupervised_anomaly_data_validator",
                aggregation_methods=[RankingAggregationMethod.ROUND_ROBIN],
                given_data_modality=adbench_dataset.input_data_name
            )

            x=3
