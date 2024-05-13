from typing import Dict, Union

import numpy as np
import torch

from src.actions.action import RemoveTopKAnomalousSamplesAction
from src.aggregation.rankings import RankingAggregationMethod, ResultAggregator
from src.datasets.data_detective_dataset import DataDetectiveDataset, dd_random_split
from src.data_detective_engine import DataDetectiveEngine
from src.datasets.adbench_dataset import ADBenchDDDataset
from src.enums.enums import DataType

SEED = 142
INFERENCE_SIZE = 20

class TestAction:

    def test_remove_top_k_action(self):
        seed = SEED
        finished = False
        INFERENCE_SIZE = 20

        np.random.seed(seed)
        torch.manual_seed(seed)

        npz_files = [
            # "4_breastw.npz",
            "6_cardio.npz",
            # # "16_http.npz",
            # "21_Lymphography.npz",
            # "25_musk.npz",
            # # "31_satimage-2.npz",
            # "38_thyroid.npz",
            # "42_WBC.npz",
            # "43_WDBC.npz",
        ]

        results_for_table = []

        for npz_filename in npz_files:
            print(npz_filename)
            # TODO: add proper datasets augmentation strategy
            adbench_dataset: ADBenchDDDataset = ADBenchDDDataset(
                # npz_filename="16_http.npz",
                npz_filename=npz_filename,
                input_data_type=DataType.MULTIDIMENSIONAL,
                output_data_type=DataType.CATEGORICAL,
            )

            test_validation_schema : dict = {
                "default_inclusion": False,
                "validators": {
                    "unsupervised_anomaly_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label",
                        # ],
                    },
                }
            }


            inference_dataset, everything_but_inference_dataset = dd_random_split( adbench_dataset, [INFERENCE_SIZE, adbench_dataset.__len__() - INFERENCE_SIZE])
            true_results = []
            for idx in range(inference_dataset.__len__()):
                sample = inference_dataset[idx]
                true_results.append(sample['label'])
            true_results = np.array(true_results)

            while len(np.unique(true_results)) < 2:
                inference_dataset, everything_but_inference_dataset = dd_random_split(adbench_dataset,
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
            train_dataset, val_dataset, test_dataset = dd_random_split( everything_but_inference_dataset, [train_size, val_size, test_size])

            #TODO: lists for validation sets and test sets.
            data_object: Dict[str, Union[Dict, DataDetectiveDataset]] = {
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
            aggregator = ResultAggregator(results_object=results)
            total_rankings = aggregator.aggregate_results_multimodally("unsupervised_anomaly_data_validator", [RankingAggregationMethod.LOWEST_RANK, RankingAggregationMethod.HIGHEST_RANK, RankingAggregationMethod.ROUND_ROBIN])
            total_rankings

            action = RemoveTopKAnomalousSamplesAction()
            new_data_object = action.get_new_data_object(
                data_object,
                total_rankings, 
                "round_robin_agg_rank"
            )

            c=3

    def test_repeat_remove_top_k_action_with_duplicates(self):
        seed = SEED
        finished = False
        INFERENCE_SIZE = 20

        np.random.seed(seed)
        torch.manual_seed(seed)

        npz_files = [
            # "4_breastw.npz",
            "6_cardio.npz",
            # # "16_http.npz",
            # "21_Lymphography.npz",
            # "25_musk.npz",
            # # "31_satimage-2.npz",
            # "38_thyroid.npz",
            # "42_WBC.npz",
            # "43_WDBC.npz",
        ]

        results_for_table = []

        for npz_filename in npz_files:
            print(npz_filename)
            # TODO: add proper datasets augmentation strategy
            adbench_dataset: ADBenchDDDataset = ADBenchDDDataset(
                # npz_filename="16_http.npz",
                npz_filename=npz_filename,
                input_data_type=DataType.MULTIDIMENSIONAL,
                output_data_type=DataType.CATEGORICAL,
            )

            test_validation_schema : dict = {
                "default_inclusion": False,
                "validators": {
                    "unsupervised_anomaly_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label",
                        # ],
                    },
                }
            }


            inference_dataset, everything_but_inference_dataset = dd_random_split( adbench_dataset, [INFERENCE_SIZE, adbench_dataset.__len__() - INFERENCE_SIZE])
            true_results = []
            for idx in range(inference_dataset.__len__()):
                sample = inference_dataset[idx]
                true_results.append(sample['label'])
            true_results = np.array(true_results)

            while len(np.unique(true_results)) < 2:
                inference_dataset, everything_but_inference_dataset = dd_random_split(adbench_dataset,
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
            train_dataset, val_dataset, test_dataset = dd_random_split( everything_but_inference_dataset, [train_size, val_size, test_size])

            #TODO: lists for validation sets and test sets.
            data_object: Dict[str, Union[Dict, DataDetectiveDataset]] = {
                "standard_split": {
                    "training_set": train_dataset,
                    "validation_set": val_dataset,
                    "test_set": test_dataset,
                },
                "entire_set": adbench_dataset,
                "everything_but_inference_set": everything_but_inference_dataset,
                "inference_set": inference_dataset
            }

            for _ in range(2): 
                results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
                aggregator = ResultAggregator(results_object=results)
                total_rankings = aggregator.aggregate_results_multimodally("unsupervised_anomaly_data_validator", [RankingAggregationMethod.LOWEST_RANK, RankingAggregationMethod.HIGHEST_RANK, RankingAggregationMethod.ROUND_ROBIN])
                total_rankings

                action = RemoveTopKAnomalousSamplesAction()
                new_data_object = action.get_new_data_object(
                    data_object,
                    total_rankings, 
                    "round_robin_agg_rank"
                )

                data_object = new_data_object