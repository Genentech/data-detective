import json
from typing import Dict

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torchvision.transforms import transforms
from src.data_detective_engine import DataDetectiveEngine

from src.datasets.adbench_dataset import ADBenchDataset
from src.enums.enums import DataType

INFERENCE_SIZE = 20

class TestADBenchIntegration:
    SEED = 142

    def test_example(self):
        seed = TestADBenchIntegration.SEED
        finished = False

        np.random.seed(seed)
        torch.manual_seed(seed)

        npz_files = [
            "4_breastw.npz",
            "6_cardio.npz",
            # "16_http.npz",
            "21_Lymphography.npz",
            "25_musk.npz",
            # "31_satimage-2.npz",
            "38_thyroid.npz",
            "42_WBC.npz",
            "43_WDBC.npz",
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

            test_validation_schema : dict = {
                "default_inclusion": False,
                "validators": {
                    "unsupervised_anomaly_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label",
                        # ],
                    },
                    "split_covariate_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label"
                        # ]
                    },
                    "ood_inference_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label"
                        # ]
                    }
                }
            }


            inference_dataset, everything_but_inference_dataset = torch.utils.data.random_split( adbench_dataset, [INFERENCE_SIZE, adbench_dataset.__len__() - INFERENCE_SIZE])
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
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split( everything_but_inference_dataset, [train_size, val_size, test_size])

            #TODO: lists for validation sets and test sets.
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
            # print(results)

            auc_dict = {}

            #############
            anomaly_results_dict = results['unsupervised_anomaly_data_validator']
            for validator_method in anomaly_results_dict.keys():
                predicted_scores = anomaly_results_dict[validator_method]
                result_key = list(predicted_scores.keys())[0]
                predicted_scores = predicted_scores[result_key]

                true_scores = adbench_dataset.y
                roc_auc_score = sklearn.metrics.roc_auc_score(true_scores, predicted_scores)
                auc_dict[validator_method] = roc_auc_score
                print(validator_method, roc_auc_score)

                results_for_table.append({
                    "validator": "unsupervised anomaly datasets validator",
                    "validator_method": validator_method,
                    "dataset": npz_filename.split(".")[0].split("_")[1],
                    "metric_value": roc_auc_score,
                    "metric_unit": "aucroc"
                })

            iforest_results = anomaly_results_dict['iforest_anomaly_validator_method'][f'{adbench_dataset.input_data_name}_results']
            pca_results = anomaly_results_dict['pca_anomaly_validator_method'][f'{adbench_dataset.input_data_name}_results']
            cblof_results = anomaly_results_dict['cblof_anomaly_validator_method'][f'{adbench_dataset.input_data_name}_results']
            true_results = adbench_dataset.y
            np.savez(f"anomaly_results_{npz_filename}", iforest_results=iforest_results, pca_results=pca_results, cblof_results=cblof_results, true_results=true_results)

            ood_inference_results_dict = results['ood_inference_data_validator']
            iforest_results = ood_inference_results_dict['iforest_ood_inference_validator_method']['results']['ood_scores']
            pca_results = ood_inference_results_dict['pca_ood_inference_validator_method']['results']['ood_scores']
            cblof_results = ood_inference_results_dict['cblof_ood_inference_validator_method']['results']['ood_scores']

            true_results = []
            for idx in range(inference_dataset.__len__()):
                sample = inference_dataset[idx]
                true_results.append(sample['label'])
            true_results = np.array(true_results)

            for validator_method in ood_inference_results_dict.keys():
                predicted_results = ood_inference_results_dict[validator_method]['results']['ood_scores']

                roc_auc_score = sklearn.metrics.roc_auc_score(true_results, predicted_results)
                auc_dict[validator_method] = roc_auc_score
                print(validator_method, roc_auc_score)

                results_for_table.append({
                    "validator": "ood inference datasets validator",
                    "validator_method": validator_method,
                    "dataset": npz_filename.split(".")[0].split("_")[1],
                    "metric_value": roc_auc_score,
                    "metric_unit": "aucroc"
                })

            np.savez(f"ood_inference_results_{npz_filename}", iforest_results=iforest_results, pca_results=pca_results, cblof_results=cblof_results, true_results=true_results)

            split_results_dict = results['split_covariate_data_validator']
            ks_results = split_results_dict['kolmogorov_smirnov_multidimensional_split_validator_method']
            ks_results = {k: ([{"statistic": str(val.statistic), "pvalue": str(val.pvalue)} for val in v[0]], v[1]) for k, v in ks_results.items()}
            kruskal_wallis_results = split_results_dict['kruskal_wallis_multidimensional_split_validator_method']
            kruskal_wallis_results = {k: ([{"statistic": str(val.statistic), "pvalue": str(val.pvalue)} for val in v[0]], v[1]) for k, v in kruskal_wallis_results.items()}
            mann_whitney_results = split_results_dict['mann_whitney_multidimensional_split_validator_method']
            mann_whitney_results = {k: ([{"statistic": str(val.statistic), "pvalue": str(val.pvalue)} for val in v[0]], v[1]) for k, v in mann_whitney_results.items()}

            results_dict = {
                "kolmogorov_smirnov": ks_results,
                "kruskal_wallis": kruskal_wallis_results,
                "mann_whitney": mann_whitney_results,
            }

            for validator_method, res in results_dict.items():
                min_pvalue = np.inf
                has_significant_result = False

                for dataset_split_name, (pvals_and_stats_list_of_dicts, has_significant_result_dict) in res.items():
                    min_pvalue = min(min_pvalue, float(min(pvals_and_stats_list_of_dicts, key=lambda d: d['pvalue'])['pvalue']))
                    has_significant_result = has_significant_result or has_significant_result_dict['has_significant_result']

                results_for_table.append({
                    "validator": "split distribution shift datasets validator",
                    "validator_method": validator_method,
                    "dataset": npz_filename.split(".")[0].split("_")[1],
                    "metric_value": min_pvalue,
                    "metric_unit": "p-value",
                })

                results_for_table.append({
                    "validator": "split distribution shift datasets validator",
                    "validator_method": validator_method,
                    "dataset": npz_filename.split(".")[0].split("_")[1],
                    "metric_value": has_significant_result,
                    "metric_unit": "has significant result",
                })



            with open(f"split_results_{npz_filename.split('.')[0]}.json", "w") as fp:
                json.dump(results_dict, fp)

        df = pd.DataFrame.from_dict(results_for_table)
        df.to_csv("results.csv")
            ###############
