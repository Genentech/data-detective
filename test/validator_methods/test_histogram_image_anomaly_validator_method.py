from typing import Dict

import numpy as np
import pytest
import torch
from torchvision.transforms import transforms

from src.datasets.my_cifar_10 import MyCIFAR10
from src.datasets.my_fashion_mnist import MyFashionMNIST
from src.datasets.synthetic_anomalous_dataset import SyntheticAnomalousDataset

@pytest.fixture()
def fashion_mnist():
    fashion_mnist: MyFashionMNIST = MyFashionMNIST(
        root='./datasets/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

@pytest.fixture
def cifar_10():
    cifar_10: MyCIFAR10 = MyCIFAR10(
        root='./datasets/CIFAR10',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    yield cifar_10

@pytest.fixture()
def synthetic_anomalous_dataset():
    cifar_10: MyCIFAR10 = MyCIFAR10(
        root='./datasets/CIFAR10',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )


    synthetic_anomalous_dataset = SyntheticAnomalousDataset(cifar_10,
                                                            normal_class=0,
                                                            include_all=True,
                                                            desired_normal_class_proportion=0.99)

    yield synthetic_anomalous_dataset

class TestHistogramImageAnomalyValidatorMethod:
    def test_positive_example(self, synthetic_anomalous_dataset):
        np.random.seed(42)

        test_validation_schema : dict = {
            "default_inclusion": False,
            "transforms": {
                "cifar_image": [{
                    "name": "histogram",
                    "in_place": "False",
                    "options": {},
                }],
            },
            "validators": {
                "unsupervised_anomaly_data_validator": {
                    "include": [
                        "histogram_cifar_image",
                        "cifar_image",
                        "label"
                    ],
                    "validator_kwargs": {
                        "contamination": 0.01,
                        "max_features": 100,
                        "max_samples": 10000,
                        "n_bins": 100,
                        "n_estimators": 5000,
                    }
                }
            }
        }

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": synthetic_anomalous_dataset
        }

        # data_object: Dict[str, torch.utils.data.Dataset] = {
        #     "rename: splits": {
        #         "entire_set": synthetic_anomalous_dataset
        #     },
        #     # "transforms": {
        #     #     "column": [transform1, transform2]
        #     # }
        # }



        # takes too long

        # results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        # experiment_results = results['unsupervised_anomaly_data_validator']['histogram_cifar_image_anomaly_validator_method']['results']

        # index_lst, label, (score, anom_pred) = experiment_results
        # lst = [(l, s, a, i) for l, s, a, i in zip(label, score, anom_pred, index_lst)]

        # false_negatives = len([x for x in lst if x[0] != 0 and x[2] == 1])
        # true_negatives = len([x for x in lst if x[0] == 0 and x[2] == 1])
        # false_positives = len([x for x in lst if x[0] == 0 and x[2] != 1])
        # true_positives = len([x for x in lst if x[0] != 0 and x[2] != 1])

        # error_rate = float(false_negatives + false_positives) / len(lst)
        # recall = float(true_positives) / (true_positives + false_negatives)

        # assert(error_rate < 0.1)
        # assert(recall > 0.95)

        x=4

    def test_negative_example(self, fashion_mnist):
        np.random.seed(42)

        test_validation_schema: dict = {
            "default_inclusion": False,
            "validators": {
                "unsupervised_anomaly_data_validator": {
                    "include": [
                        "cifar_image",
                        "label"
                    ],
                    "validator_kwargs": {
                        "contamination": "auto",
                        "max_features": 100,
                        "max_samples": 10000,
                        "n_bins": 100,
                        "n_estimators": 5000,
                    }
                }
            }
        }

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": fashion_mnist
        }

        # takes too long

        # results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        # experiment_results = results['unsupervised_anomaly_data_validator']['histogram_image_anomaly_validator_method']['results']

        # index_lst, label, (score, anom_pred) = experiment_results
        # should be 0 anomalies

        # assert(error_rate < 0.1)
        # assert(recall > 0.95)

        x = 4