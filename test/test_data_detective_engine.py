from typing import Dict

import torch
from torchvision.transforms import transforms

from constants import SEED
from src.datasets.synthetic_normal_dataset import SyntheticNormalDataset
from src.data_detective_engine import DataDetectiveEngine


class TestDataDetectiveEngine:
    def test_validate_from_schema(self):
        """
        Tests for functionality over validate_from_schema in the data detective engine.
        """
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
        train_dataset, val_dataset, test_dataset=torch.utils.data.random_split( normal_dataset, [train_size, val_size, test_size])

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": normal_dataset
        }

        data_detective_engine = DataDetectiveEngine()
        results = data_detective_engine.validate_from_schema(test_validation_schema, data_object)

    def test_split(self):
        pass


    # def test_anomaly_loader_reproducibility(self):
    #     MyFashionMNIST = synthetic_data_generators.MyFashionMNIST
    #     fashion_mnist: MyFashionMNIST = MyFashionMNIST(
    #         root='./datasets/FashionMNIST',
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor()
    #         ])
    #     )
    #
    #     anomaly_loader_1 = iter(load_dataset_as_anomaly(
    #         train_dataset=fashion_mnist,
    #         normal_class_number=0,
    #         normal_class_proportion=0.8,
    #         random_seed=SEED,
    #     ))
    #
    #     anomaly_loader_2 = iter(load_dataset_as_anomaly(
    #         train_dataset=fashion_mnist,
    #         normal_class_number=0,
    #         normal_class_proportion=0.8,
    #         random_seed=SEED,
    #     ))
    #
    #     assert(len(anomaly_loader_1) == len(anomaly_loader_2))
    #
    #     for sample_1, sample_2 in zip(anomaly_loader_1, anomaly_loader_2):
    #         for key in sample_1.keys():
    #             assert(torch.equal(sample_1[key], sample_2[key]))



