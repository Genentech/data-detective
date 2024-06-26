from typing import Dict

import pytest
import torch
from torchvision.models import resnet50, ResNet50_Weights

import torchvision.transforms as transforms

from src.datasets.data_detective_dataset import dd_random_split
from src.datasets.my_cifar_10 import MyCIFAR10
from src.transforms.embedding_transformer import TransformedDataset
from src.data_detective_engine import DataDetectiveEngine
from src.transforms.transform_library import GaussianBlurTransform


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

class TestEmbeddingTransformer:
    def test_embedding_transformer(self, cifar_10):
        gaussian_blur = GaussianBlurTransform()
        gaussian_blur.initialize_transform({"kernel_size": 9})
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-1]
        backbone = torch.nn.Sequential(*modules)
        transform_dict = {
            "cifar_image": [gaussian_blur]
        }
        transformed_dataset = TransformedDataset(cifar_10, transform_dict)
        x = transformed_dataset[0]

        assert("blurred_cifar_image" in x.keys())
        assert("blurred_cifar_image" in transformed_dataset.datatypes())
        assert("cifar_image" in x.keys())
        assert("cifar_image" in transformed_dataset.datatypes())


    def test_embedding_transformer_integration(self, cifar_10):
        test_validation_schema : dict = {
            "default_inclusion": False,
            "transforms": {
                "cifar_image": [{
                    "name": "resnet50",
                    "in_place": "False",
                    "options": {},
                }],
            },
            "validators": {
                "unsupervised_anomaly_data_validator": {
                    "include": [
                        "cifar_image",
                        "label",
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

        cifar_10, _ = dd_random_split(cifar_10, [100, len(cifar_10) - 100])

        train_size: int = int(0.6 * len(cifar_10))
        val_size: int = int(0.2 * len(cifar_10))
        test_size: int = len(cifar_10) - train_size - val_size
        train_dataset, val_dataset, test_dataset = dd_random_split(cifar_10, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            # "standard_split": {
            #     "training_set": train_dataset,
            #     "validation_set": val_dataset,
            #     "test_set": test_dataset,
            # },
            "entire_set": cifar_10
        }

        results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
        c=3

