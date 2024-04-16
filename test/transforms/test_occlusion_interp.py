import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import time
import torch
import torchvision.transforms as transforms

import torch.nn as nn
import copy

from torch.nn.functional import pad

from constants import FloatTensor

from torchvision.datasets import MNIST
from typing import Dict, Union
from tqdm import tqdm

from constants import FloatTensor
from src.aggregation.rankings import ResultAggregator, RankingAggregationMethod
from src.data_detective_engine import DataDetectiveEngine
from src.datasets.data_detective_dataset import dd_random_split
from src.datasets.my_cifar_10 import MyCIFAR10
from src.datasets.my_fashion_mnist import MyFashionMNIST
from src.enums.enums import DataType
from src.transforms.embedding_transformer import Transform
from src.transforms.transform_library import TRANSFORM_LIBRARY
from src.utils import OcclusionTransform, occlusion_interpretability


class TestOcclusionInterp:
    def test_occlusion_interp(self):
        """
        Runs the main body of the Interpretability.ipynb notebook.
        """
        dataset: MyFashionMNIST = MyFashionMNIST(
            root='./datasets/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        percent_to_keep = 0.01
        dataset, _ = dd_random_split(dataset, [int(dataset.__len__() * percent_to_keep), dataset.__len__() - int(dataset.__len__() * percent_to_keep)])
        print(dataset.__len__())
        print(_.__len__())

        inference_size: int = 20
        everything_but_inference_size: int = dataset.__len__() - inference_size
        inference_dataset, everything_but_inference_dataset = dd_random_split(dataset, [inference_size, dataset.__len__() - inference_size])
            
        train_size: int = int(0.6 * len(everything_but_inference_dataset))
        val_size: int = int(0.2 * len(everything_but_inference_dataset))
        test_size: int = len(everything_but_inference_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = dd_random_split(everything_but_inference_dataset, [train_size, val_size, test_size])

        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": dataset,
            "everything_but_inference_set": everything_but_inference_dataset,
            "inference_set": inference_dataset,
            "train/val/test":{
                "training_set": train_dataset,
                "validation_set": val_dataset,
                "test_set": test_dataset,
            }
        }

        print(f"size of inference_dataset: {inference_dataset.__len__()}")
        print(f"size of everything_but_inference_dataset: {everything_but_inference_dataset.__len__()}")
        print(f"size of train_dataset: {train_dataset.__len__()}")
        print(f"size of entire dataset: {dataset.__len__()}")
        print(f"size of val_dataset: {val_dataset.__len__()}")
        print(f"size of test_dataset: {test_dataset.__len__()}")

        validation_schema : Dict = {
            "validators": {
                "unsupervised_anomaly_data_validator": {
                    "validator_kwargs": {
                        "should_return_model_instance": True
                    }
                },
            }
        }

        transform_schema : Dict = {
            "transforms": {
                "IMAGE": [{
                    "name": "resnet50",
                    "in_place": "False",
                    "options": {},
                }],
            }
        }
            
        full_validation_schema: Dict = {
            **validation_schema, 
            **transform_schema
        }

        data_detective_engine = DataDetectiveEngine()

        start_time = time.time()
        results = data_detective_engine.validate_from_schema(full_validation_schema, data_object)

        # results = data_detective_engine.validate_from_schema(full_validation_schema, data_object)
        print("--- %s seconds ---" % (time.time() - start_time))

        resnet = TRANSFORM_LIBRARY['resnet50']()
        resnet.initialize_transform(transform_kwargs={})
        print(resnet)


        METHOD = 'cblof_anomaly_validator_method'

        sample = dataset[7]
        img = sample['fashion_mnist_image']
        occ = OcclusionTransform(width=5)
        occed = occ(img, (15, 15))

        model_results, model = results['unsupervised_anomaly_data_validator'][METHOD]['resnet50_backbone_fashion_mnist_image_results']
        res_min, res_max = min(model_results.values()), max(model_results.values())
        interp_results = occlusion_interpretability(img, model, occ, (res_min, res_max))