import os
from collections import defaultdict
from typing import List, Set, Dict, Type, Union

import random
import numpy as np
import pandas as pd
import torch
import typing
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from tqdm import tqdm

from constants import SEED, DEVICE
from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset, get_class_counts
from src.validator_methods.data_validator_method import DataValidatorMethod

from scipy import stats

from src.validator_methods.isolation_forest_anomaly_validator_method import IsolationForestAnomalyValidatorMethod

"""
TODO: this was originally framed as an image anomaly detection task. Need to spend some effort
to turn it back into a method for tabular outlier detection. 
"""

class HistogramImageAnomalyValidatorMethod(DataValidatorMethod):
    """
    A method for determining outliers of an image dataset by histogram analysis.
    src: https://www.dfki.de/fileadmin/user_upload/import/6431_HBOS-poster.pdf
    src: https://medium.com/dataman-in-ai/anomaly-detection-with-histogram-based-outlier-detection-hbo-bc10ef52f23f

    Basically, creates a histogram of intensity values for the image.
    """
    DEFAULT_CONTAMINATION = "auto"
    DEFAULT_MAX_FEATURES = 256
    DEFAULT_MAX_SAMPLES = 10000
    DEFAULT_N_BINS = 256
    DEFAULT_N_ESTIMATORS = 5000

    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        #TODO: make it s you don't need to include categorical to include image label.
        return {
            DataType.IMAGE,
            DataType.CATEGORICAL
        }

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return { ValidatorMethodParameter.ENTIRE_SET }

    @staticmethod
    def get_method_kwargs(data_object: typing.Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
        """
        Gets the arguments for each run of the validator_method, and what to store the results under.

        @param data_object: the datasets object containing the datasets (train, test, entire, etc.)
        @param validator_kwargs: the kwargs from the validation schema.
        @return: a dict mapping from the key the result from calling .validate() on the kwargs values.
        """
        kwargs_dict = {}
        entire_dataset: Dataset = data_object["entire_set"]

        # for normal_label in get_class_counts(entire_dataset).keys():
        #     kwargs_dict[f'normal_class_is_{normal_label}'] = {
        #         "dataset": entire_dataset,
        #         "n_bins" : validator_kwargs.get("n_bins", HistogramImageAnomalyValidatorMethod.DEFAULT_N_BINS),
        #         "normal_class": normal_label,
        #     }
        kwargs_dict['results'] = {
            "contamination" : validator_kwargs.get("contamination", HistogramImageAnomalyValidatorMethod.DEFAULT_CONTAMINATION),
            "dataset": entire_dataset,
            "max_features": validator_kwargs.get("max_features", HistogramImageAnomalyValidatorMethod.DEFAULT_MAX_FEATURES),
            "max_samples": validator_kwargs.get("max_samples", HistogramImageAnomalyValidatorMethod.DEFAULT_MAX_SAMPLES),
            "n_bins" : validator_kwargs.get("n_bins", HistogramImageAnomalyValidatorMethod.DEFAULT_N_BINS),
            "n_estimators": validator_kwargs.get("n_estimators", HistogramImageAnomalyValidatorMethod.DEFAULT_N_ESTIMATORS),
        }

        return kwargs_dict

    @staticmethod
    def validate(
            contamination: float = DEFAULT_CONTAMINATION,
            dataset: Dataset = None,
            max_features: int = DEFAULT_MAX_FEATURES,
            max_samples: int = DEFAULT_MAX_SAMPLES,
            n_bins: int = DEFAULT_N_BINS,
            n_estimators: int = DEFAULT_N_ESTIMATORS,
    ) -> object:
        if not dataset:
            raise Exception("No dataset present.")

        histograms = None
        labels: List[int] = []

        idx_lst = np.arange(dataset.__len__())
        np.random.shuffle(idx_lst)

        idx_lst = idx_lst[:10000]

        for idx in tqdm(idx_lst):
            sample = dataset[idx]
            image = sample['image']
            if 'label' in sample.keys():
                labels.append(sample['label'])
            histogram, _ = np.histogram(image, bins=n_bins)
            flattened_histogram = histogram.flatten()
            flattened_histogram = flattened_histogram.reshape((1, flattened_histogram.shape[0]))

            histograms = flattened_histogram if histograms is None else np.append(histograms, flattened_histogram, axis=0)

            # if histograms.shape[0] > 10000:
            #     break

        results = IsolationForestAnomalyValidatorMethod.validate(
            contamination=contamination,
            data_matrix=histograms,
            max_features=max_features,
            max_samples=max_samples,
            n_estimators=n_estimators,
        )

        return idx_lst, np.array(labels), results




# def load_dataset_as_anomaly(
#         train_dataset: Dataset,
#         normal_class_number: int,
#         normal_class_proportion: float = 0.5,
#         random_seed: int = SEED,
# ) -> DataLoader:
#     """
#     Produces a dataloader that equally samples between a single class and all other anomaly classes.
#     Does NOT provide coverage over the entire dataset.
#
#     @param dataset: the unfiltered, raw __training__ dataset.
#         - expects a "label" attribute that is an integer.
#     @param anomaly_class_number: the class number for the normal class (all others are anomaly)
#     @return: a dataloader that has equal probability of sampling from normal vs non normal class.
#     """
#     class_counts: defaultdict = defaultdict(lambda: 0)
#     classes: List[int] = []
#     dataset_length = train_dataset.__len__()
#
#     for idx in range(dataset_length):
#         sample: dict[str, Union[int, torch.FloatTensor]] = train_dataset[idx]
#         sample_class: int = sample['label']
#         class_counts[sample_class] += 1
#         classes.append(sample_class)
#
#     normalized_class_proportion = {k: float(v) / dataset_length for k, v in class_counts.items()}
#     ideal_class_proportion = {
#         class_number: (1 - normal_class_proportion) / (len(class_counts.values()) - 1)
#             for class_number in class_counts.keys()
#             if class_number != normal_class_number
#     }
#     ideal_class_proportion[normal_class_number] = normal_class_proportion
#     # what do we want: 50% for a single class, and the rest in the rest...
#     # since the weighted random sampler doesn't need to be normalized, we can just enter the raw % for each entry.
#
#     # how much to over/under - sample by.
#     sampling_factor = {
#         class_number: ideal_class_proportion[class_number] / normalized_class_proportion[class_number]
#             for class_number in class_counts.keys()
#     }
#
#
#     """
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         torch.use_deterministic_algorithms(True)
#         if DEVICE.type == "cuda":
#             torch.set_default_tensor_type('torch.cuda.FloatTensor')
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#
#         def seed_worker(worker_id):
#             worker_seed = torch.initial_seed() % 2 ** 32
#             np.random.seed(worker_seed)
#             random.seed(worker_seed)
#
#         random.seed(SEED)
#         np.random.seed(SEED)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(SEED)
#         else:
#             torch.manual_seed(SEED)
#
#         train, test_set = loaders.pendulum_features.PendulumDataset(
#             dataset_type='train'), loaders.pendulum_features.PendulumDataset(dataset_type='test')
#
#         generator = torch.Generator(device=DEVICE)
#         generator.manual_seed(SEED)
#
#         train_loader_1 = iter(DataLoader(train, batch_size=1, shuffle=False, num_workers=0, worker_init_fn=seed_worker, sampler=torch.utils.datasets.RandomSampler(train,generator=generator)))
#
#         random.seed(SEED)
#         np.random.seed(SEED)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(SEED)
#         else:
#             torch.manual_seed(SEED)
#
#         generator = torch.Generator(device=DEVICE)
#         generator.manual_seed(SEED)
#
#         train_loader_2 = iter(DataLoader(train, batch_size=1, shuffle=False, num_workers=0, worker_init_fn=seed_worker, sampler=torch.utils.datasets.RandomSampler(train,generator=generator)))
#     """
#
#     weights = np.array([sampling_factor[class_number] for class_number in sorted(list(class_counts.keys()))])
#     samples_weight = torch.from_numpy(weights)
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)
#     if DEVICE.type == "cuda":
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2 ** 32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)
#
#     random.seed(SEED)
#     np.random.seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(SEED)
#     else:
#         torch.manual_seed(SEED)
#
#     generator = torch.Generator(device=DEVICE)
#     generator.manual_seed(SEED)
#
#     sampler = WeightedRandomSampler(samples_weight, len(samples_weight), generator=generator)
#     loader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
#
#     return loader
