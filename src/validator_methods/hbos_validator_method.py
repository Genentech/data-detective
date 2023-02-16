from collections import defaultdict
from typing import List, Set, Dict, Type, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset
from src.validator_methods.data_validator_method import DataValidatorMethod

from scipy import stats

DEFAULT_ALPHA = 0.1
DEFAULT_CONTAMINATION = 0.1
DEFAULT_N_BINS = 10
DEFAULT_TOL = 0.5

"""
TODO: this was originally framed as an image anomaly detection task. Need to spend some effort
to turn it back into a method for tabular outlier detection. 
"""

class HBOSValidatorMethod(DataValidatorMethod):
    """
    A method for determining outliers of an image dataset by histogram analysis.
    src: https://www.dfki.de/fileadmin/user_upload/import/6431_HBOS-poster.pdf
    src: https://medium.com/dataman-in-ai/anomaly-detection-with-histogram-based-outlier-detection-hbo-bc10ef52f23f

    Basically, creates a histogram of intensity values for the imag
    """
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return { DataType.CONTINUOUS, DataType.CATEGORICAL }

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {ValidatorMethodParameter.ENTIRE_SET}

    @staticmethod
    def get_method_kwargs(data_object: dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
        """
        Gets the arguments for each run of the validator_method, and what to store the results under.

        @param data_object: the datasets object containing the datasets (train, test, entire, etc.)
        @param validator_kwargs: the kwargs from the validation schema.
        @return: a dict mapping from the key the result from calling .validate() on the kwargs values.
        """
        entire_dataset: Dataset = data_object["entire_set"]

        for normal_label in get_class_counts(entire_dataset).keys():
            kwargs_dict[f'normal_class_is_{label}'] = {
                {
                    "alpha" : validator_kwargs.get("alpha", DEFAULT_ALPHA),
                    "contamination": validator_kwargs.get("contamination", DEFAULT_CONTAMINATION),
                    "dataset": entire_dataset,
                    "n_bins" : validator_kwargs.get("n_bins", DEFAULT_N_BINS),
                    "normal_class": normal_label,
                    "tol" : validator_kwargs.get("tol", DEFAULT_TOL),
                }
            }

        return kwargs_dict

    @staticmethod
    def validate(
            alpha: float = DEFAULT_ALPHA,
            contamination: float = DEFAULT_CONTAMINATION,
            dataset: Type[Dataset] = None,
            n_bins: Union[int, str] = DEFAULT_N_BINS,
            normal_class: int = 0,
            tol: float = DEFAULT_TOL,
    ) -> object:
        """

        Runs a
        Input dict:
        {
            "featr": {
                0: 1.1412321,
                ...
                9999: -0.4123643
            }
        }

        @return: the stats object that it needs when it gets back.
        """
        return stats.kstest(series, 'norm')

def get_class_counts(train_dataset: type[Dataset]):
    class_counts: Type[defaultdict] = defaultdict(lambda: 0)
    classes: List[int] = []

    for idx in range(len(train_dataset)):
        sample: dict[str, Union[int, torch.FloatTensor]] = train_dataset[idx]
        sample_class: int = sample['label']
        class_counts[sample_class] += 1

    return class_counts

def load_dataset_as_anomaly(
        train_dataset: Type[Dataset],
        normal_class_number: int,
        normal_class_proportion: float = 0.5
    ):
    """
    Produces a dataloader that equally samples between a single class and all other anomaly classes.

    @param dataset: the unfiltered, raw __training__ dataset.
        - expects a "label" attribute that is an integer.
    @param anomaly_class_number: the class number for the normal class (all others are anomaly)
    @return: a dataloader that has equal probability of sampling from normal vs non normal class.
    """
    class_counts: Type[defaultdict] = defaultdict(lambda: 0)
    classes: List[int] = []

    for idx in range(len(train_dataset)):
        sample: dict[str, Union[int, torch.FloatTensor]] = train_dataset[idx]
        sample_class: int = sample['label']
        class_counts[sample_class] += 1
        classes.append(sample_class)

    normalized_class_proportion = {k: float(v) / len(train_dataset) for k, v in class_counts.items()}
    ideal_class_proportion = {
        class_number: (1 - normal_class_proportion) / (len(train_dataset) - 1)
            for class_number in class_counts.keys()
            if class_number != normal_class_number
    }
    ideal_class_proportion[normal_class_number] = normal_class_proportion
    # what do we want: 50% for a single class, and the rest in the rest...
    # since the weighted random sampler doesn't need to be normalized, we can just enter the raw % for each entry.

    # how much to over/under - sample by.
    sampling_factor = {
        class_number: ideal_class_proportion[class_number] / normalized_class_proportion[class_number]
            for class_number in class_counts.keys()
    }

    weights = np.array([sampling_factor[class_number] for class_number in classes])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)

    return loader
