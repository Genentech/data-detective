from collections import defaultdict
from typing import List, Set, Dict, Type, Union, Literal

import numpy as np
import pandas as pd
import pyod.models.pca
import pyod.models.cblof
import sklearn
import torch
import typing
from sklearn import ensemble
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.utils import filter_dataset
from src.validator_methods.data_validator_method import DataValidatorMethod

from scipy import stats

"""
TODO: this was originally framed as an image anomaly detection task. Need to spend some effort
to turn it back into a method for tabular outlier detection. 
"""


class CBLOFOODInferenceValidatorMethod(DataValidatorMethod):

    """
    A method for determining multidimensional anomalies. Operates on continuous data.
    Explained further (and implementation inspired from) https://towardsdatascience.com/anomaly-detection-in-python-part-2-multivariate-unsupervised-methods-and-code-b311a63f298b


    Basically, takes in a multidimensional (1D) feature vector
    """
    # DEFAULT_CONTAMINATION = 'auto'
    # DEFAULT_MAX_FEATURES = 400
    # DEFAULT_MAX_SAMPLES = 1000
    # DEFAULT_N_ESTIMATORS = 10

    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return { DataType.MULTIDIMENSIONAL, DataType.TIME_SERIES }


    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return { ValidatorMethodParameter.EVERYTHING_BUT_INFERENCE_SET, ValidatorMethodParameter.INFERENCE_SET }

    @staticmethod
    def get_method_kwargs(data_object: typing.Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
        """
        Gets the arguments for each run of the validator_method, and what to store the results under.

        @param data_object: the data object containing the datasets (train, test, entire, etc.)
        @param validator_kwargs: the kwargs from the validation schema.
        @return: a dict mapping from the key the result from calling .validate() on the kwargs values.
        """
        everything_but_inference_dataset: Dataset = data_object["everything_but_inference_set"]
        inference_dataset: Dataset = data_object["inference_set"]

        # todo: figure out matrix representation for both data formats.
        # matrix_representation = np.array([list(d.values()) for d in entire_dataset[:].values()]).T,
        matrix_representation = list(everything_but_inference_dataset[:].values())[0]
        matrix_representation_inference = list(inference_dataset[:].values())[0]

        kwargs_dict = {
            "results": {
                # "contamination": validator_kwargs.get("contamination", IsolationForestAnomalyValidatorMethod.DEFAULT_CONTAMINATION),
                "data_matrix": matrix_representation,
                "inference_data_matrix": matrix_representation_inference,
            }
        }

        return kwargs_dict

    @staticmethod
    def validate(
        # contamination: Union[float, Literal['auto']] = DEFAULT_CONTAMINATION,
        data_matrix: Type[np.array] = None, # n x d
        inference_data_matrix:  Type[np.array] = None,

        # max_features: Union[int, float] = DEFAULT_MAX_FEATURES,
        # max_samples: Union[float, int, Literal['auto']] = DEFAULT_MAX_SAMPLES,
        # n_estimators: int = DEFAULT_N_ESTIMATORS,
    ) -> object:
        """
        Runs an isolation forest to try and perform anomaly detection.

        Input dict:
        {
            "featr": {
                0: 1.1412321,
                ...
                9999: -0.4123643
            }
        }

        @param n_estimators:
        @param max_samples:
        @param contamination:
        @param max_features:
        @return:
        """
        model = pyod.models.cblof.CBLOF()
        model.fit(data_matrix)

        anomaly_scores = model.decision_function(inference_data_matrix)
        # predictions = model.predict(data_matrix)

        return anomaly_scores