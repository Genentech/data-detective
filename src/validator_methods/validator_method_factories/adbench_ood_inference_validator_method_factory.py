import typing
from typing import Set, Dict, Type

import numpy as np
import pyod
from pyod.models import ecod, copod, cblof, cof, iforest, pca, loda, hbos, sod, ocsvm, lof, knn
from pytypes import override
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class ADBenchOODInferenceValidatorMethodFactory:
    models = {
        "cblof": cblof.CBLOF,
        "cof": cof.COF,
        "copod": copod.COPOD,
        "ecod": ecod.ECOD,
        "hbos": hbos.HBOS,
        "iforest": iforest.IForest,
        "knn": knn.KNN,
        "loda": loda.LODA,
        "lof": lof.LOF,
        "ocsvm": ocsvm.OCSVM,
        "pca": pca.PCA,
        "sod": sod.SOD,
    }

    @staticmethod
    def get_all_validator_methods():
        return [ADBenchOODInferenceValidatorMethodFactory.get_validator_method(model_name)
                for model_name in ADBenchOODInferenceValidatorMethodFactory.models.keys()]

    @staticmethod
    def get_validator_method(model_name: str):
        model = ADBenchOODInferenceValidatorMethodFactory.models[model_name]

        class ADBenchOODInferenceValidatorMethod(DataValidatorMethod):
            """
            A method for determining multidimensional anomalies. Operates on continuous datasets.
            """
            @override
            def name(self) -> str:
                method_name = model.__module__.split(".")[-1]
                return f"{method_name}_anomaly_validator_method"

            @staticmethod
            def datatype() -> Set[DataType]:
                """
                Returns the datatype the validators method operates on
                @return: the datatype the validators method operates on
                """
                return {DataType.MULTIDIMENSIONAL}

            @staticmethod
            def param_keys() -> Set[ValidatorMethodParameter]:
                """
                Useful for documentation purposes. Lists the parameters in the datasets object that the validators
                operates on.
                @return: a list of parameters for the .validate() method.
                """
                return {ValidatorMethodParameter.INFERENCE_SET, ValidatorMethodParameter.EVERYTHING_BUT_INFERENCE_SET}

            @staticmethod
            def get_method_kwargs(data_object: typing.Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
                """
                Gets the arguments for each run of the validator_method, and what to store the results under.

                @param data_object: the datasets object containing the datasets (train, test, entire, etc.)
                @param validator_kwargs: the kwargs from the validation schema.
                @return: a dict mapping from the key the result from calling .validate() on the kwargs values.
                """
                everything_but_inference_dataset: Dataset = data_object["everything_but_inference_set"]
                inference_dataset: Dataset = data_object["inference_set"]

                def get_matrix_rep(dataset):
                    matrix = []

                    for idx in range(dataset.__len__()):
                        sample = dataset[idx]
                        matrix.append(
                            np.concatenate([k.flatten() for k in sample.values()])
                        )

                    return np.array(matrix)

                matrix_representation = get_matrix_rep(everything_but_inference_dataset)
                matrix_representation_inference = get_matrix_rep(inference_dataset)

                kwargs_dict = {
                    "results": {
                        "data_matrix": matrix_representation,
                        "inference_data_matrix": matrix_representation_inference,
                    }
                }

                return kwargs_dict

            @staticmethod
            def validate(
                    data_matrix: Type[np.array] = None,  # n x d
                    inference_data_matrix: Type[np.array] = None,
            ) -> object:
                """
                Runs anomaly detection.
                @return:
                """
                model_instance = model()
                model_instance.fit(data_matrix)
                ood_scores = model_instance.decision_function(inference_data_matrix)
                sorted_original_ood_scores = sorted(model_instance.decision_function(data_matrix))

                # src: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
                def find_nearest(array, value):
                    array = np.asarray(array)
                    idx = (np.abs(array - value)).argmin()
                    return idx

                nearest_idx = np.array([find_nearest(sorted_original_ood_scores, ood_score) for ood_score in ood_scores])
                percentiles = nearest_idx / len(sorted_original_ood_scores)

                return {
                    "ood_scores": ood_scores,
                    "percentiles": percentiles,
                }

        return ADBenchOODInferenceValidatorMethod


