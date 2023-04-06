from typing import Set, Dict, Type

import numpy as np
from pyod.models import ecod, copod, cblof, cof, iforest, pca, loda, hbos, sod, ocsvm, lof, knn
from pytypes import override
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod


class ADBenchMultimodalValidatorMethodFactory:
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
        return [ADBenchMultimodalValidatorMethodFactory.get_validator_method(model_name)
                for model_name in ADBenchMultimodalValidatorMethodFactory.models.keys()]

    @staticmethod
    def get_validator_method(model_name: str):
        model = ADBenchMultimodalValidatorMethodFactory.models[model_name]

        class ADBenchAnomalyValidatorMethod(DataValidatorMethod):
            """
            A method for determining multidimensional anomalies. Operates on continuous datasets.
            """
            @override
            def name(self) -> str:
                method_name = model.__module__.split(".")[-1]
                return f"{method_name}_multimodal_anomaly_validator_method"

            @staticmethod
            def datatype() -> Set[DataType]:
                """
                Returns the datatype the validators method operates on
                @return: the datatype the validators method operates on
                """
                return {DataType.MULTIDIMENSIONAL, DataType.CONTINUOUS, DataType.CATEGORICAL}

            @staticmethod
            def param_keys() -> Set[ValidatorMethodParameter]:
                """
                Useful for documentation purposes. Lists the parameters in the datasets object that the validators
                operates on.
                @return: a list of parameters for the .validate() method.
                """
                return {ValidatorMethodParameter.ENTIRE_SET}

            @staticmethod
            def get_method_kwargs(data_object: Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
                """
                Gets the arguments for each run of the validator_method, and what to store the results under.

                @param data_object: the datasets object containing the datasets (train, test, entire, etc.)
                @param validator_kwargs: the kwargs from the validation schema.
                @return: a dict mapping from the key the result from calling .validate() on the kwargs values.
                """
                entire_dataset: Dataset = data_object["entire_set"]

                matrix = []

                for idx in range(entire_dataset.__len__()):
                    sample = entire_dataset[idx]
                    matrix.append(
                        np.concatenate([k.flatten() for k in sample.values()])
                    )

                matrix = np.array(matrix)

                kwargs_dict = {
                    f"results": {
                        "data_matrix": matrix,
                    }
                }

                return kwargs_dict

            @staticmethod
            def validate(
                data_matrix: Type[np.array] = None,  # n x d
            ) -> object:
                """
                Runs anomaly detection.

                @param data_matrix: an n x d matrix with the datasets needed for the model.
                @return: a list of anomaly scores
                """
                model_instance = model()
                model_instance.fit(data_matrix)
                anomaly_scores = model_instance.decision_function(data_matrix)

                return anomaly_scores

        return ADBenchAnomalyValidatorMethod
