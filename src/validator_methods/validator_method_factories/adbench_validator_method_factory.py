from typing import Set, Dict, Type

import numpy as np

from pyod.models import ecod, copod, cblof, cof, iforest, pca, loda, hbos, sod, ocsvm, lof, knn
from torch.utils.data import Dataset

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod

class ADBenchValidatorMethodFactory:
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
        return [ADBenchValidatorMethodFactory.get_validator_method(model_name) for model_name in ADBenchValidatorMethodFactory.models.keys()]

    @staticmethod
    def get_validator_method(model_name: str):
        model = ADBenchValidatorMethodFactory.models[model_name]
        class ADBenchAnomalyValidatorMethod(DataValidatorMethod):
            """
            A method for determining multidimensional anomalies. Operates on continuous datasets.
            """
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
                Useful for documentation purposes. Lists the parameters in the datasets object that the validators operates on.
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
                should_return_model_instance = validator_kwargs.get("should_return_model_instance", False)
                entire_dataset: Dataset = data_object["entire_set"]
                matrix_dict = entire_dataset.get_matrix(column_wise=True)
                kwargs_dict = {
                    f"{column}_results": {
                        "data_matrix": column_data,
                        "id_list": [entire_dataset.get_sample_id(k) for k in range(len(entire_dataset))],
                        "should_return_model_instance": should_return_model_instance,
                    } for column, column_data in matrix_dict.items()
                }

                return kwargs_dict

            @staticmethod
            def validate(
                data_matrix: Type[np.array] = None,  # n x d
                id_list: list[str] = None,
                should_return_model_instance: bool = False
            ) -> object:
                """
                Runs anomaly detection.

                @param data_matrix: an n x d matrix with the datasets needed for the model.
                @return:
                """
                model_instance = model()
                model_instance.fit(data_matrix)
                anomaly_scores = model_instance.decision_function(data_matrix)

                if should_return_model_instance: 
                    return {
                        id: anomaly_score 
                        for (id, anomaly_score) 
                        in zip(id_list, anomaly_scores)
                    }, model_instance
                
                return {
                    id: anomaly_score 
                    for (id, anomaly_score) 
                    in zip(id_list, anomaly_scores)
                }
                #todo: may need to return model instance for further processing

        return ADBenchAnomalyValidatorMethod