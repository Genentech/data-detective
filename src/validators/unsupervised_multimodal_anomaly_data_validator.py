from typing import Set, Type

from src.validator_methods.data_validator_method import DataValidatorMethod
# from src.validator_methods.diffi_anomaly_explanation_validator_method import DiffiAnomalyExplanationValidatorMethod
from src.validator_methods.validator_method_factories.adbench_multimodal_validator_method_factory import \
    ADBenchMultimodalValidatorMethodFactory

from src.validators.data_validator import DataValidator


class UnsupervisedMultimodalAnomalyDataValidator(DataValidator):
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
    """
    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            ADBenchMultimodalValidatorMethodFactory.get_validator_method("cblof"),
            ADBenchMultimodalValidatorMethodFactory.get_validator_method("pca"),
            ADBenchMultimodalValidatorMethodFactory.get_validator_method("iforest"),
            # ShapTreeValidatorMethod,
            # DiffiAnomalyExplanationValidatorMethod
        }