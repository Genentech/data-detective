from typing import Set, Type

from src.validator_methods.data_validator_method import DataValidatorMethod

from src.validators.data_validator import DataValidator
from src.validator_methods.validator_method_factories.adbench_validator_method_factory import ADBenchValidatorMethodFactory


class UnsupervisedAnomalyDataValidator(DataValidator):
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
    """
    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            ADBenchValidatorMethodFactory.get_validator_method("cblof"),
            ADBenchValidatorMethodFactory.get_validator_method("pca"),
            ADBenchValidatorMethodFactory.get_validator_method("iforest"),
        }