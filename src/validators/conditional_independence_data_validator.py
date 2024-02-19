from typing import Set, Type

from src.validator_methods.data_validator_method import DataValidatorMethod
from src.validator_methods.fcit_validator_method import FCITValidatorMethod
from src.validators.data_validator import DataValidator


class ConditionalIndependenceDataValidator(DataValidator):
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
    """
    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            FCITValidatorMethod
        }