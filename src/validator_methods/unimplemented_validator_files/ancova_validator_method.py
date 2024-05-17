from typing import Set

from src.enums.enums import DataType, ValidatorMethodParameter
from src.validator_methods.data_validator_method import DataValidatorMethod



class ANCOVAValidatorMethod(DataValidatorMethod):
    """
    A quick note:
        In an effort to keep everything stateless and relatively functional, the DataValidators and the
        DataValidatorMethods will be completely functional; that is, they will not contain any variables relating to
        the actual datasets. Does this make sense in the following cases:
            - splitting params:
                2-3 datasets: train(/val)/test
            - bimodal:
                just need one dataset, can fit a mixture model
            -
    """
    @staticmethod
    def datatype() -> Set[DataType]:
        """
        @return: the datatype the validators method operates on
        """
        return {DataType.CONTINUOUS}

    @staticmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        """
        Useful for documentation purposes. Lists the parameters that the validators operates on.

        @return: a list of parameters for the .validate() method.
        """
        return {
            ValidatorMethodParameter.TRAINING_SET,
            ValidatorMethodParameter.TEST_SET,
            ValidatorMethodParameter.VALIDATION_SET
        }

    def validate(self, **kwargs) -> bool:
        """
        Determines

        @param kwargs:
        @return:
        """
        return True

