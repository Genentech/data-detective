from typing import List, Dict, Set, Type

from torch.utils.data import Dataset

from src.enums.enums import DataType
from src.validator_methods.data_validator_method import DataValidatorMethod
from src.validators.data_validator import DataValidator
from src.validator_methods.duplicate_high_dimensional_validator_method import DuplicateHighDimensionalValidatorMethod
from src.validator_methods.duplicate_sample_validator_method import DuplicateSampleValidatorMethod
from src.validator_methods.near_duplicate_multidimensional_validator_method import NearDuplicateMultidimensionalValidatorMethod

class DuplicateDataValidator(DataValidator):
    """
    A data validator to detect exact/approximate duplicates in higher dimensional samples, 
    or to detect  
    """
    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            DuplicateHighDimensionalValidatorMethod,
            DuplicateSampleValidatorMethod,
            NearDuplicateMultidimensionalValidatorMethod
        }