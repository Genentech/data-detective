from typing import List, Dict, Set, Type

from torch.utils.data import Dataset

from src.enums.enums import DataType
from src.validator_methods.data_validator_method import DataValidatorMethod
from src.validators.data_validator import DataValidator

# class TemplateDataValidator(DataValidator):
#     """
#     A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
#     datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
#     """
#     @staticmethod
#     def validator_methods() -> Set[Type[DataValidatorMethod]]:
#         return {
#             # Method1
#             # Method2
#             # Method3
#         }