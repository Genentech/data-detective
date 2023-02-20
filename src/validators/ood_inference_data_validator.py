from typing import Set, Type

from src.validator_methods.cblof_ood_inference_validator_method import CBLOFOODInferenceValidatorMethod
from src.validator_methods.data_validator_method import DataValidatorMethod
from src.validator_methods.isolation_forest_ood_inference_validator_method import \
    IsolationForestOODInferenceValidatorMethod
from src.validator_methods.pca_ood_inference_validator_method import PCAOODInferenceValidatorMethod
from src.validators.data_validator import DataValidator


class OodInferenceDataValidator(DataValidator):
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
    """
    @staticmethod
    def is_default():
        return False

    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            # HistogramImageAnomalyValidatorMethod,
            # CBLOFOODInferenceValidatorMethod,
            IsolationForestOODInferenceValidatorMethod,
            # PCAOODInferenceValidatorMethod,
        }