from typing import Set, Type

from src.validator_methods.cblof_anomaly_validator_method import CBLOFAnomalyValidatorMethod
from src.validator_methods.data_validator_method import DataValidatorMethod
# from src.validator_methods.fcit_validator_method import FCITValidatorMethod
# from src.validator_methods.hbos_validator_method import HBOSValidatorMethod
from src.validator_methods.histogram_image_anomaly_validator_method import HistogramImageAnomalyValidatorMethod
from src.validator_methods.isolation_forest_anomaly_validator_method import IsolationForestAnomalyValidatorMethod
from src.validator_methods.pca_anomaly_validator_method import PCAAnomalyValidatorMethod
from src.validator_methods.validator_method_factories.adbench_multimodal_validator_method_factory import \
    ADBenchMultimodalValidatorMethodFactory
from src.validator_methods.validator_method_factories.adbench_validator_method_factory import \
    ADBenchValidatorMethodFactory

from src.validators.data_validator import DataValidator


class UnsupervisedMultimodalAnomalyDataValidator(DataValidator):
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
    """
    @staticmethod
    def is_default():
        return True

    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            # HistogramImageAnomalyValidatorMethod,
            # CBLOFAnomalyValidatorMethod,
            # IsolationForestAnomalyValidatorMethod,
            # PCAAnomalyValidatorMethod,
            ADBenchMultimodalValidatorMethodFactory.get_validator_method("cblof"),
            ADBenchMultimodalValidatorMethodFactory.get_validator_method("pca"),
            ADBenchMultimodalValidatorMethodFactory.get_validator_method("iforest"),
        }