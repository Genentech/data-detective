from typing import List, Set, Type

from src.validator_methods.data_validator_method import DataValidatorMethod
from src.validator_methods.kolmogorov_smirnov_multidimensional_split_validator_method import \
    KolmogorovSmirnovMultidimensionalSplitValidatorMethod
from src.validator_methods.kolmogorov_smirnov_split_validator_method import KolmogorovSmirnovSplitValidatorMethod
from src.validator_methods.kruskal_wallis_multidimensional_split_validator_method import \
    KruskalWallisMultidimensionalSplitValidatorMethod
from src.validator_methods.kruskal_wallis_split_validator_method import KruskalWallisSplitValidatorMethod
from src.validator_methods.mann_whitney_multidimensional_split_validator_method import \
    MannWhitneyMultidimensionalSplitValidatorMethod
from src.validator_methods.mann_whitney_split_validator_method import MannWhitneySplitValidatorMethod
from src.validators.data_validator import DataValidator
from src.validator_methods.chi_square_split_validator_method import ChiSquareSplitValidatorMethod

class SplitCovariateDataValidator(DataValidator):
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A DataValidator is a collection of ValidatorMethods for a unique purpose.
    """

    @staticmethod
    def is_default(self):
        return False

    @staticmethod
    def validator_methods() -> Set[Type[DataValidatorMethod]]:
        return {
            KolmogorovSmirnovSplitValidatorMethod,
            KolmogorovSmirnovMultidimensionalSplitValidatorMethod,
            KruskalWallisSplitValidatorMethod,
            KruskalWallisMultidimensionalSplitValidatorMethod,
            MannWhitneySplitValidatorMethod,
            MannWhitneyMultidimensionalSplitValidatorMethod,
            ChiSquareSplitValidatorMethod
        }