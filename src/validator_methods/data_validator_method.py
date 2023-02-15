from abc import abstractmethod
from typing import List, Set, Dict

from src.enums.enums import DataType, ValidatorMethodParameter
from torch.utils.data import Dataset


class DataValidatorMethod:
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
    def name(self) -> str:
        return self.__module__.split(".")[-1]

    @staticmethod
    @abstractmethod
    def param_keys() -> Set[ValidatorMethodParameter]:
        pass

    @staticmethod
    @abstractmethod
    def datatype() -> Set[DataType]:
        pass

    @staticmethod
    @abstractmethod
    def get_method_kwargs(data_object: Dict[str, Dataset], validator_kwargs: Dict = None) -> Dict:
        pass

    @staticmethod
    @abstractmethod
    def validate(**kwargs) -> object:
        pass
