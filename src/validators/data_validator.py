from abc import abstractmethod
from typing import Type, Dict, Set

from torch.utils.data import Dataset

import src.validator_methods.data_validator_method as data_validator_method
import src.datasets.synthetic_data_generators as synthetic_data_generators
from src.datasets.column_filtered_dataset import ColumnFilteredDataset
from src.enums.enums import DataType


class DataValidator:
    """
    A dataset has many features/columns, and each column has many ValidatorMethods that apply to it, depending on the
    datatype. A validators's job includes:
        - running validators methods on the correct datatypes in a dataset (for example, running an outlier detector
        on both categorical and continuous datasets)
        - setting up the parameters correctly for each validator_method
        - applying the validator_methods individually on all of the specified columns.

    """
    @staticmethod
    @abstractmethod
    def validator_methods() -> Set[Type[data_validator_method.DataValidatorMethod]]:
        # raise NotImplementedError()
        pass

    @staticmethod
    @abstractmethod
    def is_default() -> bool:
        # raise NotImplementedError()
        pass

    @classmethod
    def supported_datatypes(cls) -> Set[DataType]:
        return {method.datatype for method in cls.validator_methods()}

    @staticmethod
    def _method_applies(data_object, validator_method):
        """
        Returns True iff there is some overlap between the validator method's datatypes
        and the datasets object's datatypes.

        @param method_specific_data_object:
        @return:
        """

        for key, filtered_dataset in data_object.items():
            filtered_dataset_datatype_set = set(filtered_dataset.datatypes().values())
            validator_method_datatype_set = validator_method.datatype()
            if len(filtered_dataset_datatype_set.intersection(validator_method_datatype_set)) == 0:
                return False

        return True


    @classmethod
    def validate(cls, data_object: Dict[str, Dataset], validator_kwargs: Dict) -> Dict:
        """
        Run all of the validators methods on the dataset. What are the things we might need in a .validate method?
            - training dataset
            - validation dataset
            - test dataset
            - entire dataset (in cases of cross-validation)
            - point(s) of interest for test-time OOD
        """

        """
        Ok, so what can we expect out of default behavior?
            - the method operates on a single column/feature at a time. 
            - the method operates on just the datasets objects specified
        If this is no longer true, then we just have to subclass this and reapply some of the logic.
        """
        results = {}

        for validator_method in cls.validator_methods():
            print(f"   running {validator_method}...")
            # this filters out methods that don't accept the datatypes needed.
            if not DataValidator._method_applies(data_object, validator_method):
                continue

            # look at the datatypes on the validators method, and apply
            method_results = {}

            # for every datasets object:
                # add an .item from data_object key to filtered dataset (by datatype matching method/dataset)
                # if the datasets object is included in the method's param_keys
            method_specific_data_object = {
                key: ColumnFilteredDataset(dataset, matching_datatypes=[e.value for e in list(validator_method.datatype())])
                    for key, dataset in data_object.items()
                    if key in {e.value for e in validator_method.param_keys()}
            }

            for result_key, method_kwargs in validator_method.get_method_kwargs(data_object = method_specific_data_object, validator_kwargs=validator_kwargs).items():
                method_results[result_key] = validator_method.validate(**method_kwargs)

            results[validator_method().name()] = method_results

        return results