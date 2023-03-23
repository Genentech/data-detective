import os
from abc import abstractmethod
from typing import Type, Dict, Set

from joblib import Parallel, delayed
from torch.utils.data import Dataset

import src.validator_methods.data_validator_method as data_validator_method
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
    def name(self) -> str:
        return self.__module__.split(".")[-1]

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

        @param data_object: the data object to use to determine overlap with the validator method.
        @param validator_method: the method to use to determinne overlap with the data object

        @return:
        """
        for key, filtered_dataset in data_object.items():
            filtered_dataset_datatype_set = set(filtered_dataset.datatypes().values())
            validator_method_datatype_set = validator_method.datatype()
            if len(filtered_dataset_datatype_set.intersection(validator_method_datatype_set)) == 0:
                return False

        return True

    @classmethod
    def get_results(cls, data_object, validator_method, validator_kwargs):
        def thread_print(s):
            print(f"thread {os.getpid()}: {s}")

        print(f"thread {os.getpid()} entered to handle validator method {validator_method().name()}")
        thread_print(f"   running {validator_method().name()}...")
        # this filters out methods that don't accept the datatypes needed.
        if not DataValidator._method_applies(data_object, validator_method):
            return

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

        for result_key, method_kwargs in validator_method.get_method_kwargs(data_object=method_specific_data_object,
                                                                            validator_kwargs=validator_kwargs).items():
            method_results[result_key] = validator_method.validate(**method_kwargs)


        print(f"{os.getpid()} finished")
        return (validator_method().name(), method_results)

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
        # results = {}

        # for validator_method in cls.validator_methods():
        #     print(f"   running {validator_method().name()}...")
        #     # this filters out methods that don't accept the datatypes needed.
        #     if not DataValidator._method_applies(data_object, validator_method):
        #         continue
        #
        #     # look at the datatypes on the validators method, and apply
        #     method_results = {}
        #
        #     # for every datasets object:
        #         # add an .item from data_object key to filtered dataset (by datatype matching method/dataset)
        #         # if the datasets object is included in the method's param_keys
        #     method_specific_data_object = {
        #         key: ColumnFilteredDataset(dataset, matching_datatypes=[e.value for e in list(validator_method.datatype())])
        #             for key, dataset in data_object.items()
        #             if key in {e.value for e in validator_method.param_keys()}
        #     }
        #
        #     for result_key, method_kwargs in validator_method.get_method_kwargs(data_object = method_specific_data_object, validator_kwargs=validator_kwargs).items():
        #         method_results[result_key] = validator_method.validate(**method_kwargs)
        #
        #     results[validator_method().name()] = method_results
        result_items = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(cls.get_results)(data_object, validator_method, validator_kwargs)
                        for validator_method in cls.validator_methods())

        results = {k: v for k, v in [x for x in result_items if x is not None]}

        return results