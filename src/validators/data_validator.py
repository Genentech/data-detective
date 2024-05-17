import os
import threading
from abc import abstractmethod
from typing import Type, Dict, Set, List, Tuple, Optional

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
    @classmethod
    def name(cls) -> str:
        return cls.__module__.split(".")[-1]

    @staticmethod
    @abstractmethod
    def validator_methods() -> Set[Type[data_validator_method.DataValidatorMethod]]:
        # raise NotImplementedError()
        pass

    @classmethod
    def supported_datatypes(cls) -> Set[DataType]:
        """
        A list of supported datatypes by the validator.
        @return: a set of supported datatypes by the validator.
        """
        return {method.datatype for method in cls.validator_methods()}

    @staticmethod
    def _method_applies(data_object, validator_method):
        """
        Returns True iff there is some overlap between the validator method's datatypes
        and the datasets object's datatypes.

        @param data_object: the data object to use to determine overlap with the validator method.
        @param validator_method: the method to use to determine overlap with the data object

        @return:
        """
        for key, filtered_dataset in data_object.items():
            if isinstance(filtered_dataset, dict): 
                result = DataValidator._method_applies(data_object=filtered_dataset, validator_method=validator_method)
                # only return if False
                if result == False: 
                    return result
            else: 
                filtered_dataset_datatype_set = set(filtered_dataset.datatypes().values())
                validator_method_datatype_set = validator_method.datatype()
                if len(filtered_dataset_datatype_set.intersection(validator_method_datatype_set)) == 0:
                    return False

        return True

    @classmethod
    def validate(cls, data_object, validator_method, validator_kwargs) -> Optional[Tuple]:
        """
        Runs a single validator method against the data object.

        @param data_object: the data object for the validator_method
        @param validator_method: the validator methood to run the data object against
        @param validator_kwargs: any other kwargs needed for the validator method

        @return a tuple consisting of the validator class, the validator method name, and the results from the validator
        method.
        """
        def thread_print(s):
            print(f"thread {threading.get_ident()}: {s}")

        def process_print(s):
            print(f"process {os.getpid()}: {s}")

        print(f"thread {threading.get_ident()} entered to handle validator method {validator_method().name()}")
        thread_print(f"   running {validator_method().name()}...")
        # this filters out methods that don't accept the datatypes needed.
        if not DataValidator._method_applies(data_object, validator_method):
            return

        # look at the datatypes on the validators method, and apply
        method_results = {}

        # for every datasets object:
        # add an .item from data_object key to filtered dataset (by datatype matching method/dataset)
        # if the datasets object is included in the method's param_keys
        def get_method_specific_data_object(data_object):
            method_specific_data_object = {}
            for key, dataset in data_object.items(): 
                if isinstance(dataset, dict): 
                    method_specific_data_object[key] = get_method_specific_data_object(dataset)
                else: 
                    method_specific_data_object[key] = ColumnFilteredDataset(dataset, matching_datatypes=[e.value for e in list(validator_method.datatype())])

            return method_specific_data_object

        method_specific_data_object = get_method_specific_data_object(data_object)
        # method_specific_data_object = {
        #     key: ColumnFilteredDataset(dataset, matching_datatypes=[e.value for e in list(validator_method.datatype())])
        #     for key, dataset in data_object.items()
        # }

        for result_key, method_kwargs in validator_method.get_method_kwargs(data_object=method_specific_data_object,
                                                                            validator_kwargs=validator_kwargs).items():
            method_results[result_key] = validator_method.validate(**method_kwargs)

        thread_print(f"finished")
        return (cls.name(), validator_method().name(), method_results)

    @classmethod
    def get_task_list(cls, data_object: Dict[str, Dataset], validator_kwargs: Dict) -> List[Tuple]:
        """
        Gets a list of the task class and task arguments as a list of tuples.

        @param data_object: the data object for the validator_method
        @param validator_kwargs: any other kwargs needed for the validator method
        @return: a list of task descriptors (method / arg tuples)
        """
        return [
            (
                cls.validate,
                (data_object, validator_method, validator_kwargs)
            ) for validator_method in cls.validator_methods()
        ]
