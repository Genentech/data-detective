from abc import abstractmethod
from typing import Any, Dict, List, Union
import joblib
import pandas as pd
import torch

from src.enums.enums import DataType


class DataDetectiveDataset(torch.utils.data.Dataset):
    """
    What should be true of every Data Detective Dataset?
    - it should override __getitem__()
    - it should override __len__()
    - it should override .datatypes()
    - it should satisfy the following identification criteria: 
        - optional map to subject ids  
        - optional map to sample ids
        - if no sample id is provided, one will be provided based on
          the data dictionary

    example: 
    dd_test = dataset(indices=indices) # these indices are then used for the data. 
    """
    def __init__(self, show_id = False ):
        self.show_id = show_id
        # for now, just do this, but not always. super slow
        index_objects = []
        for data_idx in range(self.__len__()):
            index_objects.append({
                "data_idx": data_idx, 
                "sample_id": self.get_sample_id(data_idx),
                "subject_id": self.get_subject_id(data_idx),
            })

        self.index_df = pd.DataFrame(index_objects)

    # can be overridden
    # let's try to avoid using idx and focus on data_idx, sample_idx, and subject_idx
    # should this map from internal id or 
    def get_subject_id(self, data_idx_or_sample_id: int) -> str: 
        if isinstance(data_idx_or_sample_id, int): 
            data_idx = data_idx_or_sample_id
            sample_id = self.get_sample_id(data_idx)
        else: 
            sample_id = data_idx_or_sample_id

        return sample_id

    # can be overridden
    # maps from normal index 
    def get_sample_id(self, data_idx: int) -> str: 
        self.original_sample_id = True
        return joblib.hash(self.get_data(data_idx))

    def get_id(self, sample_idx) -> Dict[int, Union[str, int]]: 
        id_dict = {
            "subject_id": self.get_subject_id(sample_idx),
            "sample_id": sample_idx,
        }

        return id_dict

    @abstractmethod
    def get_data(self, data_idx_or_sample_id: Union[int, str]) -> Any: 
        pass

    # idx here can be sample idx or normal idx, hard to say. 
    def __getitem__(self, data_idx_or_sample_id) -> Dict[str, Any]: 
        if isinstance(data_idx_or_sample_id, int): 
            data_idx = data_idx_or_sample_id
            sample_id = self.get_sample_id(data_idx)
        else:
            sample_id = data_idx_or_sample_id

        # lets enforce this to be
        return {
            "id": self.get_id(sample_id),
            "data": self.get_data(data_idx_or_sample_id)
        }

    # not necessary to override
    def __len__(self) -> int: 
        pass

    @abstractmethod
    def datatypes(self) -> Dict[str, DataType]: 
        pass