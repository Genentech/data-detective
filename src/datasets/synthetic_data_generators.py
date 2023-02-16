"""
Requirements from a Dataset: 
- any multidimensional or imaging datasets must be referenced by path
- __getitem__ include key/value pairs with each feature name. 
- must have a .datatypes property that maps from every feature column to its datatype
"""
from abc import abstractmethod
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from src.enums.enums import DataType


# class DataDetectiveDataset(Dataset):
#     @abstractmethod
#     def datatypes(self) -> Dict[str, DataType]:
#         pass
#
#     @abstractmethod
#     def transforms(self) -> Dict[str, List[torch.nn.Module]]:
#         return {}
#
#     @abstractmethod
#     def __len__(self) -> int:
#         pass
#
#     def getitem(self, index: int) -> Dict[str, Any]:
#         pass
#
#     def __getitem__(self, index: int) -> Dict[str, Any]:
#         item = self.getitem(index)
#         for col_name, transform_list in self.transforms().items():

