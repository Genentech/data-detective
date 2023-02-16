import os
import re
import typing

import numpy as np
from torch.utils.data import Dataset

from src.enums.enums import DataType


class ADBenchDataset(Dataset):
    """
    Base class for using the ADBench datasets.
    """
    ROOT_PATH = "/Users/mcconnl3/Code/ADBench/datasets/Classical/"

    def __init__(self, npz_filename="", input_data_type=DataType.MULTIDIMENSIONAL, output_data_type=DataType.CONTINUOUS):
        data_path = os.path.join(ADBenchDataset.ROOT_PATH, npz_filename)
        data = np.load(data_path)
        self.X, self.y = data['X'], data['y']
        self.input_data_type, self.output_data_type = input_data_type, output_data_type
        self.input_data_name = re.split(r"\.|_", npz_filename)[1]


    def __getitem__(self, item):
        return {
            self.input_data_name: self.X[item, :],
            "label": self.y[item]
        }

    def __len__(self):
        return self.X.shape[0]

    def datatypes(self) -> typing.Dict[str, DataType]:
        return {
            self.input_data_name: self.input_data_type,
            "label": self.output_data_type,
        }
