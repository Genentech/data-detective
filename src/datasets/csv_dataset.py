import typing
from typing import Dict

import pandas as pd

from src.enums.enums import DataType


class CSVDataset:
    def __init__(self, filepath: str, datatype_dict: Dict[str, DataType]):
        self.datatypes_dict = datatype_dict
        self.df = pd.read_csv(filepath)
        self.df = self.df[list(datatype_dict.keys())]
        self.df = self.df.dropna()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        @return: the length of the dataset
        """
        return self.df.__len__()

    def __getitem__(self, item) -> typing.Dict:
        """
        Returns an item from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the data and the label.
        """
        return self.df.iloc[item]

    def datatypes(self) -> Dict[str, DataType]:
        """
        Gives the datatypes of a the dataset sample.
        @return: the datatypes of a the dataset sample.
        """
        return self.datatypes_dict