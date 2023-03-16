from typing import Dict

import pandas as pd

from src.enums.enums import DataType


class CSVDataset:
    def __init__(self, filepath: str, datatype_dict: Dict[str, DataType]):
        self.datatypes_dict = datatype_dict
        self.df = pd.read_csv(filepath)
        self.df = self.df[list(datatype_dict.keys())]
        self.df = self.df.dropna()

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, item: int):
        return self.df.iloc[item]

    def datatypes(self) -> Dict[str, DataType]:
        return self.datatypes_dict