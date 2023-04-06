import numpy as np
import pandas as pd
import typing

from PIL import Image
from typing import Dict

from src.enums.enums import DataType


class CSVDataset:
    def __init__(self, filepath: str, datatype_dict: Dict[str, DataType], unwrap_paths: bool = True):
        self.datatypes_dict = datatype_dict
        self.df = pd.read_csv(filepath)
        self.df = self.df[list(datatype_dict.keys())]
        self.df = self.df.dropna()
        self.unwrap_paths = unwrap_paths

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
        sample = self.df.iloc[item]

        for column_name, datatype in self.datatypes_dict.items():
            # if we are working with an image that is being represented as a path
            if datatype == DataType.IMAGE and isinstance(sample[column_name], str) and self.unwrap_paths:
                img_path = f"data/{sample[column_name]}"
                img = np.array(Image.open(img_path))
                sample[column_name] = img

        return sample

    def datatypes(self) -> Dict[str, DataType]:
        """
        Gives the datatypes of a the dataset sample.
        @return: the datatypes of a the dataset sample.
        """
        return self.datatypes_dict