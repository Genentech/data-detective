import typing
from typing import Dict

from torch.utils.data import Dataset


import src.utils
from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType


class SyntheticCIDataset(DataDetectiveDataset):
    def __init__(self, dataset_type: str, dataset_size: int = 10000) -> None:
        """
        Initializes the dataset and generaates the data.
        Generates a dataset with three variables: X, Y, and Z, all continuous

        @param dataset_type: the type of dataset ('CI', 'I', or 'NI')
        @param dataset_size: the size of the dataset to generate
        """
        self.dataset_size = dataset_size
        self.dataset_type = dataset_type
        self.columns = ['x', 'y', 'z']

        if self.dataset_type == "CI":
            self.x, self.y, self.z = src.utils.generate_ci_samples(dataset_size)
        elif self.dataset_type == "NI":
            self.x, self.y, self.z = src.utils.generate_ni_samples(dataset_size)

        super().__init__(self)

    def __getitem__(self, index: int) -> Dict:
        """
        Returns a sample from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the data and the label.
        """
        return {
            'x': self.x[index],
            'y': self.y[index],
            'z': self.z[index]
        }

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        @return: the length of the dataset
        """
        return self.dataset_size

    @property
    def num_columns(self) -> int:
        """
        The number of columns in the dataset.
        @return: the number of columns in the dataset.
        """
        return 3

    def datatypes(self) -> typing.Dict[str, DataType]:
        """
        Gives the datatypes of a the dataset sample.
        @return: the datatypes of a the dataset sample.
        """
        return {
            column_name: DataType.MULTIDIMENSIONAL
            for column_name in self.columns
        }