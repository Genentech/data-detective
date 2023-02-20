import typing
from typing import Dict

from torch.utils.data import Dataset

import src.utils
from src.enums.enums import DataType


class SyntheticCIDataset(Dataset):
    """
    Generates a dataset with three variables: X, Y, and Z, all continuous
    @param dataset_type: the type of dataset ('CI', 'I', or 'NI')
    """
    def __init__(self, dataset_type: str, dataset_size: int = 10000) -> None:
        self.dataset_size = dataset_size
        self.dataset_type = dataset_type
        self.columns = ['x', 'y', 'z']

        # more flexibility as needed.
        # self.x, self.y, self.z = src.utils.generate_samples_random(
        #     size=dataset_size,
        #     sType=dataset_type
        # )

        if self.dataset_type == "CI":
            self.x, self.y, self.z = src.utils.generate_ci_samples(dataset_size)
        elif self.dataset_type == "NI":
            self.x, self.y, self.z = src.utils.generate_ni_samples(dataset_size)

    def __getitem__(self, index: int) -> Dict:
        return {
            'x': self.x[index],
            'y': self.y[index],
            'z': self.z[index]
        }

    def __len__(self) -> int:
        return self.dataset_size

    @property
    def num_columns(self):
        return 3

    def datatypes(self) -> typing.Dict[str, DataType]:
        return {
            column_name: DataType.MULTIDIMENSIONAL
            for column_name in self.columns
        }

    def id(self):
        return id(self)
