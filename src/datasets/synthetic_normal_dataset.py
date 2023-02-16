from typing import Dict, Callable, List, Set

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset

from src.datasets.synthetic_data_generators import DataDetectiveDataset
from src.enums.enums import DataType


class SyntheticNormalDataset(Dataset):
    def __init__(self, num_cols: int = 1, dataset_size: int = 10000, loc: float = 0.):
        self.dataset_size = dataset_size
        self.columns = [f"feature_{j}" for j in range(num_cols)]
        self.outlier_index_set: Set[int] = set()

        #  dataframe = pd.DataFrame({ f"feature_{i}": np.random.normal(0, 1, size=10000) for i in range(10) }, columns=[f"feature_{j}" for j in range(10)])
        dataframe: DataFrame = pd.DataFrame({
            f"feature_{i}": np.random.normal(loc, 1, size=dataset_size)
            for i in range(self.num_columns)
        }, columns=self.columns)

        self.dataframe = dataframe

    def getitem(self, index: int):
        return self.dataframe.iloc[index].to_dict()

    def __len__(self) -> int:
        return self.dataset_size

    def __repr__(self):
        return self.dataframe.__repr__()

    @property
    def num_columns(self):
        return len(self.columns)

    def introduce_outliers(self,
           indices: List[int] = None,
           num_outliers : int = 1000,
           outlier_generation_function: Callable[[], List[float]] = None
        ) -> None:
        """
        Introduces outliers into the dataset. If indices is given, they will be at the start.
        """
        #TODO: add assertion that indices is a valid.
        if not indices:
            indices = np.random.choice(list(range(self.dataset_size)), size=num_outliers, replace=False)

        if not outlier_generation_function:
            num_cols = self.num_columns
            outlier_generation_function = lambda: np.random.normal(10, 1, size=num_cols)

        self.outlier_index_set |= set(indices)

        outliers = [outlier_generation_function() for _ in range(num_outliers)]
        for index, outlier in zip(indices, outliers):
            self.dataframe.iloc[index] = outlier

    def datatypes(self) -> Dict[str, DataType]:
        return {
            column_name: DataType.CONTINUOUS
            for column_name in self.columns
        }

    def to_matrix(self):
        return np.array([list(d.values()) for d in self[:].values()]).T,

    def id(self):
        return id(self)

