from typing import Dict, Callable, List, Set
import joblib

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from src.datasets.data_detective_dataset import DataDetectiveDataset

from src.enums.enums import DataType


class SyntheticNormalDatasetForIds(DataDetectiveDataset):
    def __init__(self, num_cols: int = 1, dataset_size: int = 10000, loc: float = 0., **kwargs):
        self.dataset_size = dataset_size

        self.columns = [f"feature_{j}" for j in range(num_cols)]
        self.outlier_index_set: Set[int] = set()

        dataframe: DataFrame = pd.DataFrame({
            f"feature_{i}": np.random.normal(loc, 1, size=dataset_size)
            for i in range(self.num_columns)
        }, columns=self.columns)

        self.dataframe = dataframe

        show_id = kwargs.get("show_id", True)
        kwargs["show_id"] = show_id
        include_subject_id_in_data = kwargs.get("include_subject_id_in_data", True)
        kwargs["include_subject_id_in_data"] = include_subject_id_in_data 

        super().__init__(**kwargs)

    # idx represents samddple index if exists
    # otherwise represents internal idx. 
    def __getitem__(self, index: int):
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
            column_name: DataType.MULTIDIMENSIONAL
            for column_name in self.columns
        }

class SyntheticNormalDatasetForIdsWithSampleIds(SyntheticNormalDatasetForIds):
    def __init__(self, num_cols: int = 1, dataset_size: int = 10000, loc: float = 0.):
        super().__init__(num_cols=num_cols, dataset_size=dataset_size, loc=loc, sample_ids=[joblib.hash(i) for i in range(dataset_size)])
        self.dataframe["sample_id"] = [joblib.hash(idx) for idx in self.dataframe.index]
        self.dataframe = self.dataframe.set_index("sample_id")

    def __getitem__(self, index: int):
        return self.dataframe.loc[index].to_dict()