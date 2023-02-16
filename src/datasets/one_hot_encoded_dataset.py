import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch

from constants import FloatTensor
from src.enums.enums import DataType


class OneHotEncodedDataset:
    def __init__(self,
        dataset: torch.utils.data.Dataset,
    ):
        self.dataset = dataset
        self.prepare_one_hot_encodings()

    def datatypes(self):
        return self.dataset.datatypes()

    @staticmethod
    def is_one_hot_encoded(X):
        # src: https://stackoverflow.com/questions/66670839/how-to-check-if-my-data-is-one-hot-encoded
        try:
            return (X.sum(axis=1)-torch.ones(X.shape[0])).sum()==0
        except Exception:
            return False

    def prepare_one_hot_encodings(self):
        categorical_variables = {column: [] for column, datatype in self.dataset.datatypes().items() if datatype == DataType.CATEGORICAL}

        self.one_hot_encoded_catvars = {}

        for idx in range(self.dataset.__len__()):
            sample = self.dataset[idx]
            for column in categorical_variables.keys():
                categorical_variables[column].append(sample[column])

        for column, list_of_vars in categorical_variables.items():
            # src: https://www.geeksforgeeks.org/how-to-check-if-an-object-is-iterable-in-python/
            column_dim = list_of_vars[0].__len__() if hasattr(list_of_vars[0], '__iter__') else 1
            X = np.array(list_of_vars)
            X = X.reshape((-1, column_dim))
            if not OneHotEncodedDataset.is_one_hot_encoded(X):
                self.one_hot_encoded_catvars[column] = np.array(OneHotEncoder().fit_transform(X).todense())

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        for column_name in sample.keys():
            if column_name in self.one_hot_encoded_catvars.keys():
                sample[column_name] = self.one_hot_encoded_catvars[column_name][idx]

        return sample

    def __getattr__(self, item):
        if hasattr(self, item):
            return getattr(self, item)

        return getattr(self.dataset, item)

    def __len__(self):
        return self.dataset.__len__()

