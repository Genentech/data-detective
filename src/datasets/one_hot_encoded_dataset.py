import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import DataLoader

from src.enums.enums import DataType


class OneHotEncodedDataset:
    def __init__(self,
        dataset: torch.utils.data.Dataset,
    ):
        self.dataset = dataset
        self.prepare_one_hot_encodings()

    def datatypes(self):
        """
        Gives the datatypes of a dataset sample.
        @return: the datatypes of a dataset sample.
        """
        return self.dataset.datatypes()

    @staticmethod
    def is_one_hot_encoded(X: np.array) -> bool:
        """
        Determines if the data matrix (X) is already one hot encoded.

        @param X: the data matrix to check.
        @return: a bool detailing if the data matrix is one hot encoded.
        """
        # src: https://stackoverflow.com/questions/66670839/how-to-check-if-my-data-is-one-hot-encoded
        try:
            return (X.sum(axis=1)-torch.ones(X.shape[0])).sum()==0
        except Exception:
            return False

    def prepare_one_hot_encodings(self) -> None:
        """
        Prepares the one hot encoding transforms by fitting them to the categorical variables in the dataset.
        """
        categorical_variables = {column: [] for column, datatype in self.dataset.datatypes().items() if datatype == DataType.CATEGORICAL}

        self.one_hot_encoded_catvars = {}

        for sample in iter(DataLoader(self.dataset)):
            for column in categorical_variables.keys():
                categorical_variables[column].append(sample[column])

        for column, list_of_vars in categorical_variables.items():
            # src: https://www.geeksforgeeks.org/how-to-check-if-an-object-is-iterable-in-python/
            column_dim = list_of_vars[0].__len__() if hasattr(list_of_vars[0], '__iter__') and not isinstance(list_of_vars[0], str) else 1
            X = np.array(list_of_vars)
            X = X.reshape((-1, column_dim))
            if not OneHotEncodedDataset.is_one_hot_encoded(X):
                self.one_hot_encoded_catvars[column] = np.array(OneHotEncoder().fit_transform(X).todense())

    def __getitem__(self, idx: int):
        """
        Returns an item from the dataset, applying one hot categorical transforms as well.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the data.
        """
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

