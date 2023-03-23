from collections import defaultdict
from random import randint
from typing import Dict, List, Union, Any

import numpy as np
import torch
import torch.utils.data
from joblib import delayed, Parallel
from torch.utils.data import DataLoader, Dataset

from src.datasets.column_filtered_dataset import ColumnFilteredDataset
from src.datasets.one_hot_encoded_dataset import OneHotEncodedDataset
from src.enums.enums import DataType
from src.transforms.embedding_transformer import TransformedDataset
from src.transforms.transform_library import TRANSFORM_LIBRARY
from src.validators.data_validator import DataValidator


def snake_to_camel(snake_case_string: str) -> str:
    """
    Converts a string from snake case to camel case.

    @param snake_case_string: the snake case string
    @return: the camel case string
    """
    return ''.join(word.title() for word in snake_case_string.split('_'))

def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
                            debug=False, normalize = True, seed = None, dist_z = 'gaussian'):
    # stolen shamelessly from https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/gcit/utils.py
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        f1, f2 to be within {x,x^2,x^3,tanh x, e^{-|x|}, cos x}

    Output:
        Samples X, Y, Z
    '''

    def same(x):
        return x

    def cube(x):
        return np.power(x, 3)

    def negexp(x):
        return np.exp(-np.abs(x))

    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if fixed_function == 'linear':
        f1 = same
        f2 = same
    else:
        I1 = randint(2, 6)
        I2 = randint(2, 6)

        if I1 == 2:
            f1 = np.square
        elif I1 == 3:
            f1 = cube
        elif I1 == 4:
            f1 = np.tanh
        elif I2 == 5:
            f1 = negexp
        else:
            f1 = np.cos

        if I2 == 2:
            f2 = np.square
        elif I2 == 3:
            f2 = cube
        elif I2 == 4:
            f2 = np.tanh
        elif I2 == 5:
            f2 = negexp
        else:
            f2 = np.cos
    if debug:
        print(f1, f2)

    num = size

    if dist_z == 'gaussian':
        cov = np.eye(dz)
        mu = np.ones(dz)
        Z = np.random.multivariate_normal(mu, cov, num)
        Z = np.matrix(Z)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z,(num,dz))
        Z = np.matrix(Z)

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ax = np.matrix(Ax)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)
    Ay = np.matrix(Ay)

    Axy = np.random.rand(dx, dy)
    for i in range(dy):
        Axy[:, i] = Axy[:, i] / np.linalg.norm(Axy[:, i], ord=1)
    Axy = np.matrix(Axy)

    temp = Z * Ax
    m = np.mean(np.abs(temp))
    nstd = nstd * m

    if sType == 'CI':
        X = f1(Z * Ax + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(Z * Ay + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = f1(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = f2(2 * X * Axy + Z * Ay)

    if normalize:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)

def generate_ci_samples(n_samples):
    z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z]).astype(float)
    y = np.vstack([np.random.multinomial(20, p) for p in z]).astype(float)

    return x, y, z

def generate_ni_samples(n_samples):
    z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z]).astype(float)
    y = np.vstack([np.random.multinomial(20, p) for p in z]).astype(float)

    for ind, i in enumerate(range(n_samples)):
        if np.random.binomial(1, 0.5) == 0:
            y[ind] = x[ind]

    return x, y, z

def validator_name_to_object(validator_class_name: str) -> DataValidator:
    """
    Finds the appropriate validators class object for the input string

    @param validator_class_name: the name for the validators class object
    @return: the DataValidator object itself.
    """
    camel_class_name = snake_to_camel(validator_class_name)
    module = __import__(f'src.validators.{validator_class_name}', fromlist=[camel_class_name])
    return getattr(module, camel_class_name)

def filter_dataset(dataset: torch.utils.data.Dataset, include_lst: List[str]) -> torch.utils.data.Dataset:
    """
    Filters a dataset so that only columns matching any include_lst regexp are included.

    @param dataset: the dataset to filter
    @param include_lst: the list of regexes to include
    @return:
    """
    return ColumnFilteredDataset(dataset, matching_regexes=include_lst)

def unfilter_dataset(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    if isinstance(dataset, ColumnFilteredDataset):
        return dataset.unfiltered_dataset

    raise Exception("Trying to unfilter an already unfiltered dataset.")

def get_class_counts(train_dataset: Dataset):
    class_counts: defaultdict = defaultdict(lambda: 0)
    dataset_length = train_dataset.__len__()

    for idx in range(dataset_length):
        sample: dict[str, Union[int, torch.FloatTensor]] = train_dataset[idx]
        sample_class: int = sample['label']
        class_counts[sample_class] += 1

    return class_counts