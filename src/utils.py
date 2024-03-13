import copy
import os
from collections import defaultdict
from random import randint
from typing import Dict, List, Optional, Sequence, Union, Any

import joblib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.utils.data
from joblib import delayed, Parallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.datasets.data_detective_dataset import DataDetectiveDataset

from src.datasets.column_filtered_dataset import ColumnFilteredDataset
from src.datasets.one_hot_encoded_dataset import OneHotEncodedDataset
from src.enums.enums import DataType, ValidatorMethodParameter
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

"""
<FCIT UTILITIES>
src: https://github.com/kjchalup/fcit
"""

import os
import time
import joblib
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse


def interleave(x, z, seed=None):
    """ Interleave x and z dimension-wise.

    Args:
        x (n_samples, x_dim) array.
        z (n_samples, z_dim) array.

    Returns
        An array of shape (n_samples, x_dim + z_dim) in which
            the columns of x and z are interleaved at random.
    """
    state = np.random.get_state()
    np.random.seed(seed or int(time.time()))
    total_ids = np.random.permutation(x.shape[1]+z.shape[1])
    np.random.set_state(state)
    out = np.zeros([x.shape[0], x.shape[1] + z.shape[1]])
    out[:, total_ids[:x.shape[1]]] = x
    out[:, total_ids[x.shape[1]:]] = z
    return out

def cv_besttree(x, y, z, cv_grid, logdim, verbose, prop_test):
    """ Choose the best decision tree hyperparameters by
    cross-validation. The hyperparameter to optimize is min_samples_split
    (see sklearn's DecisionTreeRegressor).

    Args:
        x (n_samples, x_dim): Input data array.
        y (n_samples, y_dim): Output data array.
        z (n_samples, z_dim): Optional auxiliary input data.
        cv_grid (list of floats): List of hyperparameter values to try.
        logdim (bool): If True, set max_features to 'log2'.
        verbose (bool): If True, print out extra info.
        prop_test (float): Proportion of validation data to use.

    Returns:
        DecisionTreeRegressor with the best hyperparameter setting.
    """
    xz_dim = x.shape[1] + z.shape[1]
    max_features='log2' if (logdim and xz_dim > 10) else None
    if cv_grid is None:
        min_samples_split = 2
    elif len(cv_grid) == 1:
        min_samples_split = cv_grid[0]
    else:
        clf = DecisionTreeRegressor(max_features=max_features)
        splitter = ShuffleSplit(n_splits=3, test_size=prop_test)
        cv = GridSearchCV(estimator=clf, cv=splitter,
            param_grid={'min_samples_split': cv_grid}, n_jobs=-1)
        cv.fit(interleave(x, z), y)
        min_samples_split = cv.best_params_['min_samples_split']
    if verbose:
        print('min_samples_split: {}.'.format(min_samples_split))
    clf = DecisionTreeRegressor(max_features=max_features,
        min_samples_split=min_samples_split)
    return clf

def obtain_error(data_and_i):
    """ 
    A function used for multithreaded computation of the fcit test statistic.
    data['x']: First variable.
    data['y']: Second variable.
    data['z']: Conditioning variable.
    data['data_permutation']: Permuted indices of the data.
    data['perm_ids']: Permutation for the bootstrap.
    data['n_test']: Number of test points.
    data['clf']: Decision tree regressor.
    """
    data, i = data_and_i
    x = data['x']
    y = data['y']
    z = data['z']
    if data['reshuffle']:
        perm_ids = np.random.permutation(x.shape[0])
    else:
        perm_ids = np.arange(x.shape[0])
    data_permutation = data['data_permutation'][i]
    n_test = data['n_test']
    clf = data['clf']

    x_z = interleave(x[perm_ids], z, seed=i)

    clf.fit(x_z[data_permutation][n_test:], y[data_permutation][n_test:])
    return mse(y[data_permutation][:n_test],
        clf.predict(x_z[data_permutation][:n_test]))


def fcit_test(x, y, z=None, num_perm=8, prop_test=.1,
    discrete=(False, False), plot_return=False, verbose=False,
    logdim=False, cv_grid=[2, 8, 64, 512, 1e-2, .2, .4], **kwargs):
    """ Fast conditional independence test, based on decision-tree regression.

    See Chalupka, Perona, Eberhardt 2017 [arXiv link coming].

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable. If z==None (default),
            then performs an unconditional independence test.
        num_perm: Number of data permutations to estimate
            the p-value from marginal stats.
        prop_test (int): Proportion of data to evaluate test stat on.
        discrete (bool, bool): Whether x or y are discrete.
        plot_return (bool): If True, return statistics useful for plotting.
        verbose (bool): Print out progress messages (or not).
        logdim (bool): If True, set max_features='log2' in the decision tree.
        cv_grid (list): min_impurity_splits to cross-validate when training
            the decision tree regressor.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    # Compute test set size.
    n_samples = x.shape[0]
    n_test = int(n_samples * prop_test)

    if z is None:
        z = np.empty([n_samples, 0])

    if discrete[0] and not discrete[1]:
        # If x xor y is discrete, use the continuous variable as input.
        x, y = y, x
    elif x.shape[1] < y.shape[1]:
        # Otherwise, predict the variable with fewer dimensions.
        x, y = y, x

    # Normalize y to make the decision tree stopping criterion meaningful.
    y = StandardScaler().fit_transform(y)

    # Set up storage for true data and permuted data MSEs.
    d0_stats = np.zeros(num_perm)
    d1_stats = np.zeros(num_perm)
    data_permutations = [
        np.random.permutation(n_samples) for i in range(num_perm)]

    # Compute mses for y = f(x, z), varying train-test splits.
    clf = cv_besttree(x, y, z, cv_grid, logdim, verbose, prop_test=prop_test)
    datadict = {
            'x': x,
            'y': y,
            'z': z,
            'data_permutation': data_permutations,
            'n_test': n_test,
            'reshuffle': False,
            'clf': clf,
            }
    d1_stats = np.array(joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
        joblib.delayed(obtain_error)((datadict, i)) for i in range(num_perm)))

    # Compute mses for y = f(x, reshuffle(z)), varying train-test splits.
    if z.shape[1] == 0:
        x_indep_y = x[np.random.permutation(n_samples)]
    else:
        x_indep_y = np.empty([x.shape[0], 0])
    clf = cv_besttree(x_indep_y, y, z, cv_grid, logdim,
                      verbose, prop_test=prop_test)
    datadict['reshuffle'] = True
    datadict['x'] = x_indep_y
    d0_stats = np.array(joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
        joblib.delayed(obtain_error)((datadict, i)) for i in range(num_perm)))

    if verbose:
        np.set_printoptions(precision=3)
        print('D0 statistics: {}'.format(d0_stats))
        print('D1 statistics: {}\n'.format(d1_stats))

    # Compute the p-value (one-tailed t-test
    # that mean of mse ratios equals 1).
    t, p_value = ttest_1samp(d0_stats / d1_stats, 1)
    if t < 0:
        p_value = 1 - p_value / 2
    else:
        p_value = p_value / 2

    if plot_return:
        return (p_value, d0_stats, d1_stats)
    else:
        return p_value

"""
</ FCIT UTILITIES>
"""

def get_split_group_keys(data_object): 
    split_group_keys = []

    for key, potential_split_group_set in data_object.items(): 
        if key not in  [member.value for member in ValidatorMethodParameter]:
            # then it must be a split group; so make sure it is not malformed
            split_group_set = potential_split_group_set
            assert isinstance(split_group_set, dict)
            assert len(split_group_set.items()) > 1, f"split group set {split_group_set} only has one item."

            for dataset_name, dataset in split_group_set.items(): 
                assert isinstance(dataset_name, str)
                assert isinstance(dataset, torch.utils.data.Dataset)

            split_group_keys.append(key)
        else: 
            assert isinstance(key, str)
            assert isinstance(potential_split_group_set, torch.utils.data.Dataset)


    return split_group_keys

#########################3
#
#
# OCCLUSION UTILITIES
#
#
##########################

class OcclusionTransform(torch.nn.Module):
    def __init__(self, width=5):
        super().__init__()
        self.width = width
        
        if width % 2 != 1: 
            raise Exception("Width must be an odd number")
        
    def forward(self, tensor, loc): 
        tensor = copy.deepcopy(tensor)
        width = self.width
        
        diff = (width - 1) / 2
        first_dim, second_dim = loc[0], loc[1]
        
        min_val_first = np.round(max(0, first_dim - diff)).astype(int)
        min_val_second = np.round(max(0, second_dim - diff)).astype(int)
        
        max_val_first = np.round(min(tensor.shape[1], first_dim + diff + 1)).astype(int)
        max_val_second = np.round(min(tensor.shape[1], second_dim + diff + 1)).astype(int)
        
        tensor[:, min_val_first:max_val_first, min_val_second:max_val_second].fill_(0)
        
        return tensor

def plot_occ_results(img, localized_anomaly_score, width, color_bounds):
    if color_bounds != "auto":
        vmin, vmax = color_bounds
    else: 
        vmin, vmax = None, None
    
    im = box_blur(localized_anomaly_score, width)
    im = im.reshape(im.shape[-2:])
    
    plt.imshow(img.reshape(img.shape[-2:]), cmap='Greys_r')
    plt.colorbar()
    plt.suptitle("Original Image")
    plt.show()

    plt.imshow(im, vmin=vmin, vmax=vmax, cmap='plasma')
    plt.colorbar()
    plt.suptitle(f"Blurred anomaly occlusion heatmap (width {width})")
    plt.show()

    plt.imshow(localized_anomaly_score, vmin=vmin, vmax=vmax, cmap='plasma')
    plt.colorbar()
    plt.suptitle(f"Unblurred anomaly occlusion heatmap (width {width})")
    plt.show()

    
def box_blur(tensor, width):
    def get_sum(sum_table,
            min_val_first, 
            min_val_second, 
            max_val_first, 
            max_val_second
    ):
        x = 0
        x += sum_table[max_val_first][max_val_second] 

        if min_val_second != 0: 
            x -= sum_table[max_val_first][min_val_second - 1]

        if min_val_first != 0:
            x -= sum_table[min_val_first - 1][max_val_second] 

        if not (min_val_first == 0 or min_val_second == 0):
            x += sum_table[min_val_first - 1][min_val_second - 1]

        return x

    def compute_overlaps(tensor, patch_size=(3, 3), patch_stride=(1, 1)):
        width = (patch_size[0] - 1) // 2
        tensor = torch.FloatTensor(tensor)
        tensor = np.pad(tensor, (width, width, width, width), "constant", 0)
        while len(tensor.shape) < 4:
            tensor = tensor.reshape((-1, *tensor.shape))

        n, c, h, w = tensor.size()
        px, py = patch_size
        sx, sy = patch_stride
        nx = ((w-px)//sx)+1
        ny = ((h-py)//sy)+1

        overlaps = torch.zeros(tensor.size()).type_as(tensor.data)
        for i in range(ny):
            for j in range(nx):
                overlaps[:, :, i*sy:i*sy+py, j*sx:j*sx+px] += 1
        overlaps = torch.autograd.Variable(overlaps)
        return overlaps[:,:,width:-width,width:-width]

    sum_table = tensor.cumsum(axis=0).cumsum(axis=1)

    res = np.zeros(tensor.shape)
    first_dim_size = tensor.shape[-2]
    second_dim_size = tensor.shape[-1]
    diff = np.round((width - 1) // 2).astype(int)


    for first_dim in range(first_dim_size): 
        for second_dim in range(second_dim_size): 

            min_val_first = np.round(max(0, first_dim - diff)).astype(int)
            min_val_second = np.round(max(0, second_dim - diff)).astype(int)

            max_val_first = np.round(min(tensor.shape[1] - 1, first_dim + diff)).astype(int)
            max_val_second = np.round(min(tensor.shape[1] - 1, second_dim + diff)).astype(int)

            res[first_dim][second_dim] = get_sum(sum_table,
                min_val_first, 
                min_val_second, 
                max_val_first, 
                max_val_second
            )

    overlap = compute_overlaps(tensor, patch_size=(width, width))
        
    return res / overlap
    
    
def occlusion_interpretability(img, model, occ, color_bounds="auto"):    
    occluded_image_dict = {}
    for first_dim in tqdm(range(img.shape[1])):
        for second_dim in range(img.shape[2]):
            occluded = occ(img, (first_dim, second_dim))
            occluded_image_dict[tuple((first_dim, second_dim))] = occluded

            
    resnet = TRANSFORM_LIBRARY['resnet50']
    resnet.initialize_transform({})
    embeddings = np.concatenate([resnet(img) for img in tqdm(occluded_image_dict.values())], axis=0)
#     embeddings = np.concatenate(Parallel(n_jobs=6)(delayed(resnet)(img) for img in tqdm(occluded_image_dict.values())), axis=0)
    localized_anomaly_scores = model.decision_function(embeddings)
    reshaped_localized_anomaly_score = torch.FloatTensor(list(localized_anomaly_scores)).reshape(img.shape[-2:])
    plot_occ_results(img, reshaped_localized_anomaly_score, occ.width, color_bounds)
    return reshaped_localized_anomaly_score