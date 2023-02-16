from collections import defaultdict
from random import randint
from typing import Dict, List, Union, Any

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

from src.datasets.column_filtered_dataset import ColumnFilteredDataset
from src.datasets.one_hot_encoded_dataset import OneHotEncodedDataset
from src.enums.enums import DataType
from src.transforms.embedding_transformer import TransformedDataset
from src.transforms.transform_library import TRANSFORM_LIBRARY
from src.validators.data_validator import DataValidator


def validate_from_schema(config_dict: Dict, data_object: Dict) -> Dict:
    """
    Validates a particular parameter object (dict of things like train_dataset and test_dataset) against
    all validators specified in the config file.

    @param config_dict: the config dict to get what validators to use. if default-inclusion is set to on, then
    it should also include all default validators and apply them everywhere that is relevant.
    @param data_object: the dict of things like train_dataset, test_dataset, etc.
    @return: a dict mapping (in the following order) feature -> validators -> val_method -> key results
    """

    """
    To some degree, we have to consider the fact that feature list simply isn't enough... as not every test 
    of validation can occur on a single feature list. For example, imagine that you needed to do a covariate check on 
    a train/test/val split... this could actually be done on a feature level. 

    Ok, here's a better example. imagine that you are trying to write a method that handles spurious correlations
    across features. Then it is more of a multi-feature input... 

    Let's formalize this. You have an n x d datasets matrix of continuous datasets, where k < d features are needed to be looked 
    at, all at once, to find out whether some bias exists by looking at all k columns at once.

    An example of this could be CI testing, where we have X ⫫ Y | A, B, C; this requires us to be able to look at the 
    n x 5 matrix including x, y, a, b, and c all at once. How do you generalize to this case?

    A n x 2 example of this is spurious correlations with an extra attribute. How do you specify this in a
    generalizable, abstract way?

    Let's start by looking at the datasets that we need to do something like CI testing:
        - the datasets object
        - the two variable feature names that are CI 
        - the conditional feature names 

    Let's think about the spurious correlation case for variable redundancy, where there might be more generic-ness.
        - the datasets object 
        - a SET of features that you might need to compare against
        - in this case, it's really not all that obvious that you need a different approach!

    So let's focus on CI testing. It looks like we just need more fine-grained detail in the "include" section...

    CI_Data_Validator.validate(conditional_independnences=list(dict(ci_info)))

    Idea: include is just a ~preliminary filter~ that we will use before applying the validators based on the behavior 
    specified in the docstring; best practice constitutes including the minimal amount of rows or columns necessary to 
    do the validation. 
    """
    # this specifies whether to use the default validators or not.
    result_dict = {}

    default_inclusion = config_dict.get("default_inclusion", True)
    validators = config_dict["validators"]

    # get all (validator_class, data_object, feature_lst) entries
    for validator_class_name, validator_params in validators.items():
        validator_class_object: DataValidator = validator_name_to_object(validator_class_name)
        include_lst: List[str] = validator_params.get("include", ['.*'])
        validator_kwargs: Dict = validator_params.get("validator_kwargs", {})

        # filter the datasets by the inclusion criteria.
        filtered_data_object = {}

        for key, dataset in data_object.items():
            #TODO: implement filtering correctly on the torch datasets.
            filtered_data_object[key] = filter_dataset(dataset, include_lst)
            filtered_data_object[key] = OneHotEncodedDataset(filtered_data_object[key])

        if 'transforms' in config_dict.keys():
            transforms_dict = config_dict['transforms']
            transforms_dict = parse_transforms(transforms_dict, filtered_data_object)
            filtered_transformed_data_object = {data_object_name: TransformedDataset(data_object_part, transforms_dict) for
                           data_object_name, data_object_part in filtered_data_object.items()}
        else:
            filtered_transformed_data_object = filtered_data_object

        ## delete, for debugging
        # train = filtered_transformed_data_object['training_set']
        # x = train[:3]
        ## delete, for debugging


        print(f"running {validator_class_name}...")
        result_dict[validator_class_name] = validator_class_object.validate(data_object=filtered_transformed_data_object,
                                                                            validator_kwargs=validator_kwargs)
        c=4

    if default_inclusion:
        #TODO: need results from the rest of validators that are listed.
        #do this later
        pass

    return result_dict

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

def parse_transforms(transform_dict: Dict[str, Any], data_object):
    output_dict = defaultdict(lambda: [])

    sample_dataset = list(data_object.items())[0][1]
    while isinstance(sample_dataset, torch.utils.data.Subset):
        sample_dataset = sample_dataset.dataset
    datatypes = sample_dataset.datatypes()

    for data_type, transform_specification_list in transform_dict.items():
         if data_type not in DataType._value2member_map_:
             raise Exception(f"datasets type {data_type} from transform dict does not exist in DataType enumeration.")

         relevant_columns = [column_name for (column_name, dtype) in datatypes.items() if data_type == dtype.value]

         for transform_specification in transform_specification_list:
             name = transform_specification['name']
             in_place = transform_specification['in_place'].lower() == 'true'
             options = transform_specification['options']

             transform = TRANSFORM_LIBRARY[name]
             transform.initialize_transform(options)
             transform.in_place = in_place

             for column_name in relevant_columns:
                 output_dict[column_name].append(transform)

    return dict(output_dict)

class FixShape(torch.nn.Module):
    def forward(self, x):
        if len(x.shape == 3):
            return torch.unsqueeze(x, 0)
        else:
            return x