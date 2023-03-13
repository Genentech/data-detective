from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Any

import torch

from src.datasets.one_hot_encoded_dataset import OneHotEncodedDataset
from src.enums.enums import DataType
from src.transforms.embedding_transformer import TransformedDataset
from src.transforms.transform_library import TRANSFORM_LIBRARY
from src.utils import snake_to_camel, filter_dataset
from src.validators.data_validator import DataValidator


class DataDetectiveEngine:
    def __init__(self):
        """
        Initializes the data detective enigne. Needs to initialize codebooks for:
            - validator methods
            - validators
            - transforms
            - (aggregators ?)
        """
        self.transform_dict = deepcopy(TRANSFORM_LIBRARY)
        self.validator_dict = {}

    def register_validator(self, validator):
        self.validator_dict[validator().name()] = validator

    def register_transform(self, transform, transform_name):
        self.transform_dict[transform_name] = transform

    def validate_from_schema(self, config_dict: Dict, data_object: Dict) -> Dict:
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
            validator_class_object: DataValidator = self.validator_name_to_object(validator_class_name)
            include_lst: List[str] = validator_params.get("include", ['.*'])
            validator_kwargs: Dict = validator_params.get("validator_kwargs", {})

            # filter the datasets by the inclusion criteria.
            filtered_data_object = {}

            for key, dataset in data_object.items():
                # TODO: implement filtering correctly on the torch datasets.
                filtered_data_object[key] = filter_dataset(dataset, include_lst)
                filtered_data_object[key] = OneHotEncodedDataset(filtered_data_object[key])

            if 'transforms' in config_dict.keys():
                transforms_dict = config_dict['transforms']
                transforms_dict = self.parse_transforms(transforms_dict, filtered_data_object)
                filtered_transformed_data_object = {
                    data_object_name: TransformedDataset(data_object_part, transforms_dict) for
                    data_object_name, data_object_part in filtered_data_object.items()}
            else:
                filtered_transformed_data_object = filtered_data_object

            print(f"running {validator_class_name}...")
            result_dict[validator_class_name] = validator_class_object.validate(
                data_object=filtered_transformed_data_object,
                validator_kwargs=validator_kwargs)

        if default_inclusion:
            # TODO: need results from the rest of validators that are listed.
            # TODO: do this later
            pass

        return result_dict

    def parse_transforms(self, transform_dict: Dict[str, Any], data_object):
        output_dict = defaultdict(lambda: [])

        sample_dataset = list(data_object.items())[0][1]
        while isinstance(sample_dataset, torch.utils.data.Subset):
            sample_dataset = sample_dataset.dataset
        datatypes = sample_dataset.datatypes()

        for data_type, transform_specification_list in transform_dict.items():
            if data_type not in DataType._value2member_map_:
                raise Exception(
                    f"datasets type {data_type} from transform dict does not exist in DataType enumeration.")

            relevant_columns = [column_name for (column_name, dtype) in datatypes.items() if data_type == dtype.value]

            for transform_specification in transform_specification_list:
                name = transform_specification['name']
                in_place = transform_specification['in_place'].lower() == 'true'
                options = transform_specification['options']

                ### getting transform from self object
                transform = TRANSFORM_LIBRARY.get(name, self.transform_dict.get(name))
                if not transform:
                    raise Exception(f"Transform {name} not found in transform library or registered to Data Detective Engine.")
                ###

                transform.initialize_transform(options)
                transform.in_place = in_place

                for column_name in relevant_columns:
                    output_dict[column_name].append(transform)

        return dict(output_dict)

    def validator_name_to_object(self, validator_class_name: str) -> DataValidator:
        """
        Finds the appropriate validators class object for the input string

        @param validator_class_name: the name for the validators class object
        @return: the DataValidator object itself.
        """
        if validator_class_name in self.validator_dict.keys():
            return self.validator_dict[validator_class_name]

        camel_class_name = snake_to_camel(validator_class_name)
        # the below line will fail and raise exception if the module does not exist in code.
        module = __import__(f'src.validators.{validator_class_name}', fromlist=[camel_class_name])
        return getattr(module, camel_class_name)