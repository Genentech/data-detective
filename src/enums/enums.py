from enum import Enum


class DataType(Enum):
    # a list of supporteed data types throughout Data Detective
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    MULTIDIMENSIONAL = "multidimensional"
    IMAGE = "image"
    TIME_SERIES = "sequential"
    CUSTOM = "custom"

class ValidatorMethodParameter(Enum):
    TRAINING_SET = "training_set"
    VALIDATION_SET = "validation_set"
    TEST_SET = "test_set"
    ENTIRE_SET = "entire_set"
    EVERYTHING_BUT_INFERENCE_SET = "everything_but_inference_set"
    INFERENCE_SET = "inference_set"
    SPLIT_GROUP_SET = "split_group_set"

