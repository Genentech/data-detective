from enum import Enum


class DataType(Enum):
    # a list of supporteed data types throughout Data Detective
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    MULTIDIMENSIONAL = "multidimensional"
    IMAGE = "image"
    SEQUENTIAL = "sequential"
    TEXT = "text"

class ValidatorMethodParameter(Enum):
    ENTIRE_SET = "entire_set"
    EVERYTHING_BUT_INFERENCE_SET = "everything_but_inference_set"
    INFERENCE_SET = "inference_set"
    SPLIT_GROUP_SET = "split_group_set"

