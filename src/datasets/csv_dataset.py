import os
import typing
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from src.enums.enums import DataType


class CSVDataset:
    def __init__(self, filepath: str, datatype_dict: Dict[str, DataType], unwrap_paths: bool = True):
        self.datatypes_dict = datatype_dict
        self.df = pd.read_csv(filepath)
        self.df = self.df[list(datatype_dict.keys())]
        self.df = self.df.dropna()
        self.unwrap_paths = unwrap_paths

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        @return: the length of the dataset
        """
        return self.df.__len__()

    def __getitem__(self, item) -> typing.Dict:
        """
        Returns an item from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the data and the label.
        """
        sample = self.df.iloc[item]

        for column_name, datatype in self.datatypes_dict.items():
            # if we are working with an image that is being represented as a path
            if datatype == DataType.IMAGE and isinstance(sample[column_name], str) and self.unwrap_paths:
                img_path = f"data/{sample[column_name]}"
                img = np.array(Image.open(img_path))
                sample[column_name] = img

        return sample

    def datatypes(self) -> Dict[str, DataType]:
        """
        Gives the datatypes of a the dataset sample.
        @return: the datatypes of a the dataset sample.
        """
        return self.datatypes_dict


"""
What are the run persistent caching options available?

First, let's examine our requirements: 
    - once a single datapoint is cached in a dataset, it should never have to be transformed again. 
    - this is basically mapping (dataset, index, transforms) to (output)
        - it seems that the crux of the issue is doing hashing/versioning/identification on teh dataset
        - problems to address:
            - how do we identify the specific instance of the dataset and load it next time? (we need to find the params that 
                - maybe we need (dataset => dataset version / hash => index => transforms)
                - but this doesn't quite make the mark, because a true data cache would only need the data about the datapoint
                and the transforms... 
                - so maybe the answer isn't even a dataset level cache but a data level cache! 
                - OMG this is going to be so much cleaner... now we just hash(sample || alphabetized_transform_list) and we are good to go!
                
            - how do we accommodate the situation where the dataset has changed slightly? (need to support dataset hashing)
        - i think we need to write some tests that are emblematic of 
        
        
    - we cannot assume that there is only one dataset of each chass type (there are splits)
    - ...so we need a form of identifier that basically maps to each version of a dataset...
        - maybe we could use wandb artifacts?
        - we absolutely need to be hashing the whole dataset... or do we?
"""