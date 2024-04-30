from abc import abstractmethod
import copy
from typing import Dict, Union
import pandas as pd
import torch

from src.datasets.data_detective_dataset import DataDetectiveDataset


class Action: 
    """
    An action is a functional object that accepts the following:
    - data_detective_schema
    - data_object            
    - aggregated results               
    and returns an updated version of the data_object that may remedy some of the problems of the previous dataset.

    Some potential actions include:
    - resplitting the dataset if there is a clear covariate shift present.
    - dropping the top_k anomalous samples in the entire_set before splitting. 
    - eliminating duplicates from the dataset. 
    
    Actions are functional mutations on the dataset, meaning that they take an object and modify it.
    """
    @abstractmethod
    def get_new_data_object(
        schema: Dict, 
        data_object: Dict[str, Union[DataDetectiveDataset, Dict[str, DataDetectiveDataset]]],
        aggregated_results: pd.DataFrame
    ): 
        pass

class RemoveTopKAnomalousSamplesAction(Action):
    def __init__(self, k=10, remove_all_duplicates=True):
        self.k = k
        self.remove_all_duplicates = remove_all_duplicates

    def get_new_data_object(
        self,
        data_object: Dict[str, Union[DataDetectiveDataset, Dict[str, DataDetectiveDataset]]],
        aggregated_results: pd.DataFrame, 
        aggregated_results_key: str = None,
    ) -> Dict[str, Union[Dict[str, DataDetectiveDataset], DataDetectiveDataset]]: 
        """
        Removes top k anomalous samples from the entire set, propagating results to all datasets and splits.
        If there are duplicates, it will continue removing samples from worst to best until at least K samples are removed.

        Aggregated results is a results dataframe indexed on samples. This means that each sample id can correspond to more
        than one sample in the dataset, if there are duplicates. 
        """
        entire_set = data_object['entire_set']

        if aggregated_results_key is not None: 
            assert(aggregated_results_key in aggregated_results.keys())
        else: 
            aggregated_results_key = min(aggregated_results.keys(), key=len)

        sample_ids_to_remove = set() 
        num_datapoints_removed = 0

        while num_datapoints_removed < self.k: 
            worst_sample_id = aggregated_results[aggregated_results_key].idxmin()
            frequency_of_worst_sample = entire_set.index_df['sample_id'].value_counts()[worst_sample_id]

            sample_ids_to_remove.add(worst_sample_id)
            num_datapoints_removed += frequency_of_worst_sample
            aggregated_results = aggregated_results.drop(worst_sample_id)

        data_object = copy.copy(data_object)
        for key, dataset_or_split_group_dict in data_object.items(): 
            if isinstance(dataset_or_split_group_dict, dict):
                split_group_dict = dataset_or_split_group_dict
                for key, dataset in split_group_dict.items():
                    split_group_dict[key] = dataset.remove_samples(sample_ids_to_remove)
            else: 
                dataset = dataset_or_split_group_dict
                data_object[key] = dataset.remove_samples(sample_ids_to_remove)

        print(f"{len(sample_ids_to_remove)} samples removed and {num_datapoints_removed} datapoints removed.")
        print(f"removed samples {sample_ids_to_remove}")

        return data_object

class ResplitDataAction(Action):
    def get_new_data_object(
        data_object: Dict[str, Union[DataDetectiveDataset, Dict[str, DataDetectiveDataset]]],
        aggregated_results: pd.DataFrame, 
    ): 
        """
        todo: implement
        An action to take a data object and resplit it from the entire set.
        """
        return data_object