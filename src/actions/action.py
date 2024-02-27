from abc import abstractmethod
from typing import Dict, Union
import pandas as pd
import torch


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
    
    Actions are functional mutations on the dataset, meaning that they 
    """
    @abstractmethod
    def get_new_data_object(
        schema: Dict, 
        data_object: Dict[str, Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]],
        aggregated_results: pd.DataFrame
    ): 
        pass

class RemoveTopKAnomalousSamplesAction(Action):
    def __init__(self, k=10, remove_all_duplicates=True):
        self.k = k
        self.remove_all_duplicates = remove_all_duplicates

    def get_new_data_object(
        data_object: Dict[str, Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]],
        aggregated_results: pd.DataFrame, 
        aggregated_results_key: str = None,
    ): 
        if aggregated_results_key is not None: 
            assert(aggregated_results_key in aggregated_results.keys())
        else: 
            aggregated_results_key = min(aggregated_results.keys(), key=len)

        items_most_to_least_anomalous = list(aggregated_results.sort_values(aggregated_results_key).index)
        indices = [int(item_name.split(" ")[1]) for item_name in items_most_to_least_anomalous]

class ResplitDataAction(Action):
    def get_new_data_object(
        data_object: Dict[str, Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]],
        aggregated_results: pd.DataFrame, 
    ): 

        return data_object