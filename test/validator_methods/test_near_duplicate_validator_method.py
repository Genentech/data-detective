from collections import defaultdict
import copy
from typing import Dict
import joblib

import numpy as np
import scipy
import torch
import torchvision.transforms as transforms
from constants import FloatTensor

from src.datasets.data_detective_dataset import DataDetectiveDataset, dd_random_split
from src.datasets.synthetic_normal_dataset import SyntheticCategoricalDataset, SyntheticNormalDatasetContinuous
from src.datasets.tutorial_dataset import TutorialDataset
from src.data_detective_engine import DataDetectiveEngine
from src.enums.enums import DataType

class DuplicateTestingDataset(DataDetectiveDataset):

    SUPPORTED_DATATYPES = [DataType.MULTIDIMENSIONAL, DataType.IMAGE]

    # duplicate type options: exact, near 
    # duplication_Breadth: sample, column
    def __init__(self, base_dataset, duplicate_type="exact", duplication_breadth="sample", duplicate_rate = 0.1):
        self.base_dataset = base_dataset
        self.duplicate_type = duplicate_type
        self.duplication_breadth = duplication_breadth
        self.relevant_datatypes = {column: datatype 
                                for column, datatype in self.base_dataset.datatypes().items() 
                                if datatype in DuplicateTestingDataset.SUPPORTED_DATATYPES}

        num_duplicates = int(len(self.base_dataset) * duplicate_rate)
        self.duplicate_pairs = self.generate_dup_pairs(num_duplicates)

        self.include_subject_id_in_data = base_dataset.include_subject_id_in_data
        self.show_id = base_dataset.show_id
        self.index_df = base_dataset.index_df

        super().__init__(sample_ids=range(len(self)), subject_ids=range(len(self)))
    
    def generate_dup_pairs(self, num_duplicates): 
        pairs = {}

        while len(pairs) < num_duplicates:
            pair = np.random.choice(range(len(self.base_dataset)), 2, replace=False) 
            # Avoid generating both (1, 2) and (2, 1) 
            pair = tuple(sorted(pair)) 
            if self.duplication_breadth == "column": 
                column = np.random.choice(list(self.relevant_datatypes.keys()))
                pair = tuple((*pair, column))
            if pair not in pairs: 
                pairs[pair[0]] = (pair) 
        
        return pairs

    def perturb_datapoint(self, datapoint, max_perturbation_amount=0.001, pert_column=None): 
        datapoint = copy.deepcopy(datapoint)

        def perturb_continuous(val): 
            return val * (1 + np.random.uniform(-max_perturbation_amount, max_perturbation_amount))
        def perturb_non_continuous(val): 
            rand_noise = (torch.rand(val.shape) - 0.5) * 2 * max_perturbation_amount
            return FloatTensor(val) + rand_noise

        pert_fn_dict = {
            DataType.CONTINUOUS: perturb_continuous, 
            DataType.MULTIDIMENSIONAL: perturb_non_continuous, 
            DataType.IMAGE: perturb_non_continuous, 
        }

        if pert_column:
            columns_to_perturb = [pert_column]
        else: 
            columns_to_perturb = list(self.relevant_datatypes.keys())

        for column in columns_to_perturb: 
            datatype = self.relevant_datatypes[column]
            pert_fn = pert_fn_dict[datatype]
            datapoint[column] = pert_fn(datapoint[column])

        return datapoint

    def __len__(self): 
        return self.base_dataset.__len__() 

    def __getitem__(self, idx): 
        if idx >= self.__len__():
            raise Exception(f"Datapoint {idx} does not exist.") 

        if idx not in set(self.duplicate_pairs.keys()):
            return self.base_dataset[idx]
        
        if self.duplicate_type == "exact":
            # todo: maybe throw an exception if duplication 
            if self.duplication_breadth == "sample": 
                dest, source = self.duplicate_pairs[idx]
                dest_datapoint = self[source]
                return dest_datapoint
            elif self.duplication_breadth == "column":
                dest, source, col = self.duplicate_pairs[idx]
                source_datapoint = self[source]
                dest_datapoint = self.base_dataset[dest]
                dest_datapoint[col] = source_datapoint[col]
                return dest_datapoint
        elif self.duplicate_type == "near": 
            # todo: maybe throw an exception if duplication 
            if self.duplication_breadth == "sample": 
                dest, source = self.duplicate_pairs[idx]
                source_datapoint = self[source]
                dest_datapoint = self.perturb_datapoint(source_datapoint)
                return dest_datapoint
            elif self.duplication_breadth == "column":
                dest, source, col = self.duplicate_pairs[idx]
                source_datapoint = self[source]
                source_datapoint = self.perturb_datapoint(source_datapoint, pert_column=col)
                dest_datapoint = self.base_dataset[dest]
                dest_datapoint[col] = source_datapoint[col]
                return dest_datapoint

    def datatypes(self): 
        return self.base_dataset.datatypes()

class TestNearDuplicateValidatorMethod:
    def check_for_column_duplicates(self, dataset):
        """
        Returns a list of (col, idx_group)
        """
        hashes = {
            col: defaultdict(lambda: [])
            for col in list(self.dataset.datatypes().keys())
        }
        for idx in len(dataset):
            sample = dataset[idx]
            for col in sample.keys():
                sample_col_hash = joblib.hash(sample[col])
                hashes[col][sample_col_hash].append(idx)

        dups = []
        for col, hash_dict in hashes.items():
            for hash_, idx_group in hash_dict.items():
                if len(idx_group) > 1:
                    dups.append(tuple((col, idx_group)))

        return dups

    def check_for_sample_duplicates(self, dataset):
        hashes = defaultdict(lambda: [])
        for idx in len(dataset):
            sample = dataset[idx]
            sample_hash = joblib.hash(sample)
            hashes[sample_hash].append(idx)

        return list({hash_: idx_group for hash_, idx_group in hashes.values() if len(idx_group) > 1}.values())

    def test_with_near_sample_duplicates(self):
        np.random.seed(42)
        torch.manual_seed(42)

        test_validation_schema: Dict = {
            "validators": {
                "duplicate_data_validator": {
                    "validator_kwargs": {
                        "angle_threshold": 19.1,
                        # "angle_threshold": 0.0001,
                    }
                }
            },
            "transforms": {
                "IMAGE": [{
                    "name": "resnet50",
                    "in_place": "False",
                    "options": {},
                }],
            }        
        }

        # exact, sample duplicates
        duplicate_dataset = DuplicateTestingDataset(
            TutorialDataset(
                normal_vec_size=100,
                root='./data/MNIST',
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor() 
                ])
            ),
            duplicate_type="near", 
            duplication_breadth="sample",
        )

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": duplicate_dataset
        }

        dup_results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)['duplicate_data_validator']

        def dfs(node, graph, visited, connected_nodes):
            visited.add(node)
            connected_nodes.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, graph, visited, connected_nodes)

        def find_connected_nodes(edges):
            graph = {}
            for edge in edges:
                node1, node2 = edge
                if node1 not in graph:
                    graph[node1] = []
                if node2 not in graph:
                    graph[node2] = []
                graph[node1].append(node2)
                graph[node2].append(node1)

            visited = set()
            connected_sets = []
            for node in graph:
                if node not in visited:
                    connected_nodes = set()
                    dfs(node, graph, visited, connected_nodes)
                    connected_sets.append(connected_nodes)

            return connected_sets

        # we expect the gt_dup_sets to be the same as the extracted sets
        list_of_dups = list(duplicate_dataset.duplicate_pairs.values())
        method_dup_sets = dup_results['duplicate_sample_validator_method']['results']
        assert method_dup_sets == []

        # we _also_ expect the same sets for all multidimensional columns. 
        method_dup_sets = dup_results['duplicate_high_dimensional_validator_method']
        for col in duplicate_dataset.relevant_datatypes.keys():
            method_dup_col_sets = method_dup_sets[col]
            assert method_dup_col_sets == []

        gt_dup_sets = find_connected_nodes(list_of_dups)
        method_dup_sets = find_connected_nodes(list(dup_results['near_duplicate_sample_validator_method'].values())[0])

        errors = []

        for gt_set in gt_dup_sets: 
            if gt_set not in method_dup_sets:
                errors.append(gt_set)
                print(f"{gt_set} in gt_dup_sets but not in method_dup_sets.")
        for dup_set in method_dup_sets: 
            if dup_set not in gt_dup_sets:
                errors.append(dup_set)
                print(f"{dup_set} in method_dup_sets but not in gt_dup_sets.")

        assert len(errors) * 2 / (len(method_dup_sets) + len(gt_dup_sets)) < 0.05

    def test_with_near_column_duplicates(self):
        np.random.seed(42)
        torch.manual_seed(42)

        test_validation_schema: Dict = {
            "validators": {
                "duplicate_data_validator": {
                    "validator_kwargs": {
                        "angle_threshold": 0.5,
                    }
                }
            },
            "transforms": {
                "IMAGE": [{
                    "name": "resnet50",
                    "in_place": "False",
                    "options": {},
                }],
            }        
        }

        # exact, sample duplicates
        duplicate_dataset = DuplicateTestingDataset(
            TutorialDataset(
                normal_vec_size=100,
                root='./data/MNIST',
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor() 
                ])
            ),
            duplicate_type="near", 
            duplication_breadth="column",
        )

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": duplicate_dataset
        }

        dup_results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)['duplicate_data_validator']

        def dfs(node, graph, visited, connected_nodes):
            visited.add(node)
            connected_nodes.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, graph, visited, connected_nodes)

        def find_connected_nodes(edges):
            graph = {}
            for edge in edges:
                node1, node2 = edge
                if node1 not in graph:
                    graph[node1] = []
                if node2 not in graph:
                    graph[node2] = []
                graph[node1].append(node2)
                graph[node2].append(node1)

            visited = set()
            connected_sets = []
            for node in graph:
                if node not in visited:
                    connected_nodes = set()
                    dfs(node, graph, visited, connected_nodes)
                    connected_sets.append(connected_nodes)

            return connected_sets

        # we expect the gt_dup_sets to be the same as the extracted sets
        list_of_dups = list(duplicate_dataset.duplicate_pairs.values())
        gt_dup_sets = {
            col: [tuple((tup[0], tup[1])) for tup in list_of_dups if tup[2] == col]
            for col in duplicate_dataset.relevant_datatypes.keys()
        }
        method_dup_sets = dup_results['duplicate_sample_validator_method']['results']
        assert method_dup_sets == []

        # we _also_ expect the same sets for all multidimensional columns. 
        method_dup_sets = dup_results['duplicate_high_dimensional_validator_method']
        for col in duplicate_dataset.relevant_datatypes.keys():
            method_dup_col_sets = method_dup_sets[col]
            assert method_dup_col_sets == []

        # and that the same thing works with near_dup
        method_dup_sets = dup_results['near_duplicate_multidimensional_validator_method']['normal_vector']
        method_dup_sets = find_connected_nodes(method_dup_sets)
        gt_dup_sets = find_connected_nodes(gt_dup_sets['normal_vector'])

        for gt_set in gt_dup_sets: 
            if gt_set not in method_dup_sets:
                print(f"{gt_set} in gt_dup_sets but not in method_dup_sets.")
        for dup_set in method_dup_sets: 
            if dup_set not in gt_dup_sets:
                print(f"{dup_set} in method_dup_sets but not in gt_dup_sets.")

        assert {tuple(sorted(s)) for s in gt_dup_sets} == {tuple(sorted(s)) for s in method_dup_sets}


    def test_with_exact_column_duplicates(self):
        np.random.seed(42)
        torch.manual_seed(42)

        test_validation_schema: Dict = {
            "validators": {
                "duplicate_data_validator": {
                    "validator_kwargs": {
                        "angle_threshold": 0.
                    }
                }
            }
        }

        # exact, sample duplicates
        duplicate_dataset = DuplicateTestingDataset(
            TutorialDataset(
                root='./data/MNIST',
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor() 
                ])
            ),
            duplicate_type="exact", 
            duplication_breadth="column",
        )

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": duplicate_dataset
        }

        dup_results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)['duplicate_data_validator']

        def dfs(node, graph, visited, connected_nodes):
            visited.add(node)
            connected_nodes.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, graph, visited, connected_nodes)

        def find_connected_nodes(edges):
            graph = {}
            for edge in edges:
                node1, node2 = edge
                if node1 not in graph:
                    graph[node1] = []
                if node2 not in graph:
                    graph[node2] = []
                graph[node1].append(node2)
                graph[node2].append(node1)

            visited = set()
            connected_sets = []
            for node in graph:
                if node not in visited:
                    connected_nodes = set()
                    dfs(node, graph, visited, connected_nodes)
                    connected_sets.append(connected_nodes)

            return connected_sets

        # we expect the gt_dup_sets to be the same as the extracted sets
        list_of_dups = list(duplicate_dataset.duplicate_pairs.values())
        gt_dup_sets = {
            col: [tuple((tup[0], tup[1])) for tup in list_of_dups if tup[2] == col]
            for col in duplicate_dataset.relevant_datatypes.keys()
        }
        method_dup_sets = dup_results['duplicate_sample_validator_method']['results']
        assert method_dup_sets == []

        # we _also_ expect the same sets for all multidimensional columns. 
        method_dup_sets = dup_results['duplicate_high_dimensional_validator_method']
        for col in duplicate_dataset.relevant_datatypes.keys():
            gt_dup_col_sets = find_connected_nodes(gt_dup_sets[col])
            method_dup_col_sets = method_dup_sets[col]
            assert {tuple(sorted(s)) for s in gt_dup_col_sets} == {tuple(sorted(s)) for s in method_dup_col_sets}

        # and that the same thing works with near_dup
        method_dup_sets = dup_results['near_duplicate_multidimensional_validator_method']['normal_vector']
        method_dup_sets = find_connected_nodes(method_dup_sets)
        gt_dup_sets = find_connected_nodes(gt_dup_sets['normal_vector'])

        for gt_set in gt_dup_sets: 
            if gt_set not in method_dup_sets:
                print(f"{gt_set} in gt_dup_sets but not in method_dup_sets.")
        for dup_set in method_dup_sets: 
            if dup_set not in gt_dup_sets:
                print(f"{dup_set} in method_dup_sets but not in gt_dup_sets.")

        assert {tuple(sorted(s)) for s in gt_dup_sets} == {tuple(sorted(s)) for s in method_dup_sets}

    def test_with_exact_sample_duplicates(self):
        np.random.seed(42)
        torch.manual_seed(42)

        test_validation_schema: Dict = {
            "validators": {
                "duplicate_data_validator": {
                    "validator_kwargs": {
                        "angle_threshold": 0.
                    }
                }
            }
        }

        # exact, sample duplicates
        duplicate_dataset = DuplicateTestingDataset(
            TutorialDataset(
                root='./data/MNIST',
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor() 
                ])
            ),
            duplicate_type="exact", 
            duplication_breadth="sample",
        )

        #TODO: lists for validation sets and test sets.
        data_object: Dict[str, torch.utils.data.Dataset] = {
            "entire_set": duplicate_dataset
        }

        dup_results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)['duplicate_data_validator']

        def dfs(node, graph, visited, connected_nodes):
            visited.add(node)
            connected_nodes.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, graph, visited, connected_nodes)

        def find_connected_nodes(edges):
            graph = {}
            for edge in edges:
                node1, node2 = edge
                if node1 not in graph:
                    graph[node1] = []
                if node2 not in graph:
                    graph[node2] = []
                graph[node1].append(node2)
                graph[node2].append(node1)

            visited = set()
            connected_sets = []
            for node in graph:
                if node not in visited:
                    connected_nodes = set()
                    dfs(node, graph, visited, connected_nodes)
                    connected_sets.append(connected_nodes)

            return connected_sets

        # we expect the gt_dup_sets to be the same as the extracted sets
        gt_dup_sets = find_connected_nodes(list(duplicate_dataset.duplicate_pairs.values()))
        method_dup_sets = dup_results['duplicate_sample_validator_method']['results']
        assert {tuple(sorted(s)) for s in gt_dup_sets} == {tuple(sorted(s)) for s in method_dup_sets}

        # we _also_ expect the same sets for all multidimensional columns. 
        method_dup_sets = dup_results['duplicate_high_dimensional_validator_method']['mnist_image']
        assert {tuple(sorted(s)) for s in gt_dup_sets} == {tuple(sorted(s)) for s in method_dup_sets}
        method_dup_sets = dup_results['duplicate_high_dimensional_validator_method']['normal_vector']
        assert {tuple(sorted(s)) for s in gt_dup_sets} == {tuple(sorted(s)) for s in method_dup_sets}

        # and that the same thing works with near_dup
        method_dup_sets = dup_results['near_duplicate_multidimensional_validator_method']['normal_vector']
        method_dup_sets = find_connected_nodes(method_dup_sets)
        for gt_set in gt_dup_sets: 
            if gt_set not in method_dup_sets:
                print(f"{gt_set} in gt_dup_sets but not in method_dup_sets.")
        for dup_set in method_dup_sets: 
            if dup_set not in gt_dup_sets:
                print(f"{dup_set} in method_dup_sets but not in gt_dup_sets.")
        assert {tuple(sorted(s)) for s in gt_dup_sets} == {tuple(sorted(s)) for s in method_dup_sets}

    def test_with_no_duplicates(self):
        np.random.seed(42)

        test_validation_schema: Dict = {
            "validators": {
                "duplicate_data_validator": {
                    "validator_kwargs": {
                        "angle_threshold": 0.
                    }
                }
            }
        }

        # no duplicates
        duplicate_dataset = TutorialDataset(
            root='./data/MNIST',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor() 
            ])
        )

        data_object: Dict[str, DataDetectiveDataset] = {
            "entire_set": duplicate_dataset
        }

        dup_results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)['duplicate_data_validator']
        assert dup_results['duplicate_high_dimensional_validator_method']['mnist_image'] == []
        assert dup_results['duplicate_high_dimensional_validator_method']['normal_vector'] == []
        assert dup_results['duplicate_sample_validator_method']['results'] == []

        assert len(dup_results['near_duplicate_multidimensional_validator_method']['normal_vector']) == 0
        assert len(dup_results['near_duplicate_sample_validator_method']['results']) == 0