import os
import sys
import numpy as np
import torch

from typing import Dict, Union
from src.datasets.adbench_dataset import ADBenchDDDataset
from src.datasets.data_detective_dataset import DataDetectiveDataset, dd_random_split
from src.datasets.synthetic_normal_dataset_for_ids import SyntheticNormalDatasetForIds, SyntheticNormalDatasetForIdsWithSampleIds

from src.data_detective_engine import DataDetectiveEngine
from src.enums.enums import DataType

SEED = 142
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))

sys.path.insert(0, PROJECT_DIR)


class TestDataDetectiveDataset:
    def test_data_detective_dataset_through_synth_normal(self):
        synth_normal = SyntheticNormalDatasetForIds()
        test_dd_idx = 0

        sample = synth_normal[test_dd_idx]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

        id_dict = sample['id']
        assert("subject_id" in id_dict.keys())
        assert("sample_id" in id_dict.keys())
        assert(id_dict['subject_id'] == id_dict['sample_id'])

        data_dict = sample['data']
        assert("feature_0" in data_dict.keys())

        synth_normal = SyntheticNormalDatasetForIds()
        test_dd_idx = synth_normal.get_sample_id(0)

        sample = synth_normal[test_dd_idx]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

        id_dict = sample['id']
        assert("subject_id" in id_dict.keys())
        assert("sample_id" in id_dict.keys())
        assert(id_dict['subject_id'] == id_dict['sample_id'])

        data_dict = sample['data']
        assert("feature_0" in data_dict.keys())

    def test_data_detective_dataset_through_synth_normal_sample(self):
        synth_normal = SyntheticNormalDatasetForIdsWithSampleIds()
        test_sample_id = synth_normal.get_sample_id(0)

        sample = synth_normal[test_sample_id]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

        id_dict = sample['id']
        assert("subject_id" in id_dict.keys())
        assert("sample_id" in id_dict.keys())
        assert(id_dict['subject_id'] == id_dict['sample_id'])

        data_dict = sample['data']
        assert("feature_0" in data_dict.keys())

        sample = synth_normal[0]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

    def test_data_detective_dataset_through_synth_normal_sample_int_key(self):
        synth_normal = SyntheticNormalDatasetForIdsWithSampleIds()
        test_key = 0

        sample = synth_normal[test_key]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

        id_dict = sample['id']
        assert("subject_id" in id_dict.keys())
        assert("sample_id" in id_dict.keys())
        assert(id_dict['subject_id'] == id_dict['sample_id'])

        data_dict = sample['data']
        assert("feature_0" in data_dict.keys())

        sample = synth_normal[0]
        assert("id" in sample.keys())
        assert("data" in sample.keys())


        # this is how it should work 
        # with sample id
        # synth_normal.__getitem__(sample_id) => item
        # synth_normal.__getitem__(data_id) => item

        # without sample id
        # synth_normal.__getitem__(data_id) => item
        # synth_normal.__getitem__(sample_id) => item
            # (in this case we need to be generating sample IDs so that we can reidentify columns.)  
              
        # how do we manage duplicates? 
        # hash(obj || subject_id) and accept that duplicate entries may exist...
        # ^^^ come back to this

        # how do we generate sample IDs for split-only datasets?
            # example: only splits are given, and we need to match points between datasets for removal...
            # in the absense of given sample IDs, we are left guessing which is which, and the presence of a salt would break matching... 
            # so generated sample IDs have to be a hash of the data columns! bummer but i guess it is what it is

        c=3

    def test_data_detective_dataset_splits(self):
        synth_normal = SyntheticNormalDatasetForIds()
        train_size: int = int(0.6 * len(synth_normal))
        val_size: int = int(0.2 * len(synth_normal))
        test_size: int = len(synth_normal) - train_size - val_size

        train, val, test = dd_random_split(synth_normal, [train_size, val_size, test_size])
        
        test_dd_idx = 0
        sample = train[test_dd_idx]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

        id_dict = sample['id']
        assert("subject_id" in id_dict.keys())
        assert("sample_id" in id_dict.keys())
        assert(id_dict['subject_id'] == id_dict['sample_id'])

        data_dict = sample['data']
        assert("feature_0" in data_dict.keys())

    def test_data_detective_dataset_splitting_sample(self):
        synth_normal = SyntheticNormalDatasetForIdsWithSampleIds()
        train_size: int = int(0.6 * len(synth_normal))
        val_size: int = int(0.2 * len(synth_normal))
        test_size: int = len(synth_normal) - train_size - val_size

        train_dataset_0, val_dataset_0, test_dataset_0 = dd_random_split(synth_normal, [train_size, val_size, test_size])
        test_sample_id = train_dataset_0.get_sample_id(0)

        sample = train_dataset_0[test_sample_id]
        assert("id" in sample.keys())
        assert("data" in sample.keys())

        id_dict = sample['id']
        assert("subject_id" in id_dict.keys())
        assert("sample_id" in id_dict.keys())
        assert(id_dict['subject_id'] == id_dict['sample_id'])

        data_dict = sample['data']
        assert("feature_0" in data_dict.keys())

    def test_data_detective_dataset_metaclass_behavior(self):
        synth_normal_subject = SyntheticNormalDatasetForIds(include_subject_id_in_data=True)
        synth_normal_no_subject = SyntheticNormalDatasetForIds(include_subject_id_in_data=False)
        subject_datatypes = synth_normal_subject.datatypes()
        no_subject_datatypes = synth_normal_no_subject.datatypes()

        assert("subject_id" in subject_datatypes.keys())
        assert("subject_id" not in no_subject_datatypes.keys())
        c=3

    def test_backwards_compatibility(self):
        seed = SEED
        finished = False
        INFERENCE_SIZE = 20

        np.random.seed(seed)
        torch.manual_seed(seed)

        npz_files = [
            # "4_breastw.npz",
            "6_cardio.npz",
            # # "16_http.npz",
            # "21_Lymphography.npz",
            # "25_musk.npz",
            # # "31_satimage-2.npz",
            # "38_thyroid.npz",
            # "42_WBC.npz",
            # "43_WDBC.npz",
        ]

        results_for_table = []

        for npz_filename in npz_files:
            print(npz_filename)
            adbench_dataset: ADBenchDDDataset = ADBenchDDDataset(
                # npz_filename="16_http.npz",
                npz_filename=npz_filename,
                input_data_type=DataType.MULTIDIMENSIONAL,
                output_data_type=DataType.CATEGORICAL,
            )

            test_validation_schema : dict = {
                "default_inclusion": False,
                "validators": {
                    "unsupervised_anomaly_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label",
                        # ],
                    },
                    "split_covariate_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label"
                        # ]
                    },
                    "ood_inference_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label"
                        # ]
                    }
                }
            }


            inference_dataset, everything_but_inference_dataset = dd_random_split( adbench_dataset, [INFERENCE_SIZE, adbench_dataset.__len__() - INFERENCE_SIZE])
            true_results = []
            for idx in range(inference_dataset.__len__()):
                sample = inference_dataset[idx]
                true_results.append(sample['label'])
            true_results = np.array(true_results)

            while len(np.unique(true_results)) < 2:
                inference_dataset, everything_but_inference_dataset = dd_random_split(adbench_dataset,
                                                                                                    [INFERENCE_SIZE,
                                                                                                     adbench_dataset.__len__() - INFERENCE_SIZE])
                true_results = []
                for idx in range(inference_dataset.__len__()):
                    sample = inference_dataset[idx]
                    true_results.append(sample['label'])
                true_results = np.array(true_results)

            train_size: int = int(0.6 * len(everything_but_inference_dataset))
            val_size: int = int(0.2 * len(everything_but_inference_dataset))
            test_size: int = len(everything_but_inference_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = dd_random_split( everything_but_inference_dataset, [train_size, val_size, test_size])

            data_object: Dict[str, Union[Dict, DataDetectiveDataset]] = {
                "standard_split": {
                    "training_set": train_dataset,
                    "validation_set": val_dataset,
                    "test_set": test_dataset,
                },
                "entire_set": adbench_dataset,
                "everything_but_inference_set": everything_but_inference_dataset,
                "inference_set": inference_dataset
            }
            
            results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
            assert(len(results.items()) > 0)
            c=3
            # print(results)

    def test_data_detective_dataset_splitting_and_reidentiication(self):
        synth_normal = SyntheticNormalDatasetForIds()

        train_size: int = int(0.6 * len(synth_normal))
        val_size: int = int(0.2 * len(synth_normal))
        test_size: int = len(synth_normal) - train_size - val_size

        train_dataset_0, val_dataset_0, test_dataset_0 = dd_random_split(synth_normal, [train_size, val_size, test_size])
        train_dataset_1, val_dataset_1, test_dataset_1 = dd_random_split(synth_normal, [train_size, val_size, test_size])

        data_object = {
            "train/val/test_0": {
                "training_set": train_dataset_0,
                "validation_set": val_dataset_0,
                "test_set": test_dataset_0,
            },
            "train/val/test_1": {
                "training_set": train_dataset_1,
                "validation_set": val_dataset_1,
                "test_set": test_dataset_1,
            },
        }

    def test_result_identification(self):
        seed = SEED
        finished = False
        INFERENCE_SIZE = 20

        np.random.seed(seed)
        torch.manual_seed(seed)

        npz_files = [
            # "4_breastw.npz",
            "6_cardio.npz",
            # # "16_http.npz",
            # "21_Lymphography.npz",
            # "25_musk.npz",
            # # "31_satimage-2.npz",
            # "38_thyroid.npz",
            # "42_WBC.npz",
            # "43_WDBC.npz",
        ]

        results_for_table = []

        for npz_filename in npz_files:
            print(npz_filename)
            adbench_dataset: ADBenchDDDataset = ADBenchDDDataset(
                # npz_filename="16_http.npz",
                npz_filename=npz_filename,
                input_data_type=DataType.MULTIDIMENSIONAL,
                output_data_type=DataType.CATEGORICAL,
            )

            test_validation_schema : dict = {
                "default_inclusion": False,
                "validators": {
                    "unsupervised_anomaly_data_validator": {
                        # "include": [
                        #     adbench_dataset.input_data_name,
                        #     "label",
                        # ],
                    },
                    # "split_covariate_data_validator": {
                    #     # "include": [
                    #     #     adbench_dataset.input_data_name,
                    #     #     "label"
                    #     # ]
                    # },
                    # "ood_inference_data_validator": {
                    #     # "include": [
                    #     #     adbench_dataset.input_data_name,
                    #     #     "label"
                    #     # ]
                    # }
                }
            }


            inference_dataset, everything_but_inference_dataset = dd_random_split( adbench_dataset, [INFERENCE_SIZE, adbench_dataset.__len__() - INFERENCE_SIZE])
            true_results = []
            for idx in range(inference_dataset.__len__()):
                sample = inference_dataset[idx]
                true_results.append(sample['label'])
            true_results = np.array(true_results)

            while len(np.unique(true_results)) < 2:
                inference_dataset, everything_but_inference_dataset = dd_random_split(adbench_dataset,
                                                                                                    [INFERENCE_SIZE,
                                                                                                     adbench_dataset.__len__() - INFERENCE_SIZE])
                true_results = []
                for idx in range(inference_dataset.__len__()):
                    sample = inference_dataset[idx]
                    true_results.append(sample['label'])
                true_results = np.array(true_results)

            train_size: int = int(0.6 * len(everything_but_inference_dataset))
            val_size: int = int(0.2 * len(everything_but_inference_dataset))
            test_size: int = len(everything_but_inference_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = dd_random_split( everything_but_inference_dataset, [train_size, val_size, test_size])

            data_object: Dict[str, Union[Dict, DataDetectiveDataset]] = {
                "standard_split": {
                    "training_set": train_dataset,
                    "validation_set": val_dataset,
                    "test_set": test_dataset,
                },
                "entire_set": adbench_dataset,
                "everything_but_inference_set": everything_but_inference_dataset,
                "inference_set": inference_dataset
            }
            
            results = DataDetectiveEngine().validate_from_schema(test_validation_schema, data_object)
            assert(len(results.items()) > 0)
            c=3
            # print(results)

        



