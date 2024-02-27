import os
import sys
import time
import torch

from typing import Dict
from src.datasets.data_detective_dataset import dd_random_split
from src.datasets.synthetic_normal_dataset_for_ids import SyntheticNormalDatasetForIds, SyntheticNormalDatasetForIdsWithSampleIds

from src.data_detective_engine import DataDetectiveEngine
from src.enums.enums import DataType

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


        # this is how it should work 
        # with sample id
        # synth_normal.__getitem__(sample_id) => item
        # synth_normal.__getitem__(data_id) => error

        # without sample id
        # synth_normal.__getitem__(data_id) => item
        # synth_normal.__getitem__(sample_id) => error (for now, eventually extend) 
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
        synth_normal_subject.datatypes()
        synth_normal_no_subject.datatypes()
        c=3


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



