import os
import sys
import time

from typing import Dict, Union

from src.data_detective_engine import DataDetectiveEngine
from src.datasets.csv_dataset import CSVDataset
from src.datasets.data_detective_dataset import dd_random_split, DataDetectiveDataset
from src.enums.enums import DataType

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

class TestCSVDataset:
    def test_csv_dataset_openbhb(self):
        dataset = CSVDataset(
            filepath="openbhb_image_paths_tiny.csv",
            datatype_dict={ "openbhb_image": DataType.IMAGE, },
        )

        # hack for setting length to 100 for testing
        # desired_length = 100
        # dataset, _ = dd_random_split(dataset, [desired_length, dataset.__len__() - desired_length])

        inference_size: int = 20
        everything_but_inference_size: int = dataset.__len__() - inference_size
        inference_dataset, everything_but_inference_dataset = dd_random_split(dataset, [inference_size,
                                                                                                      dataset.__len__() - inference_size])

        train_size: int = int(0.6 * len(everything_but_inference_dataset))
        val_size: int = int(0.2 * len(everything_but_inference_dataset))
        test_size: int = len(everything_but_inference_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = dd_random_split(everything_but_inference_dataset,
                                                                                 [train_size, val_size, test_size])

        data_object: Dict[str, Union[Dict, DataDetectiveDataset]] = {
            "standard_split":  {
                "training_set": train_dataset,
                "validation_set": val_dataset,
                "test_set": test_dataset,
            },
            "entire_set": dataset,
            "everything_but_inference_set": everything_but_inference_dataset,
            "inference_set": inference_dataset
        }

        print(f"size of inference_dataset: {inference_dataset.__len__()}")
        print(f"size of everything_but_inference_dataset: {everything_but_inference_dataset.__len__()}")
        print(f"size of train_dataset: {train_dataset.__len__()}")
        print(f"size of entire dataset: {dataset.__len__()}")
        print(f"size of val_dataset: {val_dataset.__len__()}")
        print(f"size of test_dataset: {test_dataset.__len__()}")

        validation_schema: Dict = {
            "default_inclusion": False,
            "validators": {
                "unsupervised_anomaly_data_validator": {},
                "unsupervised_multimodal_anomaly_data_validator": {},
                "split_covariate_data_validator": {},
                "ood_inference_data_validator": {}
            }
        }

        transform_schema: Dict = {
            "transforms": {
                "openbhb_image": [{
                    "name": "resnet50",
                    "in_place": "False",
                    "options": {},
                }],
            }
        }

        full_validation_schema: Dict = {
            **validation_schema,
            **transform_schema
        }

        data_detective_engine = DataDetectiveEngine()

        # 1 thread, --- 220.85648322105408 seconds ---
        # multithreadinng (joblib), --- 149.11400604248047 seconds ---
        # thread pools, --- 81.38025784492493 seconds ---
        # data-level caching, clean cache, --- 75.22503590583801 seconds ---
        # sample-level caching, clean cache--- 26.184876918792725 seconds ---
        # data-level caching, dirty cache, --- 22.925609827041626 seconds ---
        # sample-level caching, dirty cache, --- 19.73765206336975 seconds ---

        start_time = time.time()
        results = data_detective_engine.validate_from_schema(full_validation_schema, data_object)
        print("--- %s seconds ---" % (time.time() - start_time))

        return results