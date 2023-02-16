from src.datasets import synthetic_data_generators
from src.datasets.synthetic_data_generators import SyntheticCIDataset


class TestSyntheticCIDataset:
    def test_no_construction_errors(self):
        ci_dataset = SyntheticCIDataset(dataset_type='CI')
        c=2