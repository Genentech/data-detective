from src.data import synthetic_data_generators
from src.data.synthetic_data_generators import SyntheticCIDataset


class TestSyntheticCIDataset:
    def test_no_construction_errors(self):
        ci_dataset = SyntheticCIDataset(dataset_type='CI')
        c=2