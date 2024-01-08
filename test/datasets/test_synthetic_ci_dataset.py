from src.datasets.synthetic_ci_dataset import SyntheticCIDataset


class TestSyntheticCIDataset:
    def test_no_construction_errors(self):
        ci_dataset = SyntheticCIDataset(dataset_type='CI')