from src.datasets.synthetic_normal_dataset import SyntheticNormalDataset


class TestSyntheticNormalDataset:
    def test_dataset_len(self):
        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size = 10000)
        assert(len(normal_dataset) == 10000)

