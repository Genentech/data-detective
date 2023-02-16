from src.enums.enums import DataType
import src.datasets.synthetic_data_generators as synthetic_data_generators


class TestSyntheticNormalDataset:
    def test_dataset_len(self):
        SyntheticNormalDataset = synthetic_data_generators.SyntheticNormalDataset
        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=1, dataset_size = 10000)
        assert(len(normal_dataset) == 10000)

