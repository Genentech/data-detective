import src.data.synthetic_data_generators as synthetic_data_generators

from src.enums.enums import DataType


class TestColumnFilteredDataset:
    def test_column_filtering(self):
        SyntheticNormalDataset = synthetic_data_generators.SyntheticNormalDataset
        ColumnFilteredDataset = synthetic_data_generators.ColumnFilteredDataset

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=100, dataset_size = 10000)
        filtered_dataset: ColumnFilteredDataset = ColumnFilteredDataset(normal_dataset, ["feature_\d{2}"])

        # should have 10 - 99 [90 items]
        assert(len(filtered_dataset.datatypes().items()) == 90)

        for i in range(10,99):
            assert(f"feature_{i}" in filtered_dataset.datatypes().keys())

    def test_get_one_column(self):
        """
        Getting a single column from a dataset.
        """
        SyntheticNormalDataset = synthetic_data_generators.SyntheticNormalDataset
        ColumnFilteredDataset = synthetic_data_generators.ColumnFilteredDataset

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=100, dataset_size = 10000)
        filtered_dataset: ColumnFilteredDataset = ColumnFilteredDataset(normal_dataset, ["^feature_22$"])

        assert(len(filtered_dataset[:].keys()) == 1)

    def test_datatypes_filter(self):
        SyntheticNormalDataset = synthetic_data_generators.SyntheticNormalDataset
        ColumnFilteredDataset = synthetic_data_generators.ColumnFilteredDataset

        normal_dataset: SyntheticNormalDataset = SyntheticNormalDataset(num_cols=100, dataset_size=10000)
        filtered_dataset: ColumnFilteredDataset = ColumnFilteredDataset(normal_dataset, ["feature_\d{2}"])

        assert(len(filtered_dataset.datatypes().items()) == 90)

        count = 10
        for key, value in filtered_dataset.datatypes().items():
            assert(key == f"feature_{count}")
            count += 1
            assert(value.value == DataType.CONTINUOUS.value)
