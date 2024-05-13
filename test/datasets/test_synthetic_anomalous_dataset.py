import pytest
import torch
import torchvision.transforms as transforms

from src.datasets.my_fashion_mnist import MyFashionMNIST
from src.datasets.synthetic_anomalous_dataset import SyntheticAnomalousDataset


@pytest.fixture()
def synthetic_anomalous_dataset():
    # TODO: add proper datasets augmentation strategy
    fashion_mnist: MyFashionMNIST = MyFashionMNIST(
        root='./datasets/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    synthetic_anomalous_dataset = SyntheticAnomalousDataset(fashion_mnist,
                                                            normal_class=0,
                                                            include_all=True,
                                                            desired_normal_class_proportion=0.5)

    yield synthetic_anomalous_dataset


class TestSyntheticAnomalousDataset:
    def test_correct_added_normal_elements_computation(self, synthetic_anomalous_dataset):
        assert synthetic_anomalous_dataset.added_normal_elements == 48000

    def test_out_of_bounds_access(self, synthetic_anomalous_dataset):
        assert synthetic_anomalous_dataset[64000]['label'] == 0

    def test_in_bounds_access(self, synthetic_anomalous_dataset):
        assert 0 <= synthetic_anomalous_dataset[57000]['label'] <= 9
        assert isinstance(synthetic_anomalous_dataset[57000]['fashion_mnist_image'], torch.FloatTensor)

    def test_class_counts(self, synthetic_anomalous_dataset):
        manual_class_counts = {ind: 6000 for ind in range(10)}
        computed_class_counts = synthetic_anomalous_dataset._get_class_counts(synthetic_anomalous_dataset.original_dataset)
        assert manual_class_counts == computed_class_counts

    def test_class_indices(self, synthetic_anomalous_dataset):
        computed_class_indices = synthetic_anomalous_dataset._get_class_indices(synthetic_anomalous_dataset.original_dataset)
        for ind in range(synthetic_anomalous_dataset.original_dataset.__len__()):
            label = synthetic_anomalous_dataset.original_dataset[ind]['label']
            assert ind in computed_class_indices[label]