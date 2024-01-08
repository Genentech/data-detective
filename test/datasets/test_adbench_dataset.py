import pytest

import torchvision.transforms as transforms

from constants import FloatTensor
from src.datasets.my_cifar_10 import MyCIFAR10
from src.datasets.adbench_dataset import ADBenchDataset
from src.enums.enums import DataType

@pytest.fixture
def cifar_10():
    # TODO: add proper datasets augmentation strategy
    cifar_10: MyCIFAR10 = MyCIFAR10(
        root='./datasets/CIFAR10',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    yield cifar_10

class TestMyCIFAR10:
    def test_length(self, cifar_10):
        assert(len(cifar_10) == 50000)

    def test_getitem(self, cifar_10):
        sample = cifar_10[0]
        assert(isinstance(sample['cifar_image'], FloatTensor))
        assert(isinstance(sample['label'], int))

    def test_datatypes(self, cifar_10):
        datatypes = cifar_10.datatypes()
        assert(datatypes['cifar_image'] == DataType.IMAGE)
        assert(datatypes['label'] == DataType.CATEGORICAL)