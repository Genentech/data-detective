import pytest

import src.data.synthetic_data_generators as synthetic_data_generators

import torchvision.transforms as transforms

from constants import FloatTensor
from src.enums.enums import DataType

@pytest.fixture
def cifar_10():
    MyCIFAR10 = synthetic_data_generators.MyCIFAR10

    # TODO: add proper data augmentation strategy
    cifar_10: MyCIFAR10 = MyCIFAR10(
        root='./data/CIFAR10',
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
        assert(isinstance(sample['image'], FloatTensor))
        assert(isinstance(sample['label'], int))

    def test_datatypes(self, cifar_10):
        datatypes = cifar_10.datatypes()
        assert(datatypes['image'] == DataType.IMAGE)
        assert(datatypes['label'] == DataType.CATEGORICAL)