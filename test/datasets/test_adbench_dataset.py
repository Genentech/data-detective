import pytest

import src.datasets.synthetic_data_generators as synthetic_data_generators

import torchvision.transforms as transforms

from constants import FloatTensor
from src.enums.enums import DataType

@pytest.fixture
def adbench_speech():
    ADBenchDataset = synthetic_data_generators.ADBenchDataset

    # TODO: add proper datasets augmentation strategy
    adbench_speech: ADBenchDataset = ADBenchDataset(
        npz_filename="36_speech.npz",
        input_data_type=DataType.TIME_SERIES,
        output_data_type=DataType.CONTINUOUS,
    )

    yield adbench_speech

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