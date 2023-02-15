import src.data.synthetic_data_generators as synthetic_data_generators

import torchvision.transforms as transforms

from constants import FloatTensor
from src.enums.enums import DataType


class TestMyFashionMNIST:
    def test_error_free_construction(self):
        MyFashionMNIST = synthetic_data_generators.MyFashionMNIST

        #TODO: add proper data augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

    def test_length(self):
        MyFashionMNIST = synthetic_data_generators.MyFashionMNIST
        # TODO: add proper data augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        assert(len(fashion_mnist) == 60000)

    def test_getitem(self):
        MyFashionMNIST = synthetic_data_generators.MyFashionMNIST
        # TODO: add proper data augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        sample = fashion_mnist[0]
        assert(isinstance(sample['image'], FloatTensor))
        assert(isinstance(sample['label'], int))

    def test_datatypes(self):
        MyFashionMNIST = synthetic_data_generators.MyFashionMNIST
        # TODO: add proper data augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        datatypes = fashion_mnist.datatypes()
        assert(datatypes['image'] == DataType.IMAGE)
        assert(datatypes['label'] == DataType.CATEGORICAL)


