import torchvision.transforms as transforms

from constants import FloatTensor
from src.datasets.my_fashion_mnist import MyFashionMNIST
from src.enums.enums import DataType


class TestMyFashionMNIST:
    def test_error_free_construction(self):
        #TODO: add proper datasets augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./datasets/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

    def test_length(self):
        # TODO: add proper datasets augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./datasets/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        assert(len(fashion_mnist) == 60000)

    def test_getitem(self):
        # TODO: add proper datasets augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./datasets/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        sample = fashion_mnist[0]
        assert(isinstance(sample['fashion_mnist_image'], FloatTensor))
        assert(isinstance(sample['label'], int))

    def test_datatypes(self):
        # TODO: add proper datasets augmentation strategy
        fashion_mnist: MyFashionMNIST = MyFashionMNIST(
            root='./datasets/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        datatypes = fashion_mnist.datatypes()
        assert(datatypes['fashion_mnist_image'] == DataType.IMAGE)
        assert(datatypes['label'] == DataType.CATEGORICAL)


