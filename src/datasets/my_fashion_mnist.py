import typing
from typing import Dict, Union

from torchvision.datasets import FashionMNIST

from constants import FloatTensor
from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType
class MyFashionMNIST(DataDetectiveDataset):
    def __init__(self, **kwargs): 
        self.fashion_mnist = FashionMNIST(**kwargs)
        super().__init__(
            show_id=False, 
            include_subject_id_in_data=False,
            sample_ids = [str(s) for s in list(range(self.__len__()))],
            subject_ids = [str(s) for s in list(range(self.__len__()))]
        )

    def __getitem__(self, idx: int) -> typing.Dict[str, Union[FloatTensor, int]]:
        """
        Returns an item from the fashionMNIST dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the image and the label.
        """
        sample = self.fashion_mnist.__getitem__(idx)
        return {
            "fashion_mnist_image": sample[0],
            "label": sample[1],
        }

    def __len__(self): 
        return self.fashion_mnist.__len__()

    def datatypes(self) -> typing.Dict[str, DataType]:
        """
        Gives the datatypes of a FashionMNIST dataset sample.
        @return: the datatypes of a FashionMNIST dataset sample.
        """
        return {
            "fashion_mnist_image": DataType.IMAGE,
            "label": DataType.CATEGORICAL,
        }
