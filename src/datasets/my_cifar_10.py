from typing import Dict, Union

from torchvision.datasets import CIFAR10

from constants import FloatTensor
from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType


class MyCIFAR10(DataDetectiveDataset):
    def __init__(self, **kwargs): 
        self.cifar = CIFAR10(**kwargs)
        super().__init__(
            show_id=False, 
            include_subject_id_in_data=False,
            sample_ids = [str(s) for s in list(range(self.__len__()))],
            subject_ids = [str(s) for s in list(range(self.__len__()))]
        )

    def __getitem__(self, idx: Union[int, slice, list]) -> Dict[str, Union[FloatTensor, int]]:
        """
        Returns an item from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the image and the label.
        """
        sample = self.cifar.__getitem__(idx)
        return {
            "cifar_image": sample[0],
            "label": sample[1],
        }

    def __len__(self): 
        return self.cifar.__len__()

    def datatypes(self) -> Dict[str, DataType]:
        """
        Gives the datatypes of a dataset sample.
        @return: the datatypes of a dataset sample.
        """
        return {
            "cifar_image": DataType.IMAGE,
            "label": DataType.CATEGORICAL,
        }
