import time
from typing import Union, Dict
import torch

import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST

from constants import FloatTensor
from src.datasets.data_detective_dataset import DataDetectiveDataset
from src.enums.enums import DataType

DATASET_SIZE = 1000

class TutorialDataset(DataDetectiveDataset):
    def __init__(self, **kwargs):
        self.mnist = MNIST(**kwargs)
        dataset_size = DATASET_SIZE
        np.random.seed(42)
        self.normal_column = np.random.normal(size=(dataset_size, 2))
        self.normal_column_2 = np.random.normal(size=dataset_size)

        super().__init__(
            show_id=False, 
            include_subject_id_in_data=False,
            sample_ids = [str(s) for s in list(range(dataset_size))],
            subject_ids = [str(s) for s in list(range(dataset_size))]
        )

    def __getitem__(self, idx: Union[int, slice, list]) -> Dict[str, Union[FloatTensor, int]]:
        """
        Returns a dictionary of the image, vector, and label.
        """
        start = time.time() 
        sample = self.mnist.__getitem__(idx)
        mnist_image = sample[0]
        output_size = (mnist_image.size(1) * 55, mnist_image.size(2) * 55)
        upsampled_tensor = torch.nn.functional.interpolate(mnist_image.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False)

        if idx == 10: 
            mnist_image = 1 - mnist_image
        end = time.time() 
        # print(f"Getting untransformed object {idx}: {1000 * (end - start)} ms")

        return {
            # "mnist_image": mnist_image,
            "mnist_image": upsampled_tensor,
            "normal_vector": self.normal_column[idx][:],
            "normal_vector_2": self.normal_column_2[idx],
            "label": sample[1],
        }

    def __len__(self) -> int:
        return DATASET_SIZE

    def datatypes(self) -> Dict[str, DataType]:
        return {
            "mnist_image": DataType.IMAGE,
            "normal_vector": DataType.MULTIDIMENSIONAL,
            "normal_vector_2": DataType.CONTINUOUS,
            "label": DataType.CATEGORICAL,
        }

    def show_datapoint(self, idx: int):
        """
        Shows data point from tutorial.
        """
        # src: https://stackoverflow.com/questions/31556446/how-to-draw-axis-in-the-middle-of-the-figure
        sample = self[idx]
        print(sample["label"])
        print(sample["mnist_image"].min(), sample['mnist_image'].max())
        plt.imshow(sample["mnist_image"].squeeze())
        plt.show()

        ax = plt.gca()
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_box_aspect(1)
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.scatter(*sample["normal_vector"])
        plt.show()

        ax = plt.gca()
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_box_aspect(1)
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.scatter(sample["normal_vector_2"], 0)
        plt.show()

