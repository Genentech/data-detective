import time
from typing import Union, Dict
import torch

import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST

from constants import FloatTensor
from src.datasets.data_detective_dataset import DataDetectiveDataset, LambdaDictWrapper
from src.enums.enums import DataType

DATASET_SIZE = 1000

class TutorialDataset(DataDetectiveDataset):
    def __init__(self, normal_vec_size=2, **kwargs):
        self.mnist = MNIST(**kwargs)
        dataset_size = DATASET_SIZE
        np.random.seed(42)
        self.normal_column = np.random.normal(size=(dataset_size, normal_vec_size))
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
        # sample = self.mnist.__getitem__(idx)
        # mnist_image = sample[0]
        # output_size = (mnist_image.size(1) * 55, mnist_image.size(2) * 55)
        # upsampled_tensor = torch.nn.functional.interpolate(mnist_image.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False)

        def get_img():
            # print("unwrapping image")
            sample = self.mnist.__getitem__(idx)
            mnist_image = sample[0]
            output_size = (mnist_image.size(1) * 55, mnist_image.size(2) * 55)
            upsampled_tensor = mnist_image
            # upsampled_tensor = torch.nn.functional.interpolate(mnist_image.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False)

            if idx == 10: 
                upsampled_tensor = 1 - upsampled_tensor

            return upsampled_tensor

        def get_label(): 
            # print("unwrapping label")
            sample = self.mnist.__getitem__(idx)
            return sample[1]

        end = time.time() 
        # print(f"Getting untransformed object {idx}: {1000 * (end - start)} ms")

        return LambdaDictWrapper({
            # "mnist_image": mnist_image,
            "mnist_image": get_img,
            "normal_vector": self.normal_column[idx][:],
            "normal_vector_2": self.normal_column_2[idx],
            "label": get_label,
        })

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
        # print(sample["mnist_image"].min(), sample['mnist_image'].max())
        plt.imshow(sample["mnist_image"].squeeze())
        plt.title(f'MNIST Image (label {sample["label"]})', y=-0.15)  # Title at the bottom        plt.show()
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
        plt.title(f'Normal Vector', y=-0.15)  
        plt.show()

        def plot_number_line(data_point, num_std=3):
            # Generate values based on standard normal distribution within the specified range
            values = np.linspace(-num_std, num_std, 1000)
            
            plt.figure(figsize=(5, 1))  # Adjust the figure size as needed
            # plt.axhline(y=0, color='k')  # Draw x-axis
            plt.plot(values, np.zeros_like(values), color='black')  # Plot number line
            plt.scatter(data_point, 0, color='r', s=50, label='Data Point')  # Plot the data point
            plt.xlabel('Normal Sample')  # Label x-axis
            plt.grid(True)  # Show grid
            plt.ylim(-0.1,0.1)  # Set y-limits for number line
            plt.yticks([])

            plt.show()

        plot_number_line(sample['normal_vector_2'])
