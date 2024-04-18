import re
from typing import Dict, Union
import pandas as pd

from PIL import Image
import torch
from torchvision import transforms


from constants import FloatTensor
from src.datasets.data_detective_dataset import DataDetectiveDataset, LambdaDictWrapper
from src.enums.enums import DataType


class PMCOADataset(DataDetectiveDataset):
    def __init__(self, embedding_file="/Users/mcconnl3/Code/data-detective-load-test/data/embeddings/pmc_oa_embed_6000_lit.csv"): 
        self.df = pd.read_csv(embedding_file)

        super().__init__(
            show_id=False, 
            include_subject_id_in_data=False,
            sample_ids = ["pmcoa_" + str(s) for s in list(range(self.__len__()))],
            subject_ids = ["pmcoa_" + str(s) for s in list(range(self.__len__()))]
        )

    def __getitem__(self, idx: Union[int, slice, list]) -> Dict[str, Union[FloatTensor, int]]:
        """
        Returns an item from the dataset.
        @param idx: the dataset index. Only accepts integer indices.
        @return: A dictionary consisting of the image and the label.
        """
        row = self.df.iloc[idx]
        def load_image(): 
            im_path = row.im_path
            image = Image.open(im_path)

            # Transform it to a tensor
            transform = transforms.Compose([
                transforms.ToTensor()  # Converts the image to torch.Tensor and scales the values to [0, 1]
            ])

            # Apply the transform to the image
            tensor_image = transform(image)
            return tensor_image

        datapoint = LambdaDictWrapper({
            "pmcoa_image": load_image,
            "caption": row.caption,
            "clip_alignment_score": row.clip_alignment_score,
            "im_clip_embedding": torch.FloatTensor([getattr(row, f"im_clip_embed_{k}") for k in range(768)]),
            "text_clip_embedding": torch.FloatTensor([getattr(row, f"text_clip_embed_{k}") for k in range(768)]),
        })

        return datapoint

    def __len__(self): 
        return len(self.df) 

    def datatypes(self) -> Dict[str, DataType]:
        """
        Gives the datatypes of a dataset sample.
        @return: the datatypes of a dataset sample.
        """
        return {
            "pmcoa_image": DataType.IMAGE,
            "caption": DataType.TEXT,
            "clip_alignment_score": DataType.CONTINUOUS,
            "im_clip_embed": DataType.MULTIDIMENSIONAL,
            "text_clip_embed": DataType.MULTIDIMENSIONAL, 
        }


    def get_matrix(self, column_wise=True, columns=None): 
        if columns is None: 
            columns = ["im_clip_embed", "text_clip_embed"]

        print("in correct get_matrix")
        matrix_df = self.df.loc[self.index_df['data_idx']]

        # Regular expression pattern for column names followed by an underscore and an integer
        pattern = re.compile(r'^(?:{}|\w+_\d+)$'.format('|'.join(columns)))

        # Filter DataFrame columns using regular expression pattern
        matrix_df = matrix_df.filter(regex=pattern)

        if column_wise: 
            matrix_dict = {}

            for column in columns: 
                columns = [df_col for df_col in matrix_df.columns if df_col[:6] == column[:6]]
                matrix_dict[column] = matrix_df[columns].values
                print(matrix_dict[column].shape)
            
            return matrix_dict
        else: 
            columns = [col for col in matrix_df.columns if "embed" in col]
            matrix = matrix_df[columns].values
            return matrix
        