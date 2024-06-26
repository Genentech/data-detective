import numpy as np
import torch

from src.enums.enums import DataType
from src.transforms.embedding_transformer import Transform


class Resnet50Transform(Transform): 
    def __init__(self, in_place: bool = False, cache_values: bool = True):        
        super().__init__(
            new_column_datatype=DataType.MULTIDIMENSIONAL,
            in_place=in_place, 
            cache_values=cache_values
        )
    
    def initialize_transform(self, transform_kwargs):
        super().initialize_transform(transform_kwargs=transform_kwargs)
        import torchvision.models

        if "data_object" in transform_kwargs.keys():
            transform_kwargs.pop("data_object")
        if "column" in transform_kwargs.keys():
            transform_kwargs.pop("column")

        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2, **transform_kwargs
        )
        modules = list(resnet.children())[:-1]
        self.backbone = torch.nn.Sequential(torch.nn.Upsample((224, 224)), *modules)

    def transform(self, x): 
        if len(x.shape) == 2:
            # add channel dimension
            x = torch.unsqueeze(x, 0)
        if len(x.shape) == 3:
            # need a 4th dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[1] == 1:
            # if 1ch need from 1ch to 3ch RGB
            x = x.expand(x.shape[0], 3, *x.shape[2:])
        x = self.backbone(x)
        x = x.squeeze()
        x = x.reshape((-1, 2048))
        x = x.detach().numpy()
        return x

    def new_column_name(self, original_name): 
        return f"resnet50_backbone_{original_name}"

class VITTransform(Transform): 
    def __init__(self, in_place: bool = False, cache_values: bool = True):        
        super().__init__(
            new_column_datatype=DataType.MULTIDIMENSIONAL,
            in_place=in_place, 
            cache_values=cache_values
        )
    
    def initialize_transform(self, transform_kwargs):
        super().initialize_transform(transform_kwargs=transform_kwargs)
        from transformers import AutoImageProcessor, ViTModel

        if "data_object" in transform_kwargs.keys():
            transform_kwargs.pop("data_object")
        if "column" in transform_kwargs.keys():
            transform_kwargs.pop("column")

        self.image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def transform(self, x):
        if len(x.shape) == 2:
            # add channel dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[0] == 1:
            # if 1ch need from 1ch to 3ch RGB
            x = x.expand(3, *x.shape[1:])

        inputs = self.image_processor(x, return_tensors="pt")
        outputs = self.model(**inputs)
        outputs = outputs.pooler_output.detach().numpy()
        return outputs

    
    def new_column_name(self, original_name): 
        return f"vit_backbone_{original_name}"

class BERTTransform(Transform): 
    def __init__(self, in_place: bool = False, cache_values: bool = True):        
        super().__init__(
            new_column_datatype=DataType.MULTIDIMENSIONAL,
            in_place=in_place, 
            cache_values=cache_values
        )
    
    def initialize_transform(self, transform_kwargs):
        super().initialize_transform(transform_kwargs=transform_kwargs)
        from transformers import AutoTokenizer, BertModel

        if "data_object" in transform_kwargs.keys():
            transform_kwargs.pop("data_object")
        if "column" in transform_kwargs.keys():
            transform_kwargs.pop("column")


        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def transform(self, x):
        inputs = self.tokenizer(x, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def new_column_name(self, original_name): 
        return f"bert_backbone_{original_name}"

class HistogramTransform(Transform): 
    def __init__(self, in_place: bool = False, cache_values: bool = True):        
        super().__init__(
            new_column_datatype=DataType.MULTIDIMENSIONAL,
            in_place=in_place, 
            cache_values=cache_values
        )
    
    def initialize_transform(self, transform_kwargs):
        super().initialize_transform(transform_kwargs=transform_kwargs)

        if "data_object" in transform_kwargs.keys():
            transform_kwargs.pop("data_object")
        if "column" in transform_kwargs.keys():
            transform_kwargs.pop("column")


        self.num_bins = transform_kwargs.get("bins", 10)

    def transform(self, x):
        x_norm = (x - x.min()) / (x.max() - x.min())
        return np.histogram(x_norm, bins=self.num_bins)

    def new_column_name(self, original_name): 
        return f"histogram_{original_name}"

class ZeroOneTransform(Transform): 
    def __init__(self, in_place: bool = False, cache_values: bool = True):        
        super().__init__(
            new_column_datatype=DataType.CONTINUOUS,
            in_place=in_place, 
            cache_values=cache_values
        )
    
    def initialize_transform(self, transform_kwargs):
        data_object = transform_kwargs.get("data_object", None)
        if data_object is None: 
            raise Exception("Data object cannot be none in unit norm transform.")
        
        dataset_to_use = transform_kwargs.get("dataset_to_use", "entire_set") 
        self.use_dataset_statistics = transform_kwargs.get("use_dataset_statistics", True)

        dataset_to_use = data_object[dataset_to_use]
        column = transform_kwargs.get("column")

        if column is None: 
            raise Exception("column cannot be none")

        if self.use_dataset_statistics:
            self.min_val, self.max_val = np.inf, -np.inf
            for i in range(len(dataset_to_use)): 
                datapoint = dataset_to_use[i][column]
                self.min_val = min(datapoint, self.min_val)
                self.max_val = max(datapoint, self.max_val)
    

    def transform(self, x):
        if self.use_dataset_statistics:
            x = (x - self.min_val) / (self.max_val - self.min_val)
        else: 
            x = (x - x.min()) / (x.max() - x.min())

        return x

    def new_column_name(self, original_name): 
        return f"{original_name}_unit_normed"

class GaussianBlurTransform(Transform): 
    def __init__(self, in_place: bool = False, cache_values: bool = True):        
        super().__init__(
            new_column_datatype=DataType.IMAGE,
            in_place=in_place, 
            cache_values=cache_values
        )
    
    def initialize_transform(self, transform_kwargs):
        from torchvision.transforms import GaussianBlur

        super().initialize_transform(transform_kwargs=transform_kwargs)

        if "data_object" in transform_kwargs.keys():
            transform_kwargs.pop("data_object")
        if "column" in transform_kwargs.keys():
            transform_kwargs.pop("column")

        self.transform_obj = GaussianBlur(**transform_kwargs)

    def transform(self, x):
        return self.transform_obj(x)

    def new_column_name(self, original_name): 
        return f"blurred_{original_name}"

TRANSFORM_LIBRARY = {
    "BERT": BERTTransform,
    "gaussian_blur": GaussianBlurTransform,
    "histogram": HistogramTransform,
    "resnet50": Resnet50Transform,
    "ViT": VITTransform, 
    "zero_one_norm": ZeroOneTransform,
}