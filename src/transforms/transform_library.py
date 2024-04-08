import numpy as np
import torch

from src.enums.enums import DataType
from src.transforms.embedding_transformer import Transform


def get_resnet50(**kwargs):
    import torchvision.models

    kwargs.pop("data_object")
    kwargs.pop("column")

    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2, **kwargs
    )
    modules = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(torch.nn.Upsample((224, 224)), *modules)

    def full_impl(x):
        if len(x.shape) == 2:
            # add channel dimension
            x = torch.unsqueeze(x, 0)
        if len(x.shape) == 3:
            # need a 4th dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[1] == 1:
            # if 1ch need from 1ch to 3ch RGB
            x = x.expand(x.shape[0], 3, *x.shape[2:])
        x = backbone(x)
        x = x.squeeze()
        x = x.reshape((-1, 2048))
        x = x.detach().numpy()
        return x

    return full_impl


# you have to do these by hand now because of the multiprocessing.
def resnet50_backbone_name(original_name):
    return f"resnet50_backbone_{original_name}"


def get_vit(**kwargs):
    from transformers import AutoImageProcessor, ViTModel

    kwargs.pop("data_object")
    kwargs.pop("column")

    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def full_impl(x):
        if len(x.shape) == 2:
            # add channel dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[0] == 1:
            # if 1ch need from 1ch to 3ch RGB
            x = x.expand(3, *x.shape[1:])

        inputs = image_processor(x, return_tensors="pt")
        outputs = model(**inputs)
        outputs = outputs.pooler_output.detach().numpy()
        return outputs

    return full_impl


def vit_backbone_name(original_name):
    return f"vit_backbone_{original_name}"


def get_bert(**kwargs):
    from transformers import AutoTokenizer, BertModel

    kwargs.pop("data_object")
    kwargs.pop("column")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    def full_impl(x):
        inputs = tokenizer(x, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state

    return full_impl


def bert_backbone_name(original_name):
    return f"bert_backbone_{original_name}"


def get_histogram(**kwargs):
    num_bins = kwargs.get("bins", 10)

    def full_impl(x):
        x_norm = (x - x.min()) / (x.max() - x.min())
        return np.histogram(x_norm, bins=num_bins)

    return full_impl


def histogram_name(original_name):
    return f"histogram_{original_name}"


def get_unit_norm(**kwargs):
    data_object = kwargs.get("data_object", None)
    if data_object is None: 
        raise Exception("Data object cannot be none in unit norm transform.")
    
    dataset_to_use = kwargs.get("dataset_to_use", "entire_set") 
    use_dataset_statistics = kwargs.get("use_dataset_statistics", True)

    dataset_to_use = data_object[dataset_to_use]
    column = kwargs.get("column")

    if column is None: 
        raise Exception("column cannot be none")

    if use_dataset_statistics:
        min_val, max_val = np.inf, -np.inf
        for i in range(len(dataset_to_use)): 
            datapoint = dataset_to_use[i][column]
            min_val = min(datapoint, min_val)
            max_val = max(datapoint, max_val)
    
    def full_impl(x): 
        if use_dataset_statistics:
            x = (x - min_val) / (max_val - min_val)
        else: 
            x = (x - x.min()) / (x.max() - x.min())

        return x

    return full_impl

def unit_norm_name(original_name):
    return f"{original_name}_unit_normed"


TRANSFORM_LIBRARY = {
    "histogram": Transform(
        transform_class=get_histogram,
        new_column_name_fn=histogram_name,
        new_column_datatype=DataType.MULTIDIMENSIONAL,
    ),
    "resnet50": Transform(
        transform_class=get_resnet50,
        new_column_name_fn=resnet50_backbone_name,
        new_column_datatype=DataType.MULTIDIMENSIONAL,
    ),
    "ViT": Transform(
        transform_class=get_vit,
        new_column_name_fn=vit_backbone_name,
        new_column_datatype=DataType.MULTIDIMENSIONAL,
    ),
    "BERT": Transform(
        transform_class=get_bert,
        new_column_name_fn=bert_backbone_name,
        new_column_datatype=DataType.MULTIDIMENSIONAL,
    ),
    "unit_norm": Transform(
        transform_class=get_unit_norm, 
        new_column_name_fn=unit_norm_name,
        new_column_datatype=DataType.CONTINUOUS,
    )
}
