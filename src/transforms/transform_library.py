import numpy as np
import torch

from src.enums.enums import DataType
from src.transforms.embedding_transformer import Transform


def get_resnet50(**kwargs):
    import torchvision.models

    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2, **kwargs
    )
    modules = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(torch.nn.Upsample((224, 224)), *modules)

    def full_impl(x):
        if len(x.shape) == 2:
            # add channel dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[0] == 1:
            # if 1ch need from 1ch to 3ch RGB
            x = x.expand(3, *x.shape[1:])
        if len(x.shape) == 3:
            # need a 4th dimension
            x = torch.unsqueeze(x, 0)

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
}
