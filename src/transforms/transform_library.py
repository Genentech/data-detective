import torch
import torchvision.transforms

from src.enums.enums import DataType
from src.transforms.embedding_transformer import Transform
import torchvision.models


def get_resnet50(**kwargs):
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2, **kwargs)
    modules = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(torch.nn.Upsample((224, 224)), *modules)
    def full_impl(x):
        if len(x.shape) == 3:
            # need a 4th dimension
            x = torch.unsqueeze(x, 0)
        if x.shape[-3] == 1:
            # need to map to multiple channels
            x2 = torch.zeros((x.shape[0], 3, x.shape[2], x.shape[3]))
            x2[:, 0, :, :] = x[:, 0, :, :]
            x2[:, 1, :, :] = x[:, 0, :, :]
            x2[:, 2, :, :] = x[:, 0, :, :]
            x = x2

        x = backbone(x)
        x = x.squeeze()
        x = x.reshape((-1, 2048))
        x = x.detach().numpy()

        return x

    return full_impl

TRANSFORM_LIBRARY = {
    "resnet50": Transform(
        transform_class=get_resnet50,
        new_column_name_fn=lambda name: f"resnet50_backbone_{name}",
        new_column_datatype=DataType.MULTIDIMENSIONAL
    ),
}