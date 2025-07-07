import torch.nn as nn
from torchvision import models


def get_model(model_name: str, pretrained: bool, num_classes: int = 10) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): Name of the model.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The model instance.
    """

    models_dict = {
        "vgg16": models.vgg16,
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "mobilenetv2": models.mobilenet_v2,
    }

    if model_name not in models_dict:
        raise ValueError(
            f"Model {model_name} is not supported. Supported models: {list(models_dict.keys())}"
        )

    model = models_dict[model_name](pretrained=pretrained)
    if num_classes != 1000:
        if model_name == "vgg16":
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == "resnet18":
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
