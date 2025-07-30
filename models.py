import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, VGG16_Weights, MobileNet_V2_Weights

def replace_module(model, module_name, new_module):
        parts = module_name.split(".")
        parent = model
        for name in parts[:-1]:
            parent = getattr(parent, name)
        setattr(parent, parts[-1], new_module)

def replace_last_layer_for_imagenette(model):
    imagenette_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    name, last_linear = list(model.named_modules())[-1]
    new_linear = nn.Linear(
        in_features=last_linear.in_features,
        out_features=10,
        bias=(last_linear.bias is not None),
    )
    new_linear.weight.data = last_linear.weight.data[imagenette_classes].clone()
    if last_linear.bias is not None:
        new_linear.bias.data = last_linear.bias.data[imagenette_classes].clone()

    replace_module(model, name, new_linear)
    return model

def get_model(model_name: str, pretrained: bool, num_classes: int = 10, dataset_name=None) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): Name of the model.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The model instance.
    """

    models_dict = {
        "vgg16": (models.vgg16, VGG16_Weights.DEFAULT),
        "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT),
        "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT),
        "resnet152": (models.resnet152, ResNet152_Weights.DEFAULT),
        "mobilenetv2": (models.mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
    }

    if model_name not in models_dict:
        raise ValueError(
            f"Model {model_name} is not supported. Supported models: {list(models_dict.keys())}"
        )

    model_cls, weights = models_dict[model_name]
    if pretrained:
        model = model_cls(weights=weights)
    else:
        model = model_cls(weights=None)
    if dataset_name == "imagenette" and num_classes == 10:
        model = replace_last_layer_for_imagenette(model)
    elif num_classes != 1000:
        if model_name == "vgg16":
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
