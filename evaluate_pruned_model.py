import wandb
import torch
import torch.nn as nn
from data_loader import Imagenet_Dataloader_Factory
from models import get_model
from metrics import measure_inference_time_and_accuracy


def replace_module(model, module_name, new_module):
        parts = module_name.split(".")
        parent = model
        for name in parts[:-1]:
            parent = getattr(parent, name)
        setattr(parent, parts[-1], new_module)
        return model

def adjust_conv_channels(conv: nn.Conv2d, in_channels, out_channels):
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        
        return new_conv

def adjust_linear_channels(linear: nn.Linear, in_features, out_features):
        new_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=(linear.bias is not None),
        )
        return new_linear

def adjust_model_for_pruned_weights(model, weights):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            output_channels, input_channels, _, _ = weights[name + ".weight"].shape
            new_module = adjust_conv_channels(module, input_channels, output_channels)
            model = replace_module(model, name, new_module)
        elif isinstance(module, nn.Linear):
            output_channels, input_channels = weights[name + ".weight"].shape
            new_module = adjust_linear_channels(module, input_channels, output_channels)
            model = replace_module(model, name, new_module)
        else:
            continue
        


    return model

def measure():

    config = {
        "batch_size": 4,
        "num_classes": 1000,
        "selected_classes": [0,1,2],
        "dataset.name": "imagenet",
        "model.name": "vgg16",
        "pruning.name": "ocap",
    }

    wandb.init(
            project="ClassAwarePruning",
            entity="smilla-fox",
            config=config,
    )
     
    dataloader_factory = Imagenet_Dataloader_Factory(
        train_batch_size=4,
        test_batch_size=4,
        selected_classes=[0,1,2],
        num_pruning_samples=256,
        use_data_augmentation=False,
        use_imagenet_labels=False
    )

    _, subset_test_loader = dataloader_factory.get_subset_dataloaders()
    model = get_model(
        "vgg16", pretrained=True, num_classes=1000, dataset_name="cfg.dataset.name"
    )
    original_model = get_model(
        "vgg16", pretrained=True, num_classes=1000, dataset_name="cfg.dataset.name"
    )

    weights = torch.load(
        f"model_{config["dataset.name"]}_{config["pruning.name"]}_{config["model.name"]}.pt", weights_only=True, map_location="cuda"
    )
    model = adjust_model_for_pruned_weights(model, weights)
    model.load_state_dict(weights)
    accuracy_new, class_accuracies_new, inference_time_new, times_new = measure_inference_time_and_accuracy(
        data_loader=subset_test_loader,
        model=model,
        device=torch.device("cuda"),
        batch_size=config["batch_size"],
        num_classes=config["num_classes"],
        all_classes=True,
        print_results=True,
        selected_classes=config["selected_classes"],
        with_onnx=False
    )
    print("Pruned model results:")
    accuracy_old, class_accuracies_old, inference_time_old, times_old = measure_inference_time_and_accuracy(
        data_loader=subset_test_loader,
        model=original_model,
        device=torch.device("cuda"),
        batch_size=config["batch_size"],
        num_classes=config["num_classes"],
        all_classes=True,
        print_results=True,
        selected_classes=config["selected_classes"],
        with_onnx=False
    )

    inference_time_ratio = (inference_time_old / inference_time_new) if inference_time_new > 0 else 0

    print("Original model results:")
    wandb.log({
        "inference_time_all_after": times_new,
        "inference_time_all_before": times_old,
        "inference_time_batch_before": inference_time_old,
        "inference_time_batch_after": inference_time_new,
        "inference_time_ratio": inference_time_ratio,
        "accuracy_after": accuracy_new,
        "accuracy_before": accuracy_old,
    })
    


if __name__ == "__main__":
    measure()