import os
import random
import wandb
import torch
import torch.nn as nn
from metrics import measure_inference_time_and_accuracy
from data_loader import dataloaderFactorys
from models import get_model
from metrics import get_parameter_ratio

def memory_usage(model, test_loader, device):
    """Measure the memory usage of the model during inference."""
    
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        break  # Only need one batch for memory measurement

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(inputs)
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    model_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    model_bytes = model_bytes / (1024 ** 2)  # Convert to MB
    return peak_memory + model_bytes
  


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

def adjust_bn_channels(bn: nn.BatchNorm2d, num_features):
        new_bn = nn.BatchNorm2d(
            num_features=num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
        )
        return new_bn


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
        elif isinstance(module, nn.BatchNorm2d):
            num_features = weights[name + ".weight"].shape[0]
            new_module = adjust_bn_channels(module, num_features)
            model = replace_module(model, name, new_module)
        else:
            continue
        


    return model

def measure(config):

    wandb.init(
            project="ClassAwarePruning",
            entity="smilla-fox",
            config=config,
            tags=["test2"]
    )

    dataloader_factory = dataloaderFactorys[config["dataset.name"]](
        train_batch_size=config["batch_size"],
        test_batch_size=config["batch_size"],
        selected_classes=config["selected_classes"],
        num_pruning_samples=config["batch_size"],
        use_data_augmentation=False,
        use_imagenet_labels=False
    )

    _, subset_test_loader = dataloader_factory.get_subset_dataloaders()
    model = get_model(
        config["model.name"], pretrained=True, num_classes=config["num_classes"], dataset_name="cfg.dataset.name"
    )
    original_model = get_model(
        config["model.name"], pretrained=True, num_classes=config["num_classes"], dataset_name="cfg.dataset.name"
    )

    file_name = config["file_name"]
    if "lnstructured" in file_name:
        file_name = file_name.replace("lnstructured","ln_structured")
    weights = torch.load(
        f"pruned_weights/{file_name}.pt", weights_only=True, map_location="cuda"
    )
    model = adjust_model_for_pruned_weights(model, weights)
    model.load_state_dict(weights)
    print("Pruned model results:")
    accuracy_new, _, inference_time_new, times_new = measure_inference_time_and_accuracy(
        data_loader=subset_test_loader,
        model=model.to(torch.device("cuda")),
        device=torch.device("cuda"),
        batch_size=config["batch_size"],
        num_classes=config["num_classes"],
        all_classes=True,
        print_results=True,
        selected_classes=config["selected_classes"],
        with_onnx=False
    )

    peak_memory_new = memory_usage(model.to(torch.device("cuda")), subset_test_loader, torch.device("cuda"))
    peak_memory_old = memory_usage(original_model.to(torch.device("cuda")), subset_test_loader, torch.device("cuda"))
    memory_ratio = (peak_memory_new / peak_memory_old) if peak_memory_old > 0 else 0
    parameter_ratio = get_parameter_ratio(original_model,model)
    print(f"Peak Memory Usage - Pruned Model: {peak_memory_new:.2f} MB")
    print(f"Peak Memory Usage - Original Model: {peak_memory_old:.2f} MB")

    wandb.log({
        "inference_time_all_after": times_new,
        "inference_time_batch_after": inference_time_new,
        "accuracy_after": accuracy_new,
        "memory_after": peak_memory_new,
        "memory_before": peak_memory_old,
        "memory_ratio": memory_ratio,
        "pruned_parameters": 1 - parameter_ratio
    })

    wandb.finish()
    


if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description="Evaluate a single pruned model by file name (without .pt).")
        parser.add_argument("file_name", help="pruned weights file name (with or without .pt)")
        args = parser.parse_args()

        file = args.file_name
        if file.endswith(".pt"):
            file = file[:-3]

        # keep naming consistent with previous behavior (convert ln_structured -> lnstructured for parsing)
        if "ln_structured" in file:
            file = file.replace("ln_structured", "lnstructured")

        parts = file.split("_")
        if len(parts) < 5:
            raise ValueError(f"Unexpected file_name format: '{file}'. Expected at least 5 underscore-separated parts.")

        config = {
            "file_name": file,
            "batch_size": 256 if parts[2] == "cifar10" else 64,
            "num_classes": 10 if parts[2] == "cifar10" else 1000,
            "selected_classes": [0, 1, 2],
            "dataset.name": parts[2],
            "model.name": parts[4],
            "pruning.name": parts[3],
            "pruning_step": parts[1],
            "retraining": parts[0],
        }
        print(config)

        measure(config)