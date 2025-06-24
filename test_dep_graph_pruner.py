import torch
from data_loader import get_CIFAR10_dataloaders
from pruner import DepGraphPruner
from torchvision import models
import torch.nn as nn
from ocap import Compute_layer_mask
from helpers import get_names_of_conv_layers, evaluate_model, get_parameter_ratio


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    batch_size = 256

    # Load CIFAR-10 data
    _, test_loader = get_CIFAR10_dataloaders(
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        use_data_Augmentation=True,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
    )

    # Load model
    weights = torch.load("best_model_epoch_18.pth", weights_only=True)
    model = models.vgg16()
    model.classifier[6] = nn.Linear(4096, 10)
    model.load_state_dict(weights)
    model.to(device)

    # Get pruning indices
    selected_classes = [0, 1, 2]

    subset_data_loader = get_CIFAR10_dataloaders(
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        use_data_Augmentation=False,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
        selected_classes=selected_classes,
    )
    layer_masks, _ = Compute_layer_mask(
        imgs_dataloader=subset_data_loader,
        model=model,
        percent=0.1,
        device=device,
        activation_func=nn.ReLU(),
    )

    names_of_conv_layers = get_names_of_conv_layers(model)
    # Skip the first layer (input layer)
    layer_masks = layer_masks[1:]
    names_of_conv_layers = names_of_conv_layers[1:]
    masks = {name: layer_masks[i] for i, name in enumerate(names_of_conv_layers)}

    pruner = DepGraphPruner(model=model, masks=masks)
    pruned_model = pruner.prune()
    pruned_model.to(device)

    # Evalute the pruned model
    print("Model pruned successfully.")
    print("Before pruning:")
    evaluate_model(model, device, test_loader, print_results=True, all_classes=True)
    print("After pruning:")
    evaluate_model(
        pruned_model, device, test_loader, print_results=True, all_classes=True
    )
    print(f"Parameter ratio after pruning: {get_parameter_ratio(model, pruned_model)}")


if __name__ == "__main__":
    main()
