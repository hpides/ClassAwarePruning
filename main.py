import torch
import os
from torch import nn
from data_loader import get_CIFAR10_dataloaders
from helpers import (
    train_model,
    evaluate_model,
    get_names_of_conv_layers,
    get_parameter_ratio,
)
from torchvision import models
from pruner import StructuredPruner
from ocap import Compute_layer_mask


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    batch_size = 256

    train_loader, test_loader = get_CIFAR10_dataloaders(
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        use_data_Augmentation=True,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
    )

    if os.path.exists("best_model_epoch_18.pth"):
        weights = torch.load("best_model_epoch_18.pth", weights_only=True)
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 10)
        model.load_state_dict(weights)
        model.to(device)
    else:
        model = models.vgg16(pretrained=True)  # Load a pre-trained VGG16 model
        model.classifier[6] = nn.Linear(4096, 10)  # Modify the last layer for CIFAR-10
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 20

        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
        )

    subset_data_loader = get_CIFAR10_dataloaders(
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        use_data_Augmentation=False,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
        selected_classes=[0, 1],
    )

    layer_masks, _ = Compute_layer_mask(
        imgs_dataloader=subset_data_loader,
        model=model,
        percent=0.05,
        device=device,
        activation_func=nn.ReLU(),
    )

    names_of_conv_layers = get_names_of_conv_layers(model)
    # Skip the first layer (input layer)
    layer_masks = layer_masks[1:]
    names_of_conv_layers = names_of_conv_layers[1:]
    masks = {name: layer_masks[i] for i, name in enumerate(names_of_conv_layers)}

    pruner = StructuredPruner(model, masks)
    pruned_model = pruner.prune()
    torch.save(pruned_model.state_dict(), "pruned_model.pth")
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
