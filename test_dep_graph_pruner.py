import torch
from data_loader import get_CIFAR10_dataloaders
from pruner import DepGraphPruner
from torchvision import models
import torch.nn as nn
from lrp import get_candidates_to_prune
from helpers import get_names_of_conv_layers, evaluate_model, get_parameter_ratio


def main():
    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = "cpu"
    batch_size = 512

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
    selected_classes = [0, 1]

    subset_data_loader = get_CIFAR10_dataloaders(
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        use_data_Augmentation=False,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
        selected_classes=selected_classes,
    )

    number_of_filters_to_prune = 3000
    data_iter = iter(subset_data_loader)
    X, y = next(data_iter)

    indices = get_candidates_to_prune(
        model=model,
        num_filters_to_prune=number_of_filters_to_prune,
        X_test=X.to(device),
        y_test_true=y.to(device),
    )

    pruner = DepGraphPruner(model=model, indices=indices, device=device)
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

    torch.save(pruned_model, "pruned_lrp_model.pth")


if __name__ == "__main__":
    main()
