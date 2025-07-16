import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from metrics import calculate_model_accuracy


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler=None,
    num_epochs=20,
):
    """Function to train the model."""
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if scheduler:
            scheduler.step()

        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc_train = 100.0 * correct / total

        test_accuracy = calculate_model_accuracy(
            model, device, test_loader, print_results=False, all_classes=False
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc_train:.2f}, Test Accuracy: {test_accuracy:.2f}"
        )

        # Save after 10 epochs and if accuracy improves
        if (epoch + 1) >= 10 and test_accuracy > best_accuracy:
            print(f"Saving model with improved accuracy: {test_accuracy:.2f}%")
            best_accuracy = test_accuracy
            model_path = f"{model_name}_best_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)

    print("Training complete. Best accuracy achieved:", best_accuracy)


def get_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float = 0.0):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_activation_function(name: str):
    """Get the activation function by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_names_of_conv_layers(model: nn.Module):
    """Get the names of all Conv2d layers in the model."""
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer_names.append(name)
    return conv_layer_names


def get_pruning_indices(masks):
    pruning_indices = {}
    for name, mask in masks.items():
        prune_indices = mask.logical_not().nonzero(as_tuple=False).squeeze(1)
        pruning_indices[name] = prune_indices.tolist()
    return pruning_indices


def get_pruning_masks(indices: dict, model: nn.Module):
    """Convert pruning indices to masks."""
    masks = {}
    for name, prune_indices in indices.items():
        layer = dict(model.named_modules())[name]
        mask = torch.ones(layer.out_channels, dtype=torch.bool)
        mask[prune_indices] = False
        masks[name] = mask
    return masks


def plot_accuracies(original_accuracies, pruned_accuracies, model_name):
    # Plot the accuracies of the different classes before and after pruning stacked in the same plot
    plt.figure(figsize=(10, 6))
    x = range(len(original_accuracies))
    plt.bar(x, original_accuracies, width=0.4, label="Original", align="center")
    plt.bar(
        [i + 0.4 for i in x],
        pruned_accuracies,
        width=0.4,
        label="Pruned",
        align="center",
    )
    plt.xlabel("Classes")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Class-wise Accuracy Comparison for {model_name}")
    plt.xticks([i + 0.2 for i in x], [str(i) for i in range(len(original_accuracies))])
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"{model_name}_accuracy_comparison.png")
