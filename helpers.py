import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    device: str,
    test_loader: DataLoader,
    print_results: bool = True,
    all_classes: bool = False,
):
    """Function to evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total

    if print_results:
        print(f"Accuracy of the model on the test set: {accuracy:.2f}%")

    if all_classes and print_results:
        for i in range(10):
            if class_total[i] > 0:
                print(
                    f"Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%"
                )
            else:
                print(f"Class {i} has no samples in the test set.")

    return accuracy


def train_model(
    model: nn.Module,
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

        test_accuracy = evaluate_model(
            model, device, test_loader, print_results=False, all_classes=False
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc_train:.2f}, Test Accuracy: {test_accuracy:.2f}"
        )

        # Save after 10 epochs and if accuracy improves
        if (epoch + 1) >= 10 and test_accuracy > best_accuracy:
            print(f"Saving model with improved accuracy: {test_accuracy:.2f}%")
            best_accuracy = test_accuracy
            model_path = f"best_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)

    print("Training complete. Best accuracy achieved:", best_accuracy)


def get_names_of_conv_layers(model: nn.Module):
    """Get the names of all Conv2d layers in the model."""
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer_names.append(name)
    return conv_layer_names


def get_parameter_ratio(model: nn.Module, pruned_model: nn.Module):
    """Calculate the ratio of parameters in the pruned model to the original model."""
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    return pruned_params / original_params if original_params > 0 else 0
