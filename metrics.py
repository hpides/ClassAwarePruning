import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
from statistics import mean


def calculate_model_accuracy(
    model: nn.Module,
    device: str,
    test_loader: DataLoader,
    print_results: bool = True,
    all_classes: bool = False,
    num_classes: int = 10,
    selected_classes: list = None,
):
    """Function to evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    if selected_classes:
        other_classes = set(range(num_classes)) - set(selected_classes)
        selected_classes.extend(list(other_classes))
        selected_classes = torch.tensor(selected_classes).to(device)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = (
                selected_classes[predicted]
                if selected_classes is not None
                else predicted
            )
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

    class_accuracies = {}
    if all_classes and print_results:
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy_i = 100 * class_correct[i] / class_total[i]
                class_accuracies[i] = accuracy_i
                print(f"Accuracy of class {i}: {accuracy_i:.2f}%")
            else:
                print(f"Class {i} has no samples in the test set.")

    return accuracy, class_accuracies


def calculate_accuracy_for_selected_classes(class_accuracies, selected_classes):
    """Calculate accuracy for selected classes."""
    accuracies = [class_accuracies[cls] for cls in selected_classes]
    accuracy = sum(accuracies) / len(selected_classes)
    return accuracy


def get_parameter_ratio(model: nn.Module, pruned_model: nn.Module):
    """Calculate the ratio of parameters in the pruned model to the original model."""
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    return pruned_params / original_params if original_params > 0 else 0


def get_model_size(model):
    """Get the size of the model in MB."""
    name = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
    torch.save(model.state_dict(), name + ".pt")
    size = os.path.getsize(name + ".pt") / 1e6
    os.remove(name + ".pt")
    return size


def measure_inference_time(
    data_loader: DataLoader, model: nn.Module, device: str, batch_size: int
):
    """Measure the inference time of the model."""
    times = []
    # Warmup phase
    C, W, H = data_loader.dataset.__getitem__(0)[0].shape
    warmup_data = torch.rand(batch_size, C, H, W).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(warmup_data)

    for x, y in data_loader:
        x, y = x.to(device), y

        if device.type == "mps":
            start_time = time.time()
            torch.mps.synchronize() 
            _ = model(x)
            torch.mps.synchronize() 
            end_time = time.time()
            times.append((end_time - start_time)*1000)  # Convert to ms
        else:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True
            ) as prof:
                with record_function("model_inference"):
                    _ = model(x)
            for event in prof.key_averages():
                if event.key == "model_inference":
                    if device.type == "cuda":
                        times.append(event.device_time_total / 1000)  # Convert to ms
                    else:
                        times.append(event.cpu_time_total / 1000)  # Convert to ms
    return mean(times) if times else 0


