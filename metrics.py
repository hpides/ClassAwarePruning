import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
from statistics import mean
import onnxruntime as ort
import numpy as np
import io


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
        # When the last layer is replaced to only predict the selected classses we 
        # need to map the model's output to the correct classes
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


def export_model_to_onnx(model: nn.Module, input_shape: tuple, device: str):
    """Export the model to ONNX format."""
    dummy_input = torch.randn(1, *input_shape).to(device)
    f = io.BytesIO()
    

    torch.onnx.export(
        model,
        dummy_input,
        f,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    onnx_model = f.getvalue()
    return onnx_model



def measure_inference_time_and_accuracy(
    data_loader: DataLoader, model: nn.Module, device: str, batch_size: int, num_classes: int, all_classes: bool, print_results: bool, selected_classes=None, with_onnx=False,
):
    model.eval()
    correct = 0
    total = 0
    if selected_classes:
        # When the last layer is replaced to only predict the selected classses we 
        # need to map the model's output to the correct classes
        other_classes = set(range(num_classes)) - set(selected_classes)
        selected_classes.extend(list(other_classes))
        selected_classes = torch.tensor(selected_classes).to(device)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    times = []
    C, W, H = data_loader.dataset.__getitem__(0)[0].shape

    if with_onnx:
        model = export_model_to_onnx(model, (C, W, H), device)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(model, providers=["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"], sess_options=session_options)
        model_func = lambda x: session.run(None, {"input": x.cpu().numpy()})[0] 
    else:
        model.to(device)
        model_func = lambda x: model(x)  
   
    warmup_data = torch.randn(batch_size, C, H, W).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model_func(warmup_data)

    for input, labels in data_loader:
        input, labels = input.to(device), labels.to(device)

        if device.type == "mps":
            start_time = time.time()
            torch.mps.synchronize() 
            output = model_func(input)
            torch.mps.synchronize() 
            end_time = time.time()
            times.append((end_time - start_time)*1000)  # Convert to ms
        else:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True
            ) as prof:
                with record_function("model_inference"):
                    output = model_func(input)
            for event in prof.key_averages():
                if event.key == "model_inference":
                    if device.type == "cuda":
                        times.append(event.device_time_total / 1000)  # Convert to ms
                    else:
                        times.append(event.cpu_time_total / 1000)  # Convert to ms
            
        _, predicted = torch.max(output.data, 1)
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
    class_accuracies = {}
    if all_classes and print_results:
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy_i = 100 * class_correct[i] / class_total[i]
                class_accuracies[i] = accuracy_i
                print(f"Accuracy of class {i}: {accuracy_i:.2f}%")  

    inference_time = mean(times) if times else 0         
    return accuracy, class_accuracies, inference_time


def measure_execution_time(selector, model):
    times = []
    for _ in range(1):
        start = time.perf_counter()
        indices = selector.select(model)
        end = time.perf_counter()
        times.append(end-start)

    elapsed_time = mean(times)
    print("time:", elapsed_time)
    return indices, elapsed_time