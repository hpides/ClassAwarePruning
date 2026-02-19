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
import statistics


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
        #other_classes = set(range(num_classes)) - set(selected_classes)
        #selected_classes.extend(list(other_classes))
        #print(f"%%%%%% SELECTED CLASSES: {selected_classes}")
        selected_classes = torch.tensor(selected_classes).to(device)
        #print(f"%%%%%% OTHER CLASSES: {other_classes}")
        num_classes = len(selected_classes)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            #print(f"***** RAW PREDICTED: {predicted}, LABEL: {labels}*****")
            #predicted = (
            #    selected_classes[predicted]
            #    if selected_classes is not None
            #    else predicted
            #)
            #print(f"***** MODIFIED PREDICTED: {predicted}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            #print(f"+++++ LABELS: {labels}")
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    #print(f"+++++ CLASS CORRECT: {class_correct}")
    #print(f"+++++ CLASS TOTAL: {class_total}")

    if print_results:
        print(f"Accuracy of the model on the test set: {accuracy:.2f}%")

    class_accuracies = {}
    if all_classes and print_results:
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy_i = 100 * class_correct[i] / class_total[i]
                class_accuracies[i] = accuracy_i
                print(f"%%%%%% Accuracy of class {i}: {accuracy_i:.2f}%")
            
    return accuracy, class_accuracies


def calculate_accuracy_for_selected_classes(class_accuracies, selected_classes):
    """Calculate accuracy for selected classes."""
    # Classes get mapped, hence [2,4,6] would become [0,1,2]
    #print(f"***** Class accuracies: {class_accuracies}")
    accuracies = [class_accuracies[i] for i in range(len(selected_classes))]
    #print(f"%%%%%% ACCURACIES: {accuracies}%")
    accuracy = sum(accuracies) / len(selected_classes)
    #print(f"xxxxx ACCURACY FOR SELECTED CLASSES: {accuracy}%")
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
        data_loader: DataLoader,
        model: nn.Module,
        device: str,
        batch_size: int,
        num_classes: int,
        all_classes: bool,
        print_results: bool,
        selected_classes=None,
        with_onnx=False,
        mapping=None
):
    model.eval()
    correct = 0
    total = 0

    # Handle selected_classes conversion
    if selected_classes:
        #other_classes = set(range(num_classes)) - set(selected_classes)
        #selected_classes.extend(list(other_classes))
        selected_classes = torch.tensor(selected_classes).to(device)
        num_classes = len(selected_classes)

    # Create mapping tensor once before loop
    mapping_tensor = None
    if mapping is not None:
        max_idx = max(mapping.keys()) + 1
        mapping_tensor = torch.zeros(max_idx, dtype=torch.long, device=device)
        for new_idx, orig_class in mapping.items():
            mapping_tensor[new_idx] = orig_class

    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    times = []
    C, W, H = data_loader.dataset.__getitem__(0)[0].shape

    # Setup model function (ONNX or PyTorch)
    if with_onnx:
        model = export_model_to_onnx(model, (C, W, H), device)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model,
            providers=["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"],
            sess_options=session_options
        )
        model_func = lambda x: session.run(None, {"input": x.cpu().numpy()})[0]
    else:
        model.to(device)
        model_func = lambda x: model(x)

        # Warmup
    torch.cuda.empty_cache() if device.type == "cuda" else None
    warmup_data = torch.randn(batch_size, C, H, W).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model_func(warmup_data)

    # Main evaluation loop
    for input, labels in data_loader:
        input, labels = input.to(device), labels.to(device)

        # Time inference
        if device.type == "mps":
            start_time = time.time()
            torch.mps.synchronize()
            output = model_func(input)
            torch.mps.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        else:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    acc_events=True
            ) as prof:
                with record_function("model_inference"):
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    output = model_func(input)
                    torch.cuda.synchronize() if device.type == "cuda" else None

            for event in prof.key_averages():
                if event.key == "model_inference":
                    if device.type == "cuda":
                        times.append(event.device_time_total / 1000)  # Convert to ms
                    else:
                        times.append(event.cpu_time_total / 1000)  # Convert to ms

        # Get predictions
        _, predicted = torch.max(output.data, 1)

        #print(f"xxxxx PREDICTED: {predicted}")

        # Apply selected_classes mapping if needed
        #print(f"xxxxx SELECTED CLASSES: {selected_classes}")
        if selected_classes is not None:
            predicted = selected_classes[predicted]
            #print(f"xxxxx PREDICTED CLASSES: {predicted}")

        # Map labels from [0,1,2] to original classes [4,6,8] if needed
        if mapping_tensor is not None:
            #print(f"xxxxx MAPPING TENSOR: {mapping_tensor}")
            labels_mapped = mapping_tensor[labels]
            #print(f"xxxxx MAPPED LABELS: {labels_mapped}")
        else:
            labels_mapped = labels

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels_mapped).sum().item()
        c = (predicted == labels_mapped).squeeze()
        #print(f"xxxxx CORRECT: {c}")

        # Per-class accuracy tracking
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += c[i].item()
            class_total[label] += 1
        #print(f"xxxxx CLASS TOTAL: {class_total}")
        #print(f"xxxxx CLASS CORRECT: {class_correct}")

    # Calculate final metrics
    accuracy = 100 * correct / total if total > 0 else 0
    class_accuracies = {}

    if all_classes and print_results:
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy_i = 100 * class_correct[i] / class_total[i]
                class_accuracies[i] = accuracy_i
                print(f"xxxxx Accuracy of class {mapping[i] if mapping is not None else i}: {accuracy_i:.2f}%")

    inference_time = mean(times) if times else 0
    return accuracy, class_accuracies, inference_time, times


def measure_execution_time(selector, model): # TODO: adjust for multiple pruning ratios
    start = time.perf_counter()
    indices = selector.select(model)
    elapsed_time = time.perf_counter() - start
    print("time:", elapsed_time)
    return indices, elapsed_time


def measure_inference_time(data_loader: DataLoader, model: nn.Module, device: str, batch_size: int):
    model.eval()
    times = []
    C, W, H = data_loader.dataset.__getitem__(0)[0].shape
    torch.cuda.empty_cache() if device.type == "cuda" else None
    warmup_data = torch.randn(batch_size, C, H, W).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(warmup_data)

    for _ in range(10):
        if device.type == "mps":
            start_time = time.time()
            torch.mps.synchronize() 
            _ = model(warmup_data)
            torch.mps.synchronize() 
            end_time = time.time()
            times.append((end_time - start_time)*1000)  # Convert to ms
        else:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True
            ) as prof:
                with record_function("model_inference"):
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    _ = model(warmup_data)
                    torch.cuda.synchronize() if device.type == "cuda" else None
            for event in prof.key_averages():
                if event.key == "model_inference":
                    if device.type == "cuda":
                        times.append(event.device_time_total / 1000)  # Convert to ms
                    else:
                        times.append(event.cpu_time_total / 1000)  # Convert to ms
            

    inference_time = statistics.mean(times) if times else 0     
    return inference_time, times