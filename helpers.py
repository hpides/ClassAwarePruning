import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf, ListConfig
import wandb
from typing import Dict
import time

from metrics import calculate_model_accuracy, measure_inference_time_and_accuracy, calculate_accuracy_for_selected_classes


def run_pruner(pruner, ratio):
    """
    Prune if ratio > 0, otherwise just replace the last layer.

    Returns:
        Tuple: Pruned model and time needed for filter removal.
    """
    start = time.perf_counter()
    if ratio > 0:
        pruned_model = pruner.prune()
        print("%%%%%% Model pruned successfully.")
    else:
        pruner._replace_last_layer()
        pruned_model = pruner.model
    return pruned_model, time.perf_counter() - start


def evaluate(model, loader, cfg, device, inference_time_before, mapping, label):
    """
    Run inference + accuracy measurement.

    Args:

    Returns:
        Tuple: Accuracy, Inference time, Ratio of inference time.
    """
    print(f"\n%%%%%% {'=' * 80}")
    print(f"%%%%%% {label}:")
    print(f"%%%%%% {'=' * 80}\n")
    _, class_accuracies, inference_time, _ = measure_inference_time_and_accuracy(
        loader, model, device,
        cfg.training.batch_size_test,
        cfg.dataset.num_classes,
        all_classes=True,
        print_results=True,
        selected_classes=cfg.selected_classes.copy() if cfg.replace_last_layer else None,
        with_onnx=cfg.inference_with_onnx,
        mapping=mapping
    )
    accuracy = calculate_accuracy_for_selected_classes(class_accuracies, cfg.selected_classes)
    ratio = (inference_time / inference_time_before) if inference_time_before > 0 else 0
    print(f"\n%%%%%% {'=' * 80}")
    print(f"%%%%% STATS FOR {label}:")
    print(f"%%%%% ACCURACY: {accuracy}, INFERENCE: {inference_time:.2f}, INFERENCE TIME RATIO: {ratio}")
    print(f"%%%%%% {'=' * 80}\n")
    return accuracy, inference_time, ratio


def train(
        cfg: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        scheduler=None,
        num_epochs=20,
        log_results=False,
        num_classes=10,
        retrain=False,
        patience=5,
        min_delta=0.1
):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        cfg.training.optimizer, model, cfg.training.lr if not retrain else cfg.training.lr_retrain,
        cfg.training.weight_decay
    )
    model_name = cfg.model.name,

    best_accuracy = 0.0
    best_epoch = 1
    epochs_without_improvement = 0

    print("=" * 60)
    print(f"Starting training for {num_epochs} epochs")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    #print(f"Model architecture:\n{model}")
    print("=" * 60)
    print(f"+++++ GPU memory right before first epoch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            #print(f"%%%%% LABELS: {labels}", flush=True)
            inputs, labels = inputs.to(device), labels.to(device)
            #print(f"+++++ GPU memory after loading inputs and labels to device: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            optimizer.zero_grad()
            outputs = model(inputs)
            #print(f"+++++ GPU memory after calculating outputs: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            loss = criterion(outputs, labels)
            #print(f"+++++ GPU memory after calculating loss: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            loss.backward()
            #print(f"+++++ GPU memory after backwards step: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            optimizer.step()
            #print(f"+++++ GPU memory after optimizer step: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            # Track loss and accuracy
            running_loss += loss.item()
            #print(f"+++++ GPU memory after adding running loss: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if scheduler:
            scheduler.step()

        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc_train = 100.0 * correct / total

        val_accuracy, _ = calculate_model_accuracy(
            model, device, val_loader, print_results=False, all_classes=False, num_classes=num_classes, selected_classes=cfg.selected_classes
        )

        if log_results:
            wandb.log({
                "epoch": epoch,
                "loss": epoch_loss,
                "train_accuracy_epoch": epoch_acc_train,
                "test_accuracy_epoch": val_accuracy,
            })

        print(
            f"%%%%% Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc_train:.2f}, "
            f"Val Accuracy: {val_accuracy:.2f}", flush=True
        )

        # Save after 10 epochs and if accuracy improves
        if val_accuracy > best_accuracy + min_delta:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if (epoch + 1) >= 10: 
                print(f"%%%%% Saving model with improved accuracy: {val_accuracy:.2f}%")
                model_path = f"model_weights/{model_name}_best_model_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), model_path)
        else:
            # No significant improvement
            epochs_without_improvement += 1
            print(f"%%%%% No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"%%%%% Early stopping triggered after {epoch + 1} epochs")
                print(f"%%%%% Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
                break

    print("%%%%% Training complete. Best accuracy achieved:", best_accuracy)
    return best_accuracy, best_epoch


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


def filter_pruning_indices_for_resnet(all_indices: dict, model_name: str):
    if model_name == "resnet152":
        save_layer = "conv3"
    else:
        save_layer = "conv2"
    new_indices = []
    for indices in all_indices:
        keys = list(indices.keys())
        for key in keys:
            if save_layer in key or "downsample" in key:
                indices.pop(key, None)
        new_indices.append(indices)
    
    return new_indices


def get_unstructured_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate actual sparsity (fraction of zero weights) in the model.
    Useful for verifying unstructured pruning.

    Args:
        model: The pruned model

    Returns:
        Dictionary with layer-wise and global sparsity metrics
    """
    sparsity_info = {}
    total_weights = 0
    total_zero = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.data
            num_weights = weights.numel()
            num_zero = (weights.abs() < 1e-8).sum().item()  # Use small threshold for floating point

            layer_sparsity = num_zero / num_weights if num_weights > 0 else 0
            sparsity_info[name] = layer_sparsity

            total_weights += num_weights
            total_zero += num_zero

    sparsity_info['global'] = total_zero / total_weights if total_weights > 0 else 0

    return sparsity_info