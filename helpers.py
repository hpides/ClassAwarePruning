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
    Prunes if ratio > 0, otherwise just replaces the last layer.

    Args:
        pruner (Pruner): The pruner in question.
        ratio (float): Ratio to prune (0.0 to 1.0)

    Returns:
        Tuple(nn.Module, float): Pruned model and time needed for filter removal.
    """
    start = time.perf_counter()
    if ratio > 0:
        pruned_model = pruner.prune()
        print("%%%%%% Model pruned successfully.")
    else:
        pruner._replace_last_layer()
        pruned_model = pruner.model
    return pruned_model, time.perf_counter() - start


def evaluate(model, loader, cfg, device, inference_time_before, mapping, label, is_pruned=True):
    """
    Run inference + accuracy measurement for evaluation.

    Args:
        model (nn.Module): The model to run inference on.
        loader (DataLoader): DataLoader for inference.
        cfg (DictConfig): Config for the current run.
        device (torch.device): Device to run inference on.
        inference_time_before (float): Inference time of the base run.
        mapping (dict): Mapping for the indices on a subset.
        label (str): Label of the current run.
        is_pruned (boolean): Whether we are dealing with the base model or a pruned model.

    Returns:
        Tuple(float, float, float): Accuracy, Inference time, Ratio of inference time compared to base run.
    """
    print(f"\n%%%%%% {"=" * 80}")
    print(f"%%%%%% {label}:")
    print(f"%%%%%% {"=" * 80}\n")
    _, class_accuracies, inference_time, _ = measure_inference_time_and_accuracy(
        loader, model, device,
        cfg.training.batch_size_test,
        cfg.dataset.num_classes,
        all_classes=True,
        print_results=True,
        selected_classes=cfg.selected_classes.copy() if (cfg.replace_last_layer and is_pruned) else None,
        with_onnx=cfg.inference_with_onnx,
        mapping=mapping
    )
    accuracy = calculate_accuracy_for_selected_classes(class_accuracies, cfg.selected_classes)
    ratio = (inference_time / inference_time_before) if inference_time_before > 0 else 0
    print(f"\n%%%%%% {"=" * 80}")
    print(f"%%%%% STATS FOR {label}:")
    print(f"%%%%% ACCURACY: {accuracy}, INFERENCE: {inference_time:.2f}, INFERENCE TIME RATIO: {ratio}")
    print(f"%%%%%% {"=" * 80}\n")
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
    """
     Run training.

     Args:
         cfg (DictConfig): Config for the current run.
         model (nn.Module): The model to train.
         train_loader (DataLoader): DataLoader for training.
         val_loader (DataLoader): DataLoader for validation.
         device (torch.device): Device to run inference on.
         scheduler (lr_scheduler): Learning rate scheduler to adjust the optimizer’s learning rate during training.
         num_epochs (int): Numer of epochs to train at max.
         log_results (boolean): Whether to log results in WandB.
         num_classes (int): Number of classes in the dataset.
         retrain (boolean): Whether to retrain the model.
         patience (int): Number of consecutive epochs needed without improvement for early stopping.
         min_delta (float): The threshold for when an epoch is classified as "without improvement".

     Returns:
         Tuple(float, int): Best accuracy and epoch said accuracy occurred.
     """
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        cfg.training.optimizer, model, cfg.training.lr if not retrain else cfg.training.lr_retrain,
        cfg.training.weight_decay
    )
    model_name = cfg.model.name,

    best_accuracy = 0.0
    best_epoch = 1
    epochs_without_improvement = 0

    print("%%%%% " + "=" * 60)
    print(f"%%%%% Starting training for {num_epochs} epochs")
    print(f"%%%%% Train batches per epoch: {len(train_loader)}")
    print(f"%%%%% Train samples: {len(train_loader.dataset)}")
    print(f"%%%%% Val samples: {len(val_loader.dataset)}")
    print("%%%%% " + "=" * 60)
    print(f"%%%%% GPU memory right before first epoch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    model.train()
    for epoch in range(num_epochs):
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
            running_loss += loss.item()
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
    """
    Returns a requested optimizer.

    Args:
        name (str): The name of the requested optimizer.
        model (nn.Module): The model to be optimized.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

    Returns:
        torch.optim: The requested optimizer.
    """
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"xxxxx Unsupported optimizer: {name}")


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
        raise ValueError(f"xxxxx Unsupported activation function: {name}")


def get_names_of_conv_layers(model: nn.Module):
    """Get the names of all Conv2d layers in the model."""
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer_names.append(name)
    return conv_layer_names


def get_pruning_indices(masks):
    """Get indices of all filters to be pruned."""
    pruning_indices = {}
    for name, mask in masks.items():
        prune_indices = mask.logical_not().nonzero(as_tuple=False).squeeze(1)
        pruning_indices[name] = prune_indices.tolist()
    return pruning_indices


def filter_pruning_indices_for_resnet(all_indices: dict, model_name: str):
    """
    Skip connections in ResNet-18 pose challenges for structured
    pruning, as they directly add a block’s input to its output,
    potentially leading to dimensional mismatches. To address
    this, we refrain from pruning the last convolutional layers in
    blocks with residual connections, thereby ensuring that the
    output dimensions match the input.
    """
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
    Calculate actual sparsity (fraction of zero weights) in the model for unstructured pruning methods.

    Args:
        model (nn.Module): The pruned model

    Returns:
        Dict(str -> float): Dictionary with layer-wise and global sparsity metrics
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

    sparsity_info["global"] = total_zero / total_weights if total_weights > 0 else 0

    return sparsity_info