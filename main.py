import torch
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf
from data_loader import get_CIFAR10_dataloaders
from metrics import (
    get_parameter_ratio,
    calculate_model_accuracy,
    get_model_size,
    measure_inference_time,
    calculate_accuracy_for_selected_classes,
)
from helpers import (
    train_model,
    get_optimizer,
    plot_accuracies,
    get_pruning_masks,
    filter_pruning_indices_for_resnet,
)
from pruner import StructuredPruner, DepGraphPruner
from selection import get_selector
from models import get_model
import wandb
import PIL


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if cfg.device:
        device = torch.device(cfg.device)

    wandb_cfg["device"] = device
    print(f"Using device: {device}")

    if cfg.log_results:
        wandb.init(
            project="ClassAwarePruning",
            entity="smilla-fox",
            config=wandb_cfg,
            name=cfg.run_name,
        )

    train_loader, test_loader = get_CIFAR10_dataloaders(
        train_batch_size=cfg.training.batch_size_train,
        test_batch_size=cfg.training.batch_size_test,
        use_data_Augmentation=True,
        download=True,
        train_shuffle=True,
    )

    model = get_model(
        cfg.model.name, pretrained=True, num_classes=cfg.dataset.num_classes
    )

    # Train the model or load pretrained weights
    if cfg.training.train:
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(
            cfg.training.optimizer, model, cfg.training.lr, cfg.training.weight_decay
        )
        model.to(device)

        train_model(
            model=model,
            model_name=cfg.model.name,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=cfg.training.epochs,
            log_results=cfg.log_results
        )
    else:
        weights = torch.load(
            cfg.model.pretrained_weights_path, weights_only=True, map_location=device
        )
        model.load_state_dict(weights)
        model.to(device)

    subset_data_loader = get_CIFAR10_dataloaders(
        train_batch_size=cfg.training.batch_size_train,
        test_batch_size=cfg.training.batch_size_test,
        use_data_Augmentation=False,
        data_path="./data/cifar",
        download=True,
        selected_classes=cfg.selected_classes,
        num_pruning_samples=cfg.num_pruning_samples,
    )

    # Select the filters to prune
    selector = get_selector(
        selector_config=cfg.pruning, data_loader=subset_data_loader, device=device, skip_first_layers=cfg.model.skip_first_layers
    )
    indices = selector.select(model=model)
    masks = get_pruning_masks(indices, model)

    if cfg.resnet_zero_insertion:
        pruner = StructuredPruner(
            model=model,
            masks=masks,
            selected_classes=cfg.selected_classes,
            replace_last_layer=cfg.replace_last_layer,
        )
    else:
        if cfg.model.name.startswith("resnet"):
            indices = filter_pruning_indices_for_resnet(indices, cfg.model.name)

        pruner = DepGraphPruner(
            model=model,
            indices=indices,
            replace_last_layer=cfg.replace_last_layer,
            selected_classes=cfg.selected_classes,
            device=device,
        )

    pruned_model = pruner.prune()
    torch.save(pruned_model.state_dict(), "pruned_model.pth")
    print("Model pruned successfully.")

    # Evaluate the model before and after pruning
    print("Before pruning:")
    model.to(device)
    pruned_model.to(device)
    _, class_accuracies_original = calculate_model_accuracy(
        model,
        device,
        test_loader,
        print_results=True,
        all_classes=True,
        num_classes=cfg.dataset.num_classes,
    )
    print("After pruning:")
    _, class_accuracies_pruned = calculate_model_accuracy(
        pruned_model,
        device,
        test_loader,
        print_results=True,
        all_classes=True,
        selected_classes=(
            cfg.selected_classes.copy() if cfg.replace_last_layer else None
        ),
        num_classes=cfg.dataset.num_classes,
    )
    accuracy_before = calculate_accuracy_for_selected_classes(
        class_accuracies_original, cfg.selected_classes
    )
    accuracy_after = calculate_accuracy_for_selected_classes(
        class_accuracies_pruned, cfg.selected_classes
    )
    print(f"Accuracy before pruning: {accuracy_before:.2f}%")
    print(f"Accuracy after pruning: {accuracy_after:.2f}%")
    model_size_before = get_model_size(model)
    model_size_after = get_model_size(pruned_model)
    inference_time_before = measure_inference_time(
        test_loader, model, device, cfg.training.batch_size_test
    )
    inference_time_after = measure_inference_time(
        test_loader, pruned_model, device, cfg.training.batch_size_test
    )

    print(f"Batch Inference time before pruning: {inference_time_before}")
    print(f"Batch Inference time after pruning: {inference_time_after}")
    print(f"Inference time ratio: {inference_time_after / inference_time_before}")

    print(f"Model size before pruning: {model_size_before} MB")
    print(f"Model size after pruning: {model_size_after} MB")
    image_path = plot_accuracies(
        class_accuracies_original, class_accuracies_pruned, cfg.model.name
    )
    image = PIL.Image.open(image_path)
    print(f"Parameter ratio after pruning: {get_parameter_ratio(model, pruned_model)}")
    if cfg.log_results:
        wandb.log(
            {
                "accuracy_before": accuracy_before,
                "accuracy_after": accuracy_after,
                "model_size_before": model_size_before,
                "model_size_after": model_size_after,
                "model_size_ratio": model_size_after / model_size_before,
                "parameter_ratio": get_parameter_ratio(model, pruned_model),
                "class_accuracies_original": class_accuracies_original,
                "class_accuracies_pruned": class_accuracies_pruned,
                "inference_time_batch_before": inference_time_before,
                "inference_time_batch_after": inference_time_after,
                "inference_time_ratio": inference_time_after / inference_time_before,
                "inference_time_per_sample_before": inference_time_before
                / cfg.training.batch_size_test,
                "inference_time_per_sample_after": inference_time_after
                / cfg.training.batch_size_test,
                "accuracies_plot": wandb.Image(image),
            }
        )
    if cfg.log_results:
        wandb.finish()


if __name__ == "__main__":
    main()
