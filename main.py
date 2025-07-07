import torch
from torch import nn
import hydra
from omegaconf import DictConfig
from data_loader import get_CIFAR10_dataloaders
from helpers import (
    train_model,
    evaluate_model,
    get_parameter_ratio,
    get_optimizer,
)
from pruner import DepGraphPruner, StructuredPruner
from selection import get_selector
from models import get_model


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

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
        train_shuffle=True,
        selected_classes=cfg.selected_classes,
        num_pruning_samples=cfg.num_pruning_samples,
    )

    # Select the filters to prune
    selector = get_selector(
        selector_config=cfg.pruning, data_loader=subset_data_loader, device=device
    )
    indices, masks = selector.select(model=model)

    # Prune the model
    # pruner = DepGraphPruner(
    #     model=model,
    #     indices=indices,
    #     replace_last_layer=cfg.replace_last_layer,
    #     selected_classes=cfg.selected_classes,
    #     device=device,
    # )
    pruner = StructuredPruner(
        model=model, masks=masks, selected_classes=cfg.selected_classes
    )

    pruned_model = pruner.prune()
    torch.save(pruned_model.state_dict(), "pruned_model.pth")
    print("Model pruned successfully.")

    # Evaluate the model before and after pruning
    print("Before pruning:")
    model.to(device)
    evaluate_model(model, device, test_loader, print_results=True, all_classes=True)
    print("After pruning:")
    evaluate_model(
        pruned_model, device, test_loader, print_results=True, all_classes=True
    )
    print(f"Parameter ratio after pruning: {get_parameter_ratio(model, pruned_model)}")


if __name__ == "__main__":
    main()
