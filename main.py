import torch
from torch import nn
import hydra
from omegaconf import DictConfig
from data_loader import get_CIFAR10_dataloaders
from helpers import (
    train_model,
    evaluate_model,
    get_names_of_conv_layers,
    get_parameter_ratio,
    get_optimizer,
)
from pruner import StructuredPruner
from ocap import Compute_layer_mask
from models import get_model


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    if cfg.dataset.name != "CIFAR10":
        raise ValueError(
            f"Dataset {cfg.dataset.name} is not supported. Supported datasets: CIFAR10"
        )

    train_loader, test_loader = get_CIFAR10_dataloaders(
        train_batch_size=cfg.batch_size_train,
        test_batch_size=cfg.batch_size_test,
        use_data_Augmentation=True,
        download=True,
        train_shuffle=True,
    )

    model = get_model(
        cfg.model.name, pretrained=True, num_classes=cfg.dataset.num_classes
    )

    if cfg.training.train:
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(
            cfg.training.optimizer, model, cfg.training.lr, cfg.training.weight_decay
        )
        model.to(device)

        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=cfg.training.epochs,
        )
    else:
        weights = torch.load(cfg.model.pretrained_weights_path, weights_only=True)
        model.load_state_dict(weights)
        model.to(device)

    subset_data_loader = get_CIFAR10_dataloaders(
        train_batch_size=cfg.training.train_batch_size,
        test_batch_size=cfg.training.test_batch_size,
        use_data_Augmentation=False,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
        selected_classes=cfg.selected_classes,
        num_pruning_samples=cfg.num_pruning_samples,
    )

    layer_masks, _ = Compute_layer_mask(
        imgs_dataloader=subset_data_loader,
        model=model,
        percent=0.1,
        device=device,
        activation_func=nn.ReLU(),
    )

    names_of_conv_layers = get_names_of_conv_layers(model)
    # Skip the first layer (input layer)
    layer_masks = layer_masks[1:]
    names_of_conv_layers = names_of_conv_layers[1:]
    masks = {name: layer_masks[i] for i, name in enumerate(names_of_conv_layers)}

    pruner = StructuredPruner(model, masks, cfg.selected_classes)
    pruned_model = pruner.prune()
    torch.save(pruned_model.state_dict(), "pruned_model.pth")
    print("Model pruned successfully.")
    print("Before pruning:")
    evaluate_model(model, device, test_loader, print_results=True, all_classes=True)
    print("After pruning:")
    evaluate_model(
        pruned_model, device, test_loader, print_results=True, all_classes=True
    )
    print(f"Parameter ratio after pruning: {get_parameter_ratio(model, pruned_model)}")


if __name__ == "__main__":
    main()
