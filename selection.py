import torch.nn as nn
import torch
import random
from abc import ABC, abstractmethod
from ocap import Compute_layer_mask
from lrp import get_candidates_to_prune
from helpers import (
    get_names_of_conv_layers,
    get_activation_function,
    get_pruning_indices,
)
from omegaconf import DictConfig


def get_selector(
    selector_config: DictConfig,
    data_loader: torch.utils.data.DataLoader | None = None,
    device: str | None = None,
) -> "PruningSelection":
    """
    Factory function to get a pruning selection strategy based on the type.
    Args:
        selector_type (str): Type of the pruning selection strategy.
        **kwargs: Additional arguments for the specific selection strategy.
    Returns:
        PruningSelection: An instance of the specified pruning selection strategy.
    """

    if selector_config.name == "random":
        return RandomSelection(pruning_ratio=selector_config.pruning_ratio)
    elif selector_config.name == "ocap":
        return OCAP(
            pruning_ratio=selector_config.pruning_ratio,
            data_loader=data_loader,
            activation_func=selector_config.activation_func,
            device=device,
            skip_first_layer=selector_config.skip_first_layer,
        )
    elif selector_config.name == "lrp":
        return LRPPruning(
            num_filters=selector_config.num_filters, data_loader=data_loader
        )


class PruningSelection(ABC):
    @abstractmethod
    def select(self, model: nn.Module):
        pass


class RandomSelection(PruningSelection):
    def __init__(self, pruning_ratio: float):
        super().__init__()
        self.pruning_ratio = pruning_ratio

    def select(self, model: nn.Module):
        """Selects a random subset of filters to prune based on the specified pruning ratio."""

        selected_filters = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                num_to_prune = int(num_filters * self.pruning_ratio)
                filters_to_prune = random.sample(range(num_filters), num_to_prune)
                selected_filters[name] = filters_to_prune
        return selected_filters


class OCAP(PruningSelection):
    def __init__(
        self,
        pruning_ratio: float,
        data_loader: torch.utils.data.DataLoader,
        activation_func: str = "relu",
        device="mps",
        skip_first_layer: bool = True,
    ):
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.data_loader = data_loader
        self.activation_func = get_activation_function(activation_func)
        self.device = device
        self.skip_first_layer = skip_first_layer

    def select(self, model: nn.Module):
        """Selects filters to prune based on the OCAP method."""

        layer_masks, _ = Compute_layer_mask(
            imgs_dataloader=self.data_loader,
            model=model,
            percent=self.pruning_ratio,
            device=self.device,
            activation_func=self.activation_func,
        )

        names_of_conv_layers = get_names_of_conv_layers(model)
        if self.skip_first_layer:
            layer_masks = layer_masks[1:]
            names_of_conv_layers = names_of_conv_layers[1:]
        masks = {name: layer_masks[i] for i, name in enumerate(names_of_conv_layers)}
        indices = get_pruning_indices(masks)

        return indices


class LRPPruning(PruningSelection):
    def __init__(
        self,
        num_filters: int,
        data_loader: torch.utils.data.DataLoader,
        device="cpu",
        skip_first_layer: bool = True,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.data_loader = data_loader
        self.device = device
        self.skip_first_layer = skip_first_layer

    def select(self, model: nn.Module):
        """Selects filters to prune based on the LAP Pruning method."""
        data_iter = iter(self.data_loader)
        X, y = next(data_iter)

        indices = get_candidates_to_prune(
            model=model.to(self.device),
            num_filters_to_prune=self.num_filters,
            X_test=X.to(self.device),
            y_test_true=y.to(self.device),
        )

        if self.skip_first_layer:
            # Skip the first layer's indices if specified
            first_layer_name = list(model.named_modules())[1][0]
            indices = {k: v for k, v in indices.items() if k != first_layer_name}

        return indices
