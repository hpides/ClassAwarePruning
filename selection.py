import torch.nn as nn
import torch
import random
import numpy as np
from abc import ABC, abstractmethod
from ocap import Compute_layer_mask
from lrp import get_candidates_to_prune
from helpers import (
    get_names_of_conv_layers,
    get_activation_function,
    get_pruning_indices,
)
from omegaconf import DictConfig
from torchpruner.attributions import TaylorAttributionMetric, APoZAttributionMetric, SensitivityAttributionMetric
from typing import List, Tuple, Union


def get_selector(
    selector_config: DictConfig,
    data_loader: torch.utils.data.DataLoader | None = None,
    device: str | None = None,
    skip_first_layers: int = 0
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
        return RandomSelection(pruning_ratio=selector_config.pruning_ratio, skip_first_layers=skip_first_layers)
    elif selector_config.name == "ocap":
        return OCAP(
            pruning_ratio=selector_config.pruning_ratio,
            data_loader=data_loader,
            activation_func=selector_config.activation_func,
            device=device,
            skip_first_layers=skip_first_layers,
        )
    elif selector_config.name == "lrp":
        return LRPPruning(
            pruning_ratio=selector_config.pruning_ratio,
            data_loader=data_loader,
            skip_first_layers=skip_first_layers,
            device=device,
        )
    elif selector_config.name == "ln_structured":
        return LnStructuredPruning(selector_config.pruning_ratio, skip_first_layers, device, selector_config.norm, selector_config.pruning_scope)
    elif selector_config.name == "cap":
        return CAP(
            pruning_ratio=selector_config.pruning_ratio,
            dataloader=data_loader,
            device=device,
            skip_first_layers=skip_first_layers,
        )
    elif selector_config.name == "torchpruner":
        return TorchPrunerAttributions(
            pruning_ratio=selector_config.pruning_ratio,
            dataloader=data_loader,
            attribution=selector_config.attribution,
            device=device,
            skip_first_layers=skip_first_layers,
        )


class PruningSelection(ABC):

    def __init__(self, skip_first_layers: int = 0):
        super().__init__()
        self.skip_first_layers = skip_first_layers

    @abstractmethod
    def select(self, model: nn.Module):
        pass

    def _remove_first_layers_in_selection(self, selection: dict, model: nn.Module):
        """
        Remove the first layers from the selected indices/masks.
        Args:
            selction (dict): Dictionary of indices or masks.
        Returns:
            dict: Filtered dictionary.
        """
        names_of_conv_layers = get_names_of_conv_layers(model)
        names_of_conv_layers = names_of_conv_layers[self.skip_first_layers:] 
        selection = {
            name: selection.get(name, None)
            for name in names_of_conv_layers
            if name in selection
            }
        return selection
    
    def _calculate_global_pruning_ratio(self, selection: dict, model: nn.Module):
        total_filters = 0
        total_pruned = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                total_filters += num_filters
                if name in selection:
                    num_pruned = len(selection[name])
                    total_pruned += num_pruned
        return total_pruned / total_filters if total_filters > 0 else 0
        
           
        
class RandomSelection(PruningSelection):
    def __init__(self, pruning_ratio: float, skip_first_layers: int = 0):
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.skip_first_layers = skip_first_layers

    def select(self, model: nn.Module):
        """Selects a random subset of filters to prune based on the specified pruning ratio."""
        
        all_selections = []
        self.global_pruning_ratio = []
        for ratio in self.pruning_ratio:
            indices = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    num_filters = module.out_channels
                    num_to_prune = int(num_filters * ratio)
                    filters_to_prune = random.sample(range(num_filters), num_to_prune)
                    indices[name] = filters_to_prune

            if self.skip_first_layers:
                indices = self._remove_first_layers_in_selection(indices, model)
            all_selections.append(indices)
            self.global_pruning_ratio.append(self._calculate_global_pruning_ratio(indices, model))

        return all_selections


class OCAP(PruningSelection):
    def __init__(
        self,
        pruning_ratio: List[float],
        data_loader: torch.utils.data.DataLoader,
        activation_func: str = "relu",
        device="mps",
        skip_first_layers: int = 1,
    ):
        super().__init__(skip_first_layers=skip_first_layers)
        self.pruning_ratios = pruning_ratio
        self.data_loader = data_loader
        self.activation_func = get_activation_function(activation_func)
        self.device = device
        self.global_pruning_ratio = []

    def select(self, model: nn.Module):
        """Selects filters to prune based on the OCAP method."""

        all_layer_masks, _ = Compute_layer_mask(
            imgs_dataloader=self.data_loader,
            model=model,
            ratios=self.pruning_ratios,
            device=self.device,
            activation_func=self.activation_func,
        )
        names_of_conv_layers = get_names_of_conv_layers(model)
        all_indices = []
        for layer_masks in all_layer_masks:
            masks = dict(zip(names_of_conv_layers, layer_masks))
            if self.skip_first_layers:
                masks = self._remove_first_layers_in_selection(masks, model)
            indices = get_pruning_indices(masks)
            self.global_pruning_ratio.append(self._calculate_global_pruning_ratio(indices, model))
            all_indices.append(indices)
            
        return all_indices


class LRPPruning(PruningSelection):
    def __init__(
        self,
        pruning_ratio: float,
        data_loader: torch.utils.data.DataLoader,
        device="cpu",
        skip_first_layers: int = None,
    ):
        super().__init__(skip_first_layers=skip_first_layers)
        self.pruning_ratio = pruning_ratio
        self.global_pruning_ratio = pruning_ratio
        self.data_loader = data_loader
        self.device = device

    def select(self, model: nn.Module):
        """Selects filters to prune based on the LAP Pruning method."""
        data_iter = iter(self.data_loader)
        X, y = next(data_iter)

        indices = get_candidates_to_prune(
            model=model.to(self.device),
            pruning_ratio=self.pruning_ratio,
            X_test=X.to(self.device),
            y_test_true=y.to(self.device),
        )

        all_indices = []
        self.global_pruning_ratio = []
        for indices_p in indices:
            if self.skip_first_layers:
                all_indices.append(self._remove_first_layers_in_selection(indices_p, model))
            self.global_pruning_ratio.append(self._calculate_global_pruning_ratio(indices_p, model))

        return all_indices


class LnStructuredPruning(PruningSelection):
    def __init__(self, pruning_ratio: float, skip_first_layers: int, device: str, norm: int=2, pruning_scope: str="layer"):
        super().__init__(skip_first_layers=skip_first_layers)
        self.pruning_ratio = pruning_ratio
        self.device = device
        self.norm = norm
        self.pruning_scope = pruning_scope


    def select(self, model: nn.Module):
        model.to(self.device)
        indices = [{} for _ in range(len(self.pruning_ratio))]

        scores_per_layer = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_scores = self._weight_norm(module)
                if self.pruning_scope == "global":
                    scores_per_layer[name] = layer_scores
                elif self.pruning_scope == "layer":  
                    num_filters = len(layer_scores)
                    for index, ratio in enumerate(self.pruning_ratio):
                        num_to_prune = int(num_filters * ratio)
                        top_indices = torch.topk(layer_scores, num_to_prune, largest=False).indices.tolist()
                        indices[index][name] = top_indices
        
        if self.pruning_scope == "global":
            scores = torch.cat(list(scores_per_layer.values()))
            scores = scores.sort().values
            for index, ratio in enumerate(self.pruning_ratio):
                threshold_element = scores[int(len(scores) * ratio)]
                for name, scores in scores_per_layer.items():
                    top_indices = (scores < threshold_element).nonzero(as_tuple=False).squeeze(1).tolist()
                    indices[index][name] = top_indices
        
        self.global_pruning_ratio = []
        for index in range(len(indices)):
            if self.skip_first_layers:
                indices[index] = self._remove_first_layers_in_selection(indices[index], model)

            self.global_pruning_ratio.append(self._calculate_global_pruning_ratio(indices[index], model))

        return indices
    
    def _weight_norm(self, module: nn.Module):
        weight = module.weight.data
        return torch.norm(weight, p=self.norm, dim=(1, 2, 3))  # Assuming 4D tensor for Conv2d
    

class CAP(PruningSelection):
    def __init__(self, pruning_ratio: float, dataloader: torch.utils.data.DataLoader, device: str, skip_first_layers: int = 0):
        super().__init__(skip_first_layers=skip_first_layers)
        self.pruning_ratio = pruning_ratio
        self.device = device
        self.data_loader = dataloader

    def select(self, model: nn.Module):
        selected_filters = {}
        index = 0
        total_filters = 0
        self._calculate_activations(model)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                num_to_prune = int(num_filters * self.pruning_ratio)
                total_filters += num_filters
                filters_to_prune = self._select_filters_for_layer(index, num_to_prune)
                selected_filters[name] = filters_to_prune
                index += 1
        if self.skip_first_layers:
            selected_filters = self._remove_first_layers_in_selection(selected_filters, model)

        self.global_pruning_ratio = self._calculate_global_pruning_ratio(selected_filters, model)
        return selected_filters
    
    def _select_filters_for_layer(self, index: int, num_to_prune: int):
        layer_activations = self.activations[index]
        layer_activations_scores = layer_activations.norm(dim=(2, 3), p=1).mean(dim=0)
        _, sorted_indices = torch.sort(layer_activations_scores)
        top_indices = sorted_indices[:num_to_prune].tolist()
        return top_indices

    def _calculate_activations(self, model: nn.Module):
        self.activations = []

        def forward_hook(module, input, output):
            self.activations.append(output.detach())

        model_handles = []
        for index, module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d):
                next_module = list(model.modules())[index + 1]
                handle = next_module.register_forward_hook(forward_hook)
                model_handles.append(handle)

        for inputs, _ in self.data_loader:
            model(inputs.to(self.device))
            break

        for handle in model_handles:
            handle.remove()


class TorchPrunerAttributions(PruningSelection):
    def __init__(self, pruning_ratio: float, dataloader: torch.utils.data.DataLoader, attribution: str, device: str, skip_first_layers: int = 0):
        super().__init__(skip_first_layers=skip_first_layers)
        self.pruning_ratio = pruning_ratio
        self.device = device
        self.data_loader = dataloader
        self.attribution = {"taylor": TaylorAttributionMetric,
                            "apoz": APoZAttributionMetric,
                            "sensitivity": SensitivityAttributionMetric}[attribution]


    def select(self, model: nn.Module):
        indices = []
        all_scores = []
        self.global_pruning_ratio = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                attr = self.attribution(model, self.data_loader, nn.CrossEntropyLoss(), self.device)
                scores = attr.run(module)
                scores = scores / np.linalg.norm(scores, ord=2)  # L2 normalization
                for index, score in enumerate(scores):
                    all_scores.append((score, name, index))
        
        all_scores.sort(key=lambda x: x[0])
        for ratio in self.pruning_ratio:
            indices_d = {}
            num_to_prune = int(len(all_scores) * ratio)
            for i in range(num_to_prune):
                _, layer_name, filter_index = all_scores[i]
                if layer_name not in indices_d:
                    indices_d[layer_name] = []
                indices_d[layer_name].append(filter_index)
            
            if self.skip_first_layers:
                indices_d = self._remove_first_layers_in_selection(indices_d, model)
            indices.append(indices_d)
            self.global_pruning_ratio.append(self._calculate_global_pruning_ratio(indices_d, model))
        
        return indices