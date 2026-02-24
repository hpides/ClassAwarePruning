import torch
import torch.nn as nn
import torch.fx as fx
import torch_pruning as tp
import copy
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Dict, List, Tuple


class StructuredPruner:
    def __init__(
        self,
        model: nn.Module,
        masks: dict[str, torch.Tensor],
        selected_classes: list[int],
        replace_last_layer: bool = True,
    ):
        """
        Args:
            model (nn.Module): The model to prune.
            masks (dict): Mapping from qualified names of Conv2d layers to filter masks.
                          Example: {'conv1': tensor([1, 0, 1, ...])}

        Usage:
            mask = torch.tensor([1] * 128 + [0] * 128, dtype=torch.bool)
            masks = {"features.10": mask}
            pruner = StructuredPruner(model, masks)
            pruned_model = pruner.prune()

        """
        self.model = copy.deepcopy(model)
        self.masks = masks
        self.selected_classes = selected_classes
        self.replace_last_layer = replace_last_layer

    def prune(self):
        # Symbolically trace the model
        traced = fx.symbolic_trace(self.model)
        modules = dict(traced.named_modules())
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(modules[node.target], nn.Conv2d):
                conv_name = node.target
                if conv_name not in self.masks or "downsample" in conv_name:
                    continue  # Skip unmasked layers and downsample layers
                old_conv = dict(self.model.named_modules())[conv_name]
                mask = self.masks[conv_name]
                keep_indices = mask.nonzero(as_tuple=False).squeeze(1)

                # Create new Conv2d layer with fewer output channels
                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=len(keep_indices),
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=old_conv.bias is not None,
                )
                new_conv.weight.data = old_conv.weight.data[keep_indices].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data[keep_indices].clone()

                # Replace layer in the original model
                self._set_module_by_qualified_name(self.model, conv_name, new_conv)

                # Adjust the next layers consuming this output
                user_nodes = list(node.users)
                parent_block = self._find_parent_block(conv_name)
                last_user = None
                while user_nodes:
                    user = user_nodes.pop(0)
                    if user.op == "call_module" or user.op == "call_function":
                        if user.op == "call_module":
                            next_mod = modules[user.target]
                            # In case of ResNet, don't modify layers in the next block
                            if parent_block and (
                                not user.target.split(".")[:-1]
                                == conv_name.split(".")[:-1]
                            ):
                                continue
                        elif user.op == "call_function" and "add" in user.name:
                            if parent_block:
                                # Add zeros to avoid dimension mismatch is addition with shortcut
                                last_user_name = last_user.target.split(".")[-1]
                                zero_insertion_module = ZeroInsertion(
                                    keep_indices, old_conv.out_channels
                                )
                                combined_module = nn.Sequential(
                                    getattr(parent_block, last_user_name),
                                    zero_insertion_module,
                                )
                                parent_block.__setattr__(
                                    last_user_name, combined_module
                                )
                                continue
                        else:
                            next_mod = None

                        if isinstance(next_mod, nn.Conv2d):
                            updated = self._adjust_input_channels(
                                next_mod, keep_indices
                            )
                            self._set_module_by_qualified_name(
                                self.model, user.target, updated
                            )
                        elif isinstance(next_mod, nn.BatchNorm2d):
                            new_bn = self._adjust_batchnorm(next_mod, keep_indices)
                            self._set_module_by_qualified_name(
                                self.model, user.target, new_bn
                            )
                            user_nodes.extend(list(user.users.keys()))
                            last_user = user
                        elif isinstance(next_mod, nn.Linear) and (
                            user.target == "classifier.0" or user.target == "fc"
                        ):
                            new_linear = self._adjust_first_linear_layer(
                                next_mod, keep_indices
                            )
                            self._set_module_by_qualified_name(
                                self.model, user.target, new_linear
                            )
                        else:
                            user_nodes.extend(list(user.users.keys()))
                            last_user = user

        # Replace the last layer for classification
        if self.replace_last_layer:
            self._replace_last_layer()

        return self.model

    def _adjust_input_channels(self, conv: nn.Conv2d, keep_indices: torch.Tensor):
        new_conv = nn.Conv2d(
            in_channels=len(keep_indices),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        new_conv.weight.data = conv.weight.data[:, keep_indices, :, :].clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
        return new_conv

    def _adjust_batchnorm(self, bn: nn.BatchNorm2d, keep_indices: torch.Tensor):
        new_bn = nn.BatchNorm2d(len(keep_indices))
        new_bn.weight.data = bn.weight.data[keep_indices].clone()
        new_bn.bias.data = bn.bias.data[keep_indices].clone()
        new_bn.running_mean = bn.running_mean[keep_indices].clone()
        new_bn.running_var = bn.running_var[keep_indices].clone()
        return new_bn

    def _set_module_by_qualified_name(
        self, root: nn.Module, qname: str, new_module: nn.Module
    ):
        parts = qname.split(".")
        for p in parts[:-1]:
            root = getattr(root, p)
        setattr(root, parts[-1], new_module)

    def _find_parent_block(self, name: str):
        parts = name.split(".")
        root = self.model
        for p in parts[:-1]:
            root = getattr(root, p)
        if isinstance(root, BasicBlock) or isinstance(root, Bottleneck):
            return root
        else:
            return None

    def _adjust_first_linear_layer(self, linear: nn.Linear, keep_indices: torch.Tensor):
        # W.shape = (out_channels, in_channels)
        # W[:, keep_indices]
        new_linear = nn.Linear(
            in_features=len(keep_indices) * 7 * 7,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
        )
        all_keep_indices = [range(i * 7 * 7, (i + 1) * 7 * 7) for i in keep_indices]
        all_keep_indices = torch.tensor(all_keep_indices).flatten()

        new_linear.weight.data = linear.weight.data[:, all_keep_indices].clone()
        if linear.bias is not None:
            new_linear.bias.data = linear.bias.data.clone()
        return new_linear

    def _replace_last_layer(self):
        layer_name, last_linear = list(self.model.named_modules())[-1]
        new_linear = nn.Linear(
            in_features=last_linear.in_features,
            out_features=len(self.selected_classes),
            bias=(last_linear.bias is not None),
        )
        new_linear.weight.data = last_linear.weight.data[self.selected_classes].clone()
        if last_linear.bias is not None:
            new_linear.bias.data = last_linear.bias.data[self.selected_classes].clone()

        self._replace_module(layer_name, new_linear)

    def _replace_module(self, module_name, new_module):
        parts = module_name.split(".")
        parent = self.model
        for name in parts[:-1]:
            parent = getattr(parent, name)
        setattr(parent, parts[-1], new_module)


class DepGraphPruner:
    def __init__(
        self,
        model: nn.Module,
        indices,
        replace_last_layer=True,
        selected_classes=[],
        device="cpu",
    ):
        """
        Args:
            model (nn.Module): The model to prune.
            masks (dict): Mapping from qualified names of Conv2d layers to filter masks.
                          Example: {'conv1': tensor([1, 0, 1, ...])}
        """
        self.model = copy.deepcopy(model)
        self.pruning_indices = indices
        self.replace_last_layer = replace_last_layer
        self.selected_classes = selected_classes
        self.device = device

    def prune(self):
        # Build the dependency graph
        self.model.to(self.device)
        DG = tp.DependencyGraph().build_dependency(
            self.model, example_inputs=torch.randn(1, 3, 224, 224).to(self.device)
        )
        print(f"%%%%%% DEPENDENCY GRAPH: {DG}")
        #print(f"%%%%%% PRUNING INDICES: {self.pruning_indices}")
        print(f"%%%%%% SELECTED CLASSES: {self.selected_classes}")

        for name, indices in self.pruning_indices.items():
            module = dict(self.model.named_modules())[name]
            # Prune the Conv2d layers
            group = DG.get_pruning_group(
                module,
                tp.prune_conv_out_channels,
                idxs=indices,
            )

            #print(f"%%%%%% GROUP: {group}")
            #print(f"%%%%%% NAME AND INDICES: {name}, {indices}")

            #if len(indices) >= module.out_channels * 0.95:
            #    print(f"##### WARNING: Attempting to prune {len(indices)}/{module.out_channels} channels - reducing")
            #    indices = indices[:int(module.out_channels * 0.9)]
            if DG.check_pruning_group(group):  # avoid over-pruning, i.e., channels=0.
                group.prune()
            else:
                indices.pop(0)
                group = DG.get_pruning_group(
                dict(self.model.named_modules())[name],
                tp.prune_conv_out_channels,
                idxs=indices,
                )
                group.prune()
        #print(f"%%%%%% MODEL AFTER PRUNING BUT BEFORE LAST LAYER SWAP: {self.model}")

        if self.replace_last_layer and self.selected_classes:
            self._replace_last_layer()

        return self.model

    def _replace_last_layer(self):
        layer_name, last_linear = list(self.model.named_modules())[-1]
        #print(f"%%%%%% LAYER NAME: {layer_name}")
        #print(f"%%%%%% LAST LINEAR LAYER: {last_linear}")
        #print(f"%%%%%% IN FEATURES: {last_linear.in_features}")
        #print(f"%%%%%% OUT FEATURES: {len(self.selected_classes)}")
        new_linear = nn.Linear(
            in_features=last_linear.in_features,
            out_features=len(self.selected_classes),
            bias=(last_linear.bias is not None),
        )
        new_linear.weight.data = last_linear.weight.data[self.selected_classes].clone()
        if last_linear.bias is not None:
            new_linear.bias.data = last_linear.bias.data[self.selected_classes].clone()

        self._replace_module(layer_name, new_linear)

    def _replace_module(self, module_name, new_module):
        parts = module_name.split(".")
        parent = self.model
        for name in parts[:-1]:
            parent = getattr(parent, name)
        setattr(parent, parts[-1], new_module)


class ZeroInsertion(nn.Module):

    def __init__(self, indices: torch.Tensor, out_features: int) -> None:
        super().__init__()
        self.register_buffer("indices", indices)
        self.out_features = out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_shape = [
            input.shape[0],
            self.out_features,
            input.shape[2],
            input.shape[3],
        ]
        output = torch.zeros(output_shape, dtype=input.dtype, device=input.device)
        output[:, self.indices] = input
        return output


##########################################
#########  UNSTRUCTURED PRUNING  #########
##########################################

class UnstructuredMagnitudePruner:
    """
    Applies magnitude-based global unstructured pruning to a model.

    Prunes weights globally across specified layers based on their absolute magnitude,
    zeroing out the smallest weights while preserving the sparsity pattern.
    """

    def __init__(
            self,
            model: nn.Module,
            sparsity: float,
            layer_types: Tuple[type, ...] = (nn.Conv2d, nn.Linear),
            exclude_layers: List[str] = None,
            replace_last_layer: bool = True,
            selected_classes: List[int] = None,
            device: str = None
    ):
        """
        Args:
            model: The model to prune
            sparsity: Global sparsity ratio (0.0 to 1.0). E.g., 0.5 = prune 50% of weights
            layer_types: Tuple of layer types to prune (default: Conv2d and Linear)
            exclude_layers: List of layer names to exclude from pruning (e.g., ['classifier'])
            replace_last_layer: Whether to replace the final classification layer
            selected_classes: List of target class indices for class-aware pruning
            device: Device to run on (auto-detected if None)
        """
        self.model = copy.deepcopy(model)
        self.sparsity = sparsity
        self.layer_types = layer_types
        self.exclude_layers = exclude_layers or []
        self.replace_last_layer = replace_last_layer
        self.selected_classes = selected_classes or []

        # Auto-detect device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

    def prune(self) -> nn.Module:
        """
        Apply global magnitude-based unstructured pruning to the model.

        Returns:
            Pruned model with weights zeroed and last layer replaced (if specified)
        """
        self.model.to(self.device)

        # Collect all prunable parameters
        parameters_to_prune = []

        for name, module in self.model.named_modules():
            # Skip excluded layers
            if any(exclude in name for exclude in self.exclude_layers):
                continue

            # Check if module is a prunable layer type
            if isinstance(module, self.layer_types):
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            print("Warning: No parameters found to prune")
            return self.model

        # Perform global magnitude-based pruning
        self._global_unstructured_pruning(parameters_to_prune)

        # Replace last layer if needed
        if self.replace_last_layer and self.selected_classes:
            self._replace_last_layer()

        # Bake masks in and remove buffers to avoid doubling model size
        self._make_permanent()

        return self.model

    def _make_permanent(self):
        """
        Permanently apply pruning masks: bake zeros into weights,
        then remove masks and hooks to restore original model size.
        """
        for name, module in self.model.named_modules():
            mask_name = 'weight_mask'
            if hasattr(module, mask_name):
                # Mask is already applied to weight.data, so just clean up
                # Remove the hook
                if hasattr(module, '_pruning_hook_handle'):
                    module._pruning_hook_handle.remove()
                    del module._pruning_hook_handle

                # Remove the mask buffer
                del module._buffers[mask_name]

        return self.model

    def _global_unstructured_pruning(self, parameters_to_prune: List[Tuple[nn.Module, str]]):
        """
        Apply global magnitude-based pruning across all specified parameters.

        This is optimized for GPU execution and follows PyTorch's global_unstructured approach.
        """
        # Gather all weights into a single tensor for efficient threshold computation
        all_weights = []
        weight_shapes = []

        for module, param_name in parameters_to_prune:
            weight = getattr(module, param_name)
            all_weights.append(weight.data.abs().flatten())
            weight_shapes.append(weight.shape)

        # Concatenate all weights on GPU for fast processing
        all_weights_tensor = torch.cat(all_weights)

        # Calculate the threshold based on global sparsity
        num_weights = all_weights_tensor.numel()
        num_to_prune = int(self.sparsity * num_weights)

        if num_to_prune == 0:
            print("Warning: Sparsity too low, no weights pruned")
            return

        # Use kthvalue for efficient threshold finding (GPU-accelerated)
        threshold = torch.kthvalue(all_weights_tensor, num_to_prune)[0]

        # Apply masks to each parameter
        for module, param_name in parameters_to_prune:
            weight = getattr(module, param_name)

            # Create binary mask (1=keep, 0=prune)
            mask = (weight.data.abs() > threshold).to(weight.dtype)

            # Apply mask by zeroing out weights
            weight.data.mul_(mask)

            # Register mask as a buffer to persist it
            module.register_buffer(f'{param_name}_mask', mask)

            # Register forward pre-hook to maintain sparsity during training
            self._register_pruning_hook(module, param_name)

    def _register_pruning_hook(self, module: nn.Module, param_name: str):
        """Register a forward pre-hook to maintain pruning mask."""
        mask_name = f'{param_name}_mask'

        # Remove existing hook if present
        if hasattr(module, '_pruning_hook_handle'):
            module._pruning_hook_handle.remove()

        def pruning_hook(mod, input):
            if hasattr(mod, mask_name):
                mask = getattr(mod, mask_name)
                param = getattr(mod, param_name)
                param.data.mul_(mask)

        hook_handle = module.register_forward_pre_hook(pruning_hook)
        module._pruning_hook_handle = hook_handle

    def _replace_last_layer(self):
        """Replace the last linear layer for class-aware pruning."""
        # Find the last Linear layer
        last_linear = None
        last_linear_name = None

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                last_linear_name = name

        if last_linear is None:
            print("Warning: No Linear layer found, skipping last layer replacement")
            return

        # Create new linear layer with selected classes
        new_linear = nn.Linear(
            in_features=last_linear.in_features,
            out_features=len(self.selected_classes),
            bias=(last_linear.bias is not None),
            device=self.device
        )

        # Copy weights for selected classes
        with torch.no_grad():
            new_linear.weight.data = last_linear.weight.data[self.selected_classes].clone()
            if last_linear.bias is not None:
                new_linear.bias.data = last_linear.bias.data[self.selected_classes].clone()

        # Replace the module
        self._replace_module(last_linear_name, new_linear)

    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Helper to replace a module in the model hierarchy."""
        parts = module_name.split(".")
        parent = self.model

        for name in parts[:-1]:
            parent = getattr(parent, name)

        setattr(parent, parts[-1], new_module)
