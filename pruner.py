import torch
import torch.nn as nn
import torch.fx as fx
import copy


class StructuredPruner:
    def __init__(
        self,
        model: nn.Module,
        masks: dict[str, torch.Tensor],
        selected_classes: list[int],
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

    def prune(self):
        # Symbolically trace the model
        traced = fx.symbolic_trace(self.model)

        modules = dict(traced.named_modules())
        for node in traced.graph.nodes:
            if node.op == "call_module" and isinstance(modules[node.target], nn.Conv2d):
                conv_name = node.target
                if conv_name not in self.masks:
                    continue  # Skip unmasked layers

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
                while user_nodes:
                    user = user_nodes.pop(0)
                    if user.op == "call_module" or user.op == "call_function":
                        if user.op == "call_module":
                            next_mod = modules[user.target]
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
                        elif (
                            isinstance(next_mod, nn.Linear)
                            and user.target == "classifier.0"
                        ):
                            new_linear = self._adjust_first_linear_layer(
                                next_mod, keep_indices
                            )
                            self._set_module_by_qualified_name(
                                self.model, user.target, new_linear
                            )
                        else:
                            user_nodes.extend(list(user.users.keys()))

        # Replace the last layer for classification
        last_linear = self.model.classifier[-1]
        new_last_linear = self.replace_last_layer(last_linear)
        self.model.classifier[-1] = new_last_linear

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

    def replace_last_layer(self, linear: nn.Linear):
        new_linear = nn.Linear(
            in_features=linear.in_features,
            out_features=len(self.selected_classes),
            bias=(linear.bias is not None),
        )
        new_linear.weight.data = linear.weight.data[self.selected_classes].clone()
        if linear.bias is not None:
            new_linear.bias.data = linear.bias.data[self.selected_classes].clone()
        return new_linear
