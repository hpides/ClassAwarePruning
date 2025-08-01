import torch
from torch import nn


# This code is from the authors of the paper "OCAP: On-Device Class-Aware Pruning for Personalized Edge DNN Models"
# with small modifications to fit the current codebase
# https://github.com/mzd2222/OCAP
def Compute_layer_mask(
    imgs_dataloader,
    model,
    percent,
    device,
    activation_func
):
    """
    :argument Calculate masks based on the input image
    :return: masks dim [layer_num, c]
    """

    activations = []

    # activation_hook for mask
    def mask_activation_hook(module, input, output):
        activations.append(output.clone().detach().cpu())
        return

    #  change the model to eval here, otherwise the input data when calculating
    #  layer_mask will change the bn layer parameters, resulting in lower accuracy
    model.eval()

    with torch.no_grad():
        hooks = []
        activations.clear()
        new_activations = []

        for index, module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d):
                next_module = list(model.modules())[index + 1]
                hook = next_module.register_forward_hook(mask_activation_hook)
                hooks.append(hook)
      
        batch_times = 1
        data_iter = iter(imgs_dataloader)
        X, _ = next(data_iter)
        _ = model(X.to(device))
        


        for hook in hooks:
            hook.remove()

        num_layers = int(len(activations) / batch_times)

        for idx1 in range(num_layers):

            layer_list = []

            for idx2 in range(batch_times):
                layer_list.append(activations[idx2 * num_layers + idx1])

            combine_activation = torch.cat(layer_list, dim=0) # Concatenate activations across batches for the same layer

            new_activations.append(combine_activation)

        # ------ layer-by-layer
        masks = []
        score_num_list = []
        # New activations are now a list of tensors, each tensor corresponds to a layer
        for layer_activations in new_activations:
            if activation_func is not None:
                layer_activations = activation_func(layer_activations) # Apply activation function if specified
            # Normalize activations across spatial dimensions, go from  [img_num, c, h, w] to [img_num, c] 
            layer_activations_score = layer_activations.norm(dim=(2, 3), p=2)
            layer_masks = torch.empty_like(layer_activations_score, dtype=torch.bool)
            # Iterate over each image's activations to compute the mask, shape of imgs_activations_score is [c]
            for idx, imgs_activations_score in enumerate(layer_activations_score):
                sorted_tensor, _ = torch.sort(imgs_activations_score)
                # Calculate the threshold index based on the percentage (prune x percent of channels)
                threshold_index = min(int(len(sorted_tensor) * percent), len(sorted_tensor) - 1)
                threshold = sorted_tensor[threshold_index]
                one_img_mask = imgs_activations_score.gt(threshold) # Get all channels that are greater than the threshold
                layer_masks[idx] = one_img_mask

            """
            1 OCAP-AB
            """
            one_layer_mask = layer_masks[0]
            # [img_num, c] => [c]  
            # Add all masks for the same layer together (logical OR operation = keep a channel if it is kept in any image)
            for img in layer_masks[1:]:
                for channel_id, channel_mask in enumerate(img):
                    one_layer_mask[channel_id] = (
                        one_layer_mask[channel_id] | channel_mask
                    )

            masks.append(one_layer_mask)

        return masks, score_num_list
