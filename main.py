import time
import torch
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf
from data_loader import dataloaderFactorys
from metrics import (
    get_parameter_ratio,
    measure_execution_time,
    get_model_size,
    measure_inference_time_and_accuracy,
    calculate_accuracy_for_selected_classes,
)
from helpers import (
    train_model,
    get_optimizer,
    filter_pruning_indices_for_resnet,
)
from pruner import DepGraphPruner
from selection import get_selector
from models import get_model
import wandb
from fvcore.nn import FlopCountAnalysis


def train(cfg: DictConfig, model, train_loader, test_loader, device, retrain=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
            cfg.training.optimizer, model, cfg.training.lr if retrain else cfg.training.lr_retrain, cfg.training.weight_decay
        )
        
    model.to(device)

    return train_model(
        model=model,
        model_name=cfg.model.name,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.training.epochs if not retrain else cfg.training.retrain_epochs,
        log_results=cfg.log_results,
        num_classes=cfg.dataset.num_classes,
    )

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if max(cfg.pruning.pruning_ratio) > 0.77 and cfg.model.name == "resnet18" and cfg.pruning.name == "lrp":
        raise ValueError("Pruning ratio too high for resnet18, please choose a value < 0.77")

    if cfg.device:
        device = torch.device(cfg.device)

    wandb_cfg["device"] = device
    print(f"Using device: {device}")

    dataloader_factory = dataloaderFactorys[cfg.dataset.name](
        train_batch_size=cfg.training.batch_size_train,
        test_batch_size=cfg.training.batch_size_test,
        selected_classes=cfg.selected_classes,
        num_pruning_samples=cfg.num_pruning_samples,
        use_data_augmentation=cfg.training.use_data_augmentation,
        use_imagenet_labels=cfg.dataset.use_imagenet_labels if "use_imagenet_labels" in cfg.dataset else False
    )

    if cfg.log_results:
        wandb.init(
            project="ClassAwarePruning",
            entity="smilla-fox",
            config=wandb_cfg,
            name=cfg.run_name,
        )

    train_loader, test_loader = dataloader_factory.get_dataloaders()

    model = get_model(
        cfg.model.name, pretrained=cfg.training.use_pretrained_model, num_classes=cfg.dataset.num_classes, dataset_name=cfg.dataset.name
    )

    # Train the model or load pretrained weights
    if cfg.training.train:
       train(cfg, model, train_loader, test_loader, device)
    elif cfg.model.pretrained_weights_path and cfg.dataset.name != "imagenet":
        weights = torch.load(
            cfg.model.pretrained_weights_path, weights_only=True, map_location=device
        )
        model.load_state_dict(weights)
    
    model.to(device)
    subset_data_loader_train, subset_data_loader_test = dataloader_factory.get_subset_dataloaders()
    if cfg.pruning.name == "torchpruner":
        subset_data_loader_train_retrain = subset_data_loader_train
        subset_data_loader_train = dataloader_factory.get_small_train_loader()
    # Select the filters to prune
    selector = get_selector(
        selector_config=cfg.pruning, data_loader=subset_data_loader_train, device=device, skip_first_layers=cfg.model.skip_first_layers
    )

    all_indices, pruning_time = measure_execution_time(selector, model)
    print("Global pruning ratio:", selector.global_pruning_ratio)

    if cfg.model.name.startswith("resnet"):
        all_indices = filter_pruning_indices_for_resnet(all_indices, cfg.model.name)

    

    for num, indices in enumerate(all_indices):
        print(f"Pruning ratio number {num}: {cfg.pruning.pruning_ratio[num]}")
        pruner = DepGraphPruner(
                model=model,
                indices=indices,
                replace_last_layer=cfg.replace_last_layer,
                selected_classes=cfg.selected_classes,
                device=device,
            )
        if cfg.pruning.pruning_ratio[num] > 0:
            pruned_model = pruner.prune()
            print("Model pruned successfully.")
        else:
            pruner._replace_last_layer()
            pruned_model = pruner.model
        # Evaluate the model before and after pruning
        print("Before pruning:")
        torch.cuda.empty_cache()
        model.to(device)
        pruned_model.to(device)
        _, class_accuracies_original, inference_time_before, inf_time_std_before = measure_inference_time_and_accuracy(
            subset_data_loader_test,
            model,
            device,
            cfg.training.batch_size_test,
            cfg.dataset.num_classes,
            all_classes=True,
            print_results=True,
            selected_classes=None,
            with_onnx=cfg.inference_with_onnx
        )
        print("After pruning:")
        _, class_accuracies_pruned, inference_time_after, inf_time_std_after = measure_inference_time_and_accuracy(
            subset_data_loader_test,
            pruned_model,
            device,
            cfg.training.batch_size_test,
            cfg.dataset.num_classes,
            all_classes=True,
            print_results=True,
            selected_classes=(
                cfg.selected_classes.copy() if cfg.replace_last_layer else None
            ),
            with_onnx=cfg.inference_with_onnx
        )
        accuracy_before = calculate_accuracy_for_selected_classes(
            class_accuracies_original, cfg.selected_classes
        )
        accuracy_after = calculate_accuracy_for_selected_classes(
            class_accuracies_pruned, cfg.selected_classes
        )

        print(f"Accuracy before pruning: {accuracy_before:.2f}%")
        print(f"Accuracy after pruning: {accuracy_after:.2f}%")
        
        retraining_time = 0
        if cfg.training.retrain_after_pruning:
            print("Retraining the pruned model...")
            if cfg.pruning.name == "torchpruner":
                subset_data_loader_train = subset_data_loader_train_retrain
            start = time.perf_counter()
            best_accuracy, best_epoch = train(
                cfg,
                pruned_model,
                subset_data_loader_train,
                subset_data_loader_test,
                device,
                retrain=True
            )
            end = time.perf_counter()
            retraining_time = end - start
            if cfg.log_results:
                wandb.log({
                    "best_accuracy_retraining": best_accuracy,
                    "best_epoch_retraining": best_epoch,
                })

        model_size_before = get_model_size(model)
        model_size_after = get_model_size(pruned_model)
        parameter_ratio = get_parameter_ratio(model, pruned_model)

        # Flop Analysis
        flops_before = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
        flops_after = FlopCountAnalysis(pruned_model, torch.randn(1, 3, 224, 224).to(device))
        print(f"FLOPs before pruning: {flops_before.total()/1e6} MFLOPs")
        print(f"FLOPs after pruning: {flops_after.total()/1e6} MFLOPs")
        print(f"FLOPs reduction ratio: {flops_after.total()/flops_before.total()}")

        print(f"Batch Inference time before pruning: {inference_time_before}")
        print(f"Batch Inference time after pruning: {inference_time_after}")
        print(f"Inference time ratio: {inference_time_after / inference_time_before}")

        print(f"Model size before pruning: {model_size_before} MB")
        print(f"Model size after pruning: {model_size_after} MB")
        print(f"Pruned parameters ratio: {1 - parameter_ratio}")
        if cfg.log_results:
            wandb.log(
                {
                    "accuracy_before": accuracy_before,
                    "accuracy_after": accuracy_after,
                    "model_size_before": model_size_before,
                    "model_size_after": model_size_after,
                    "model_size_ratio": model_size_after / model_size_before,
                    "parameter_ratio": parameter_ratio,
                    "gloabal_pruning_ratio": selector.global_pruning_ratio,
                    "class_accuracies_original": class_accuracies_original,
                    "class_accuracies_pruned": class_accuracies_pruned,
                    "inference_time_batch_before": inference_time_before,
                    "inference_time_batch_after": inference_time_after,
                    "inference_time_ratio": inference_time_after / inference_time_before,
                    "inference_time_per_sample_before": inference_time_before
                    / cfg.training.batch_size_test,
                    "inference_time_per_sample_after": inference_time_after,
                    "pruned_parameters": 1 - parameter_ratio,
                    "flops_before": flops_before.total(),
                    "flops_after": flops_after.total(),
                    "flops_ratio": flops_after.total() / flops_before.total(),
                    "pruning_time": pruning_time,
                    "retraining_time": retraining_time,
                    "total_time": pruning_time + retraining_time,
                    "inference_time_std_before": inf_time_std_before,
                    "inference_time_std_after": inf_time_std_after
                }
            )

    if cfg.log_results:
        wandb.finish()


if __name__ == "__main__":
    main()
