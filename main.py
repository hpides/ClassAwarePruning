import time
import torch
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from data_loader import dataloaderFactories
from pruner import DepGraphPruner, UnstructuredMagnitudePruner
from selection import get_selector
from models import get_model
import wandb
from fvcore.nn import FlopCountAnalysis
from distillation import KnowledgeDistillation
from metrics import (
    get_parameter_ratio,
    get_model_size,
)
from helpers import (
    train,
    filter_pruning_indices_for_resnet,
    get_unstructured_sparsity,
    run_pruner,
    evaluate
)


###################################
# ------------- MAIN ------------ #
###################################

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Hardware initialization
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if cfg.device:
        device = torch.device(cfg.device)
    print(f"%%%%%% Using device: {device}")

    # WandB Setup
    if cfg.log_results:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_cfg["device"] = device
        wandb.init(
            project="ClassAwarePruning",
            entity="sjoze",
            config=wandb_cfg,
            name=cfg.run_name,
        )
    print("%%%%%% Initialized WandB")

    # Force pruning_ratio to always be a list (Hydra turns single item lists to floats)
    if not isinstance(cfg.pruning.pruning_ratio, (list, ListConfig)):
        cfg.pruning.pruning_ratio = [cfg.pruning.pruning_ratio]


    ###################################
    # --------- DATALOADERS --------- #
    ###################################

    dataloader_factory = dataloaderFactories[cfg.dataset.name](
        train_batch_size=cfg.training.batch_size_train,
        test_batch_size=cfg.training.batch_size_test,
        selected_classes=cfg.selected_classes,
        num_pruning_samples=cfg.num_pruning_samples,
        use_data_augmentation=cfg.training.use_data_augmentation,
        use_imagenet_labels=cfg.dataset.use_imagenet_labels if "use_imagenet_labels" in cfg.dataset else False,
        subsample_ratio=cfg.dataset.subsample_ratio if "subsample_ratio" in cfg.dataset else None,
        subsample_size_per_class=cfg.dataset.subsample_size_per_class if "subsample_size_per_class" in cfg.dataset else None,
    )
    train_loader, val_loader, test_loader = dataloader_factory.get_dataloaders()

    print(f"\n%%%%%% {"=" * 80}")
    print("%%%%%% DATALOADER INFO")
    print(f"%%%%%% {"=" * 80}")
    print(f"%%%%%% Train loader length: {len(train_loader)}")
    print(f"%%%%%% Train dataset size: {len(train_loader.dataset)}")
    print(f"%%%%%% Train batch size: {train_loader.batch_size}")
    print(f"%%%%%% Val loader length: {len(val_loader)}")
    print(f"%%%%%% Val dataset size: {len(val_loader.dataset)}")
    print(f"%%%%%% Test loader length: {len(test_loader)}")
    print(f"%%%%%% Test dataset size: {len(test_loader.dataset)}")
    print(f"%%%%%% {"=" * 80}\n")

    # Subset dataloader creation
    subset_data_loader_train, subset_data_loader_val, subset_data_loader_test, pruning_dataloader = dataloader_factory.get_subset_dataloaders()

    print(f"\n%%%%%% {"=" * 80}")
    print("%%%%%% SUBSET DATALOADER INFO")
    print(f"%%%%%% {"=" * 80}")
    print(f"%%%%%% Subset train loader length: {len(subset_data_loader_train)}")
    print(f"%%%%%% Subset train dataset size: {len(subset_data_loader_train.dataset)}")
    print(f"%%%%%% Subset val dataset size: {len(subset_data_loader_val.dataset)}")
    print(f"%%%%%% Subset test dataset size: {len(subset_data_loader_test.dataset)}")
    print(f"%%%%%% {"=" * 80}\n")

    model = get_model(
        cfg.model.name, pretrained=cfg.training.use_pretrained_model, num_classes=cfg.dataset.num_classes,
        dataset_name=cfg.dataset.name
    )
    model.to(device)

    print(f"%%%%%% GPU memory after loading base model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"%%%%%% Loaded Model - Pretrained: {cfg.training.use_pretrained_model}, "
          f"Number of classes: {cfg.dataset.num_classes}, Dataset: {cfg.dataset.name}")

    # Train the model or load pretrained weights
    if cfg.training.train and not cfg.training.use_pretrained_model:
        print("%%%%%% Training Model")
        train(cfg, model, train_loader, test_loader, device)
        print("%%%%%% Done Training")
    elif cfg.training.use_pretrained_model and cfg.dataset.name == "imagenet":
        print("%%%%%% Using torchvision ImageNet pretrained weights")
    elif cfg.model.pretrained_weights_path:
        print(f"%%%%%% Loading custom weights from {cfg.model.pretrained_weights_path}")
        model.load_state_dict(torch.load(cfg.model.pretrained_weights_path, weights_only=True, map_location=device))
        print("%%%%%% Loaded custom weights")
    else:
        print("%%%%%% Using randomly initialized model")


    ###################################
    # ------- FILTER SELECTION ------ #
    ###################################

    selection_time, removal_time = 0, 0

    # Mapping indices (e.g. [105, 305, 402] to the new output nodes [0, 1, 2])
    mapping = {new_idx: orig_class for new_idx, orig_class in enumerate(cfg.selected_classes)}
    print(f"%%%%%% Mapping of classes: {mapping}")

    # Specifically for OCAP where we need less pruning samples (~25 per class according to authors)
    if pruning_dataloader is not None:
        print(f"%%%%%% Smaller training dataloader in use - Originally: {len(subset_data_loader_train.dataset)}, "
              f"Now using: {len(pruning_dataloader.dataset)}")

    # ----- STRUCTURED FILTER SELECTION -----
    if not cfg.pruning.name.startswith("unstructured_"):
        selector = get_selector(
            selector_config=cfg.pruning,
            data_loader=pruning_dataloader if pruning_dataloader is not None else subset_data_loader_train,
            device=device,
            skip_first_layers=cfg.model.skip_first_layers
        )
        print(f"%%%%%% Loaded selector: {selector}")

        start = time.perf_counter()
        all_indices = selector.select(model)
        selection_time = time.perf_counter() - start

        print(f"%%%%%% Time spent on selecting indices: {selection_time}")
        print(f"%%%%%% Global pruning ratio: {selector.global_pruning_ratio}")

        if cfg.model.name.startswith("resnet"):
            all_indices = filter_pruning_indices_for_resnet(all_indices, cfg.model.name)

        for num, indices in enumerate(all_indices):
            print(f"%%%%%% Pruning ratio number {num}: {cfg.pruning.pruning_ratio[num]}")
            pruner = DepGraphPruner(
                model=model,
                indices=indices,
                replace_last_layer=cfg.replace_last_layer,
                selected_classes=cfg.selected_classes,
                device=device,
            )
            # Filter Removal
            pruned_model, removal_time = run_pruner(pruner, cfg.pruning.pruning_ratio[num])

    # ----- UNSTRUCTURED FILTER SELECTION -----
    else:
        # Dont need to make a distinction between selection and pruning, since we don't need to remove filters.
        # We can simply set to zero.
        pruner = UnstructuredMagnitudePruner(
            model=model,
            sparsity=cfg.pruning.pruning_ratio[0],
            replace_last_layer=cfg.replace_last_layer,
            selected_classes=cfg.selected_classes,
            device=device
        )
        # Filter Removal
        pruned_model, removal_time = run_pruner(pruner, cfg.pruning.pruning_ratio[0])

    pruning_time = selection_time + removal_time
    print(f"%%%%%% Time spent on pruning: {pruning_time} = filter selection ({selection_time} + filter removal {removal_time})")
    print(f"%%%%%% GPU memory after pruning model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"%%%%%% Model architecture before:\n{model}")
    print(f"%%%%%% Model architecture after:\n{pruned_model}")

    model_size_before = get_model_size(model)
    print(f"%%%%%% Model size before pruning: {model_size_before} MB")
    model_size_after = get_model_size(pruned_model)
    print(f"%%%%%% Model size after pruning: {model_size_after} MB")

    ###################################
    # ---------- EVALUATION --------- #
    ###################################

    torch.cuda.empty_cache()
    model.to(device)
    pruned_model.to(device)

    print(f"%%%%%% GPU memory before starting evaluation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ----- BEFORE PRUNING (BASE) -----
    accuracy_before, inference_time_before, _ = evaluate(
        model, subset_data_loader_test, cfg, device, 0, mapping, label="Before pruning", is_pruned=False
    )

    # ----- AFTER PRUNING -----
    accuracy_after, inference_time_after, _ = evaluate(
        pruned_model, subset_data_loader_test, cfg, device, inference_time_before, mapping,
        label="After pruning"
    )

    # ----- KNOWLEDGE DISTILLATION SETUP -----
    if cfg.use_knowledge_distillation:
        cfg.pruning.name = "unstructured_magnitude"
        cfg.pruning.pruning_ratio = [get_parameter_ratio(model, pruned_model)]
        unstr_pruner = UnstructuredMagnitudePruner(
            model=model,
            sparsity=cfg.pruning.pruning_ratio[0],
            replace_last_layer=cfg.replace_last_layer,
            selected_classes=cfg.selected_classes,
            device=device
        )
        # Structured pruned model becomes the student and unstructured pruned model becomes the teacher, which
        # will get trained before it distills its knowledge into the student, hence the swap in
        student_model = pruned_model
        pruned_model, _ = run_pruner(unstr_pruner, cfg.pruning.pruning_ratio[0])
        pruned_model = pruned_model.to(device)

    # ----- RETRAINING -----
    retraining_time = 0
    best_accuracy, best_epoch, accuracy_after_retraining, inference_time_ratio_retraining = None, None, None, None
    if cfg.training.retrain_after_pruning:
        print("%%%%%% Retraining the pruned model...")
        print(f"+++++ GPU memory right before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        start = time.perf_counter()
        best_accuracy, best_epoch = train(
            cfg, pruned_model,
            subset_data_loader_train, subset_data_loader_val,
            device, num_epochs=cfg.training.retrain_epochs, retrain=True
        )
        retraining_time = time.perf_counter() - start
        accuracy_after_retraining, _, inference_time_ratio_retraining = evaluate(
            pruned_model, subset_data_loader_test, cfg, device, inference_time_before, mapping,
            label="After Retraining"
        )

    # ----- KNOWLEDGE DISTILLATION TRAINING -----
    if cfg.use_knowledge_distillation:
        kd = KnowledgeDistillation(
            teacher_model=pruned_model, student_model=student_model,
            selected_classes=cfg.selected_classes,
            temperature=3.0, alpha=0.7, lr=1e-3
        )
        student_model, _, _ = kd.train(
            train_loader=subset_data_loader_train,
            val_loader=subset_data_loader_val,
            epochs=100
        )
        accuracy_kd, _, _ = evaluate(
            student_model, subset_data_loader_test, cfg, device, inference_time_before, mapping,
            label="After Knowledge Distillation"
        )


    ###################################
    # ----------- LOGGING ----------- #
    ###################################

    # Model parameters
    model_size_before = get_model_size(model)
    print(f"%%%%%% Model size before pruning: {model_size_before} MB")
    model_size_after = get_model_size(pruned_model)
    print(f"%%%%%% Model size after pruning: {model_size_after} MB")
    parameter_ratio = get_parameter_ratio(model, pruned_model)
    print(f"%%%%%% Pruned parameters ratio: {1 - parameter_ratio}")
    global_pruning_ratio = None if cfg.pruning.name.startswith("unstructured_") else selector.global_pruning_ratio

    # Flop Analysis
    flops_before = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
    flops_after = FlopCountAnalysis(pruned_model, torch.randn(1, 3, 224, 224).to(device))
    print(f"%%%%%% FLOPs before pruning: {flops_before.total() / 1e6} MFLOPs")
    print(f"%%%%%% FLOPs after pruning: {flops_after.total() / 1e6} MFLOPs")
    print(f"%%%%%% FLOPs reduction ratio: {flops_after.total() / flops_before.total()}")

    print(f"%%%%%% Batch Inference time before pruning: {inference_time_before}")
    print(f"%%%%%% Batch Inference time after pruning: {inference_time_after}")
    inference_time_ratio = (inference_time_after / inference_time_before) if inference_time_before > 0 else 0
    print(f"%%%%%% Inference time ratio: {inference_time_ratio}")

    if cfg.pruning.name.startswith("unstructured_"):
        sparsity_info = get_unstructured_sparsity(pruned_model)
        print(f"%%%%%% Actual global sparsity: {sparsity_info['global']:.4f}")
        if cfg.log_results:
            wandb.log({
                "actual_sparsity": sparsity_info['global'],
                "layer_sparsity": sparsity_info
            })
    if cfg.log_results:
        wandb.log(
            {
                "accuracy_before": accuracy_before,
                "accuracy_after": accuracy_after,
                "model_size_before": model_size_before,
                "model_size_after": model_size_after,
                "model_size_ratio": model_size_after / model_size_before,
                "parameter_ratio": parameter_ratio,
                "gloabal_pruning_ratio": global_pruning_ratio,
                "inference_time_batch_before": inference_time_before,
                "inference_time_batch_after": inference_time_after,
                "inference_time_ratio": inference_time_ratio,
                "inference_time_per_sample_before": inference_time_before
                                                    / cfg.training.batch_size_test,
                "inference_time_per_sample_after": inference_time_after,
                "pruned_parameters": 1 - parameter_ratio,
                "flops_before": flops_before.total(),
                "flops_after": flops_after.total(),
                "flops_ratio": flops_after.total() / flops_before.total(),
                "selection_time": selection_time,
                "removal_time": removal_time,
                "pruning_time": pruning_time,
                "retraining_time": retraining_time,
                "total_time": pruning_time + retraining_time,
                "best_accuracy_retraining": best_accuracy,
                "best_epoch_retraining": best_epoch,
                "accuracy_after_retraining": accuracy_after_retraining,
                "inference_time_ratio_retraining": inference_time_ratio_retraining,
            }
        )

    if cfg.log_results:
        wandb.finish()


if __name__ == "__main__":
    main()