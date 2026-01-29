import time
import torch
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from data_loader import dataloaderFactories
from pruner import DepGraphPruner, UnstructuredPruner
from selection import get_selector
from models import get_model
import wandb
from fvcore.nn import FlopCountAnalysis
from distillation import KnowledgeDistillation
from metrics import (
    get_parameter_ratio,
    measure_execution_time,
    get_model_size,
    measure_inference_time_and_accuracy,
    calculate_accuracy_for_selected_classes,
)
from helpers import (
    train,
    get_optimizer,
    filter_pruning_indices_for_resnet,
    get_unstructured_sparsity
)


###################################
# ------------- MAIN ------------ #
###################################

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ----- Hardware initialization
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if cfg.device:
        device = torch.device(cfg.device)
    print(f"@@@@@ Using device: {device}")

    # ----- WandB setup
    if cfg.log_results:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_cfg["device"] = device
        wandb.init(
            project="ClassAwarePruning",
            entity="sjoze",
            config=wandb_cfg,
            name=cfg.run_name,
        )
    print("@@@@@ Initialized WandB")

    # ----- Fix pruning_ratio to always be a list
    if not isinstance(cfg.pruning.pruning_ratio, (list, ListConfig)):
        print("##### Pruning ratio is not a list")
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
    train_loader, test_loader = dataloader_factory.get_dataloaders()

    print("\n" + "=" * 60)
    print("@@@@@ DATALOADER INFO")
    print("=" * 60)
    print(f"@@@@@ Train loader length: {len(train_loader)}")
    print(f"@@@@@ Train dataset size: {len(train_loader.dataset)}")
    print(f"@@@@@ Train batch size: {train_loader.batch_size}")
    print(f"@@@@@ Total train batches: {len(train_loader)}")
    print(f"@@@@@ Test loader length: {len(test_loader)}")
    print(f"@@@@@ Test dataset size: {len(test_loader.dataset)}")
    print("=" * 60 + "\n")

    # ----- Subset dataloader creation
    subset_data_loader_train, subset_data_loader_test = dataloader_factory.get_subset_dataloaders()
    #if cfg.pruning.name == "torchpruner":
    #    subset_data_loader_train_retrain = subset_data_loader_train
    #    subset_data_loader_train = dataloader_factory.get_small_train_loader()

    print("\n" + "=" * 60)
    print("@@@@@ SUBSET DATALOADER INFO")
    print("=" * 60)
    print(f"@@@@@ Subset train loader length: {len(subset_data_loader_train)}")
    print(f"@@@@@ Subset train dataset size: {len(subset_data_loader_train.dataset)}")
    print(f"@@@@@ Subset test dataset size: {len(subset_data_loader_test.dataset)}")
    print("=" * 60 + "\n")

    print(f"+++++ GPU memory after getting dataloaders: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = get_model(
        cfg.model.name, pretrained=cfg.training.use_pretrained_model, num_classes=cfg.dataset.num_classes,
        dataset_name=cfg.dataset.name
    )
    model.to(device)

    print(f"++++++ GPU memory after loading base model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"@@@@@ Loaded Model - Pretrained: {cfg.training.use_pretrained_model}, "
          f"Number of classes: {cfg.dataset.num_classes}, Dataset: {cfg.dataset.name}")

    # ----- Train the model or load pretrained weights
    if cfg.training.train and not cfg.training.use_pretrained_model:
        print("@@@@@ Training Model")
        train(cfg, model, train_loader, test_loader, device)
        print("@@@@@ Done Training")
    elif cfg.training.use_pretrained_model and cfg.dataset.name == "imagenet":
        print("@@@@@ Using torchvision ImageNet pretrained weights")
    elif cfg.model.pretrained_weights_path:
        print(f"@@@@@ Loading custom weights from {cfg.model.pretrained_weights_path}")
        model.load_state_dict(torch.load(cfg.model.pretrained_weights_path, weights_only=True, map_location=device))
        print("@@@@@ Loaded custom weights")
    else:
        print("@@@@@ Using randomly initialized model")


    ###################################
    # ----------- SELECTOR ---------- #
    ###################################

    mapping = {new_idx: orig_class for new_idx, orig_class in enumerate(sorted(cfg.selected_classes))}
    print(f"@@@@@ Mapping of classes: {mapping}")

    selector = get_selector(
        selector_config=cfg.pruning, data_loader=subset_data_loader_train, device=device,
        skip_first_layers=cfg.model.skip_first_layers
    )
    print(f"@@@@@ Loaded selector: {selector}")

    all_indices, pruning_time = measure_execution_time(selector, model)
    print(f"@@@@@ Time spent on pruning: {pruning_time}")
    print(f"@@@@@ Global pruning ratio: {selector.global_pruning_ratio}")

    if cfg.model.name.startswith("resnet"):
        all_indices = filter_pruning_indices_for_resnet(all_indices, cfg.model.name)

    for num, indices in enumerate(all_indices):
        print(f"@@@@@ Pruning ratio number {num}: {cfg.pruning.pruning_ratio[num]}")
        if cfg.pruning.name.startswith("unstructured_"):
            # Unstructured pruning: indices are actually masks
            pruner = UnstructuredPruner(
                model=model,
                masks=indices,
                replace_last_layer=cfg.replace_last_layer,
                selected_classes=cfg.selected_classes,
                device=device,
            )
        else:
            # Structured pruning: indices are filter indices
            pruner = DepGraphPruner(
                model=model,
                indices=indices,
                replace_last_layer=cfg.replace_last_layer,
                selected_classes=cfg.selected_classes,
                device=device,
            )
        if cfg.pruning.pruning_ratio[num] > 0:
            pruned_model = pruner.prune()
            print("@@@@@ Model pruned successfully.")
        else:
            pruner._replace_last_layer()
            pruned_model = pruner.model

        print(f"++++++ GPU memory after pruning model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


        '''
        import csv
        import os
        parameter_ratio = get_parameter_ratio(model, pruned_model)
        pruned_param_ratio = 1 - parameter_ratio
        print(f"++++++++++++++++++++++++++++++++++++")
        print(f"Pruning ratio {cfg.pruning.pruning_ratio[num]} is equivalent to {pruned_param_ratio} pruned parameters.")
        print(f"++++++++++++++++++++++++++++++++++++")

        # Log to CSV
        csv_file = "pruning_ratios.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists:
                writer.writerow(['model', 'pruning_method', 'dataset', 'target_pruning_ratio', 'actual_pruned_parameters'])

            # Write data
            writer.writerow([
                cfg.model.name,
                cfg.pruning.name,
                cfg.dataset.name,
                cfg.pruning.pruning_ratio[num],
                pruned_param_ratio
            ])
        return
        '''

        print(f"@@@@@ Model architecture before:\n{model}")
        print(f"@@@@@ Model architecture after:\n{pruned_model}")


        ###################################
        # ---------- EVALUATION --------- #
        ###################################

        torch.cuda.empty_cache()
        model.to(device)
        pruned_model.to(device)

        print(f"++++++ GPU memory before starting evaluation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        print("***** Subset")
        print("\n" + "=" * 60)
        print("Train DataLoader:")
        print("=" * 60)
        #for batch_idx, (inputs, labels) in enumerate(subset_data_loader_train):
        #    print(f"Batch {batch_idx}: labels = {labels.tolist()}")
        print(f"Total: {len(subset_data_loader_train.dataset)} samples")

        print("\n" + "=" * 60)
        print("Test DataLoader:")
        print("=" * 60)
        #for batch_idx, (inputs, labels) in enumerate(subset_data_loader_test):
        #    print(f"Batch {batch_idx}: labels = {labels.tolist()}")
        print(f"Total: {len(subset_data_loader_test.dataset)} samples")
        print("=" * 60 + "\n")

        # ----- Base Model
        print("@@@@@ Before pruning:")


        _, class_accuracies_original, inference_time_before, inf_time_all_before = measure_inference_time_and_accuracy(
            subset_data_loader_test,
            model,
            device,
            cfg.training.batch_size_test,
            cfg.dataset.num_classes,
            all_classes=True,
            print_results=True,
            selected_classes=None,
            with_onnx=cfg.inference_with_onnx,
            mapping=mapping
        )
        accuracy_before = calculate_accuracy_for_selected_classes(
            class_accuracies_original, cfg.selected_classes
        )

        # ----- Pruned Model
        print("@@@@@ After pruning:")
        _, class_accuracies_pruned, inference_time_after, inf_time_all_after = measure_inference_time_and_accuracy(
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
            with_onnx=cfg.inference_with_onnx,
            mapping=mapping
        )
        accuracy_after = calculate_accuracy_for_selected_classes(
            class_accuracies_pruned, cfg.selected_classes
        )

        print(f"+++++ GPU memory after calculating base and pruned model accuracies: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        print(f"@@@@@ Accuracy before pruning: {accuracy_before:.2f}")
        print(f"@@@@@ Accuracy after pruning: {accuracy_after:.2f}")

        if cfg.use_knowledge_distillation:
            kd = KnowledgeDistillation(
                teacher_model=model,
                student_model=pruned_model,
                relevant_class_idxs=cfg.selected_classes,
                temperature=3.0,
                alpha=0.7,
                lr=1e-3
            )

            pruned_model = kd.train(
                train_loader=subset_data_loader_train,
                val_loader=subset_data_loader_test,
                epochs=20
            )


        retraining_time = 0
        best_accuracy, best_epoch, accuracy_after_retraining, inference_time_ratio_retraining = None, None, None, None
        if cfg.training.retrain_after_pruning:
            print("@@@@@ Retraining the pruned model...")
            #if cfg.pruning.name == "torchpruner":
            #    subset_data_loader_train = subset_data_loader_train_retrain
            start = time.perf_counter()

            if cfg.use_knowledge_distillation:
                pruned_model, best_accuracy, best_epoch = kd.train(
                    train_loader=subset_data_loader_train,
                    val_loader=subset_data_loader_test,
                    epochs=20
                )
            else:
                print(f"+++++ GPU memory right before going to training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
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

            print("After Retraining:")
            _, class_accuracies_pruned_r, inference_time_after_r, inf_time_all_after_r = measure_inference_time_and_accuracy(
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
                with_onnx=cfg.inference_with_onnx,
                mapping=mapping
            )

            accuracy_after_retraining = calculate_accuracy_for_selected_classes(
                class_accuracies_pruned_r, cfg.selected_classes
            )

            inference_time_ratio_retraining = (
                        inference_time_after_r / inference_time_before) if inference_time_before > 0 else 0

        model_size_before = get_model_size(model)
        model_size_after = get_model_size(pruned_model)
        parameter_ratio = get_parameter_ratio(model, pruned_model)

        # Flop Analysis
        flops_before = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
        flops_after = FlopCountAnalysis(pruned_model, torch.randn(1, 3, 224, 224).to(device))
        print(f"@@@@@ FLOPs before pruning: {flops_before.total() / 1e6} MFLOPs")
        print(f"@@@@@ FLOPs after pruning: {flops_after.total() / 1e6} MFLOPs")
        print(f"@@@@@ FLOPs reduction ratio: {flops_after.total() / flops_before.total()}")

        print(f"@@@@@ Batch Inference time before pruning: {inference_time_before}")
        print(f"@@@@@ Batch Inference time after pruning: {inference_time_after}")
        inference_time_ratio = (inference_time_after / inference_time_before) if inference_time_before > 0 else 0
        print(f"@@@@@ Inference time ratio: {inference_time_ratio}")

        print(f"@@@@@ Model size before pruning: {model_size_before} MB")
        print(f"@@@@@ Model size after pruning: {model_size_after} MB")
        print(f"@@@@@ Pruned parameters ratio: {1 - parameter_ratio}")
        if cfg.pruning.name.startswith("unstructured_"):
            sparsity_info = get_unstructured_sparsity(pruned_model)
            print(f"@@@@@ Actual global sparsity: {sparsity_info['global']:.4f}")
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
                    "gloabal_pruning_ratio": selector.global_pruning_ratio,
                    "class_accuracies_original": class_accuracies_original,
                    "class_accuracies_pruned": class_accuracies_pruned,
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
                    "pruning_time": pruning_time,
                    "retraining_time": retraining_time,
                    "total_time": pruning_time + retraining_time,
                    "inference_time_all_before": inf_time_all_before,
                    "inference_time_all_after": inf_time_all_after,
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