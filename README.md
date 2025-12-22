# Class Aware Pruning


## Project Overview

This project is a benchmarking framework for comparing class-aware and traditional pruning methods on convolutional neural networks (CNNs). I implemented this for my master's thesis on benchmarking class-aware pruning techniques. The pruning algorithms can be evaluated on different datasets (CIFAR10, ImageNet, GTSRB) and model architectures (VGG16, ResNet18) by measuring accuracy, inference time and model size before and after pruning.


## Architecture & Data Flow

### Core Pipeline (main.py → main())

1. **Configuration Loading**: Hydra config system loads dataset, model, pruning, and training parameters from `config/`
2. **Model Loading**: `models.get_model()` retrieves torchvision models and adjusts output layers
3. **Data Preparation**: `DataLoaderFactory` subclasses (CIFAR10, ImageNet, GTSRB) create train/test/pruning subset loaders
4. **Selection Phase**: `selection.get_selector()` factory returns pruning strategy instance that selects filters
5. **Pruning Phase**: `DepGraphPruner` applies selected pruning of selected filter indices via torch.fx symbolic tracing
6. **Evaluation**: `metrics.measure_inference_time_and_accuracy()` measures per-class accuracy and inference time



### Pruning Selection Strategies

All pruning strategies inherit from `PruningSelection` abstract base and implement `select(model)` → dict of layer masks/indices:

- **OCAP** (`ocap.py`): Computes activation statistics via forward hooks, applies ratios per-layer (class-aware) -> based on https://github.com/mzd2222/OCAP
- **LRP** (`lrp.py`): Layer-wise Relevance Propagation; uses backpropagation to compute relevance scores (class-aware) -> based on https://github.com/seulkiyeom/LRP_Pruning_toy_example
- **LnStructured**: Prunes by layer norm magnitude (not class-aware)
- **TorchPruner**: Wraps torch_pruning library with Taylor/APoZ attribution metrics (not class-aware) --> used https://github.com/marcoancona/TorchPruner


## Development Workflows


### Pruning a Pretrained Model
Example: Pruning a VGG16 model trained on CIFAR10 with OCAP to 85% pruning ratio for classes 0,1,2: 
```bash
python main.py model=vgg16 dataset=cifar10 training.train=false model.pretrained_weights_path=<PATH_TO_PRETRAINED_WEIGHTS>\
pruning=ocap \
pruning.pruning_ratio="[0.85]" selected_classes=[0,1,2] 
```

Further parameters can be adjusted in the config files or via CLI overrides.

### Evaluation
All metrics are printed to console. If log_results=true, results are also saved logged to Weights & Biases.

## Project-Specific Patterns & Conventions

### Hydra Configuration
- **Override pattern**: CLI args override YAML (e.g., `training.retrain_after_pruning=true`)
- **Config locations**: `config/{pruning,model,dataset}/*.yaml` + `config/config.yaml` base


### Pruning
- **Skip early layers**: `cfg.model.skip_first_layers` bypasses pruning first N conv layers as these are critical for feature extraction
- **Dealing with Skip Connection in ResNet**: `filter_pruning_indices_for_resnet()` in `helpers.py` is called for ResNets to ensure compatible pruning of skip connections (we don't prune the last conv layer in a block). If other archtitectures with skip connections are used, similar logic must be implemented
- **Last Layer Replacement**: When `replace_last_layer=true`, linear output layer is replaced so output dimension matches number of `selected_classes`


### Determinism & Reproducibility
- **Data augmentation disabled by default**: `use_data_augmentation=false` in config for consistent pruning results
- **Shuffle disabled in pruning loaders**: `get_subset_dataloaders()` sets `shuffle=False`
- **Seeding**: `random.seed(42)` in `get_small_train_loader()`; use explicit seed control for full reproducibility


### Pruning Ratios Configuration
- **Multiple ratios**: `cfg.pruning.pruning_ratio` is a list; main.py loops over each ratio, creating separate pruned models. We do this to benchmark multiple pruning levels in one run. Otherwise there can be small variations in accuracy between the runs.
- **Ratio semantics**: Fraction of filters pruned per layer (0.85 = prune 85%, keep 15%)
- **Example**: `pruning_ratio: [0.00, 0.85, 0.88, 0.90, ...]` produces 11 pruned models

### ONNX Inference Option
- **Flag**: `cfg.inference_with_onnx=true` converts model to ONNX and benchmarks via onnxruntime
- **Purpose**: Measure real-world inference speed on CPU/CPU platforms
- **Caveat**: Not all custom layers supported; falls back to PyTorch if conversion fails

## Key Files & Their Roles

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates full pipeline: training → selection → pruning → retraining → evaluation |
| `selection.py` | Abstract base `PruningSelection` and all filter selection algorithms |
| `pruner.py` | `DepGraphPruner` applies selected indices via torch.fx symbolic trace and `StructurePruner` implements a similar logic. |
| `models.py` | Model factory; handles torchvision load, last-layer replacement |
| `data_loader.py` | `DataLoaderFactory` subclasses for CIFAR10/ImageNet/GTSRB |
| `metrics.py` | Accuracy, inference time, parameter ratio, FLOP counting |
| `config/` | Hydra YAML configs (base, pruning strategies, models, datasets) |

