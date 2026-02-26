import subprocess
import itertools
import time
import sys
from typing import List, Dict, Any
from datetime import datetime
import numpy as np


###################################
# ----- CONSTANTS + HELPERS ----- #
###################################

similar_classes =  [724,  # pirate, pirate ship
                    628,  # liner, ocean liner
                    484,  # catamaran
                    576,  # gondola
                    625,  # lifeboat
                    814,  # speedboat
                    472,  # canoe
                    780,  # schooner
                    554,  # fireboat
                    871]  # trimaran

distinct_classes = [985,  # daisy
                    483,  # castle
                    812,  # space shuttle
                    924,  # guacamole
                    107,  # jellyfish
                    398,  # abacus
                    970,  # alp
                    51,   # triceratops
                    145,  # king penguin
                    916]  # web site

standard_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 500,
        "log_results": True,
    }


def evenly_spaced_prefixes(num_samples, total_classes=1000):
    """
    Deterministically samples from a dataset with even spacing.
    E.g., for five classes it would return [0, 249, 499, 749, 999].

    Args:
        num_samples (List[int]): List of number of samples you need from the dataset.
        total_classes (int): How many classes are in the dataset in total, default 1000 for ImageNet.

    Returns:
        List[int]: List of indices of selected classes
    """
    max_n = max(num_samples)
    # Build a single evenly spaced ordering
    base = np.linspace(0, total_classes - 1, max_n, dtype=int)
    # Nested prefixes (like range(100, 100+n))
    return [base[:n].tolist() for n in num_samples]


class ExperimentLauncher:
    def __init__(self, base_config: Dict[str, Any] = None):
        """
        Class for launching multiple experiments with different configurations.

        Args:
            base_config (dict): Configuration applied to all experiments.
        """
        self.base_config = base_config or {}
        self.experiments = []

    def add_experiment(self, config: Dict[str, Any], name: str = None):
        """
        Adds a single experiment configuration.

        Args:
            config (dict): Dictionary of configuration parameters.
            name (str): Optional name for the experiment.
        """
        self.experiments.append({
            "config": {**self.base_config, **config},
            "name": name or f"exp_{len(self.experiments) + 1}"
        })

    def add_sweep(
            self,
            param_grid: Dict[str, List[Any]],
            name_template: str = None
    ):
        """
        Adds a parameter sweep (Cartesian product of all parameters).

        Args:
            param_grid (dict): Dictionary mapping parameter names to lists of values
            name_template (str): Template for run names, e.g., "{pruning}_{model}".
                Can use any parameter name in curly braces.
                Example:
                    launcher.add_sweep(
                        param_grid={
                            "pruning": ["ocap", "ln_structured"],
                            "model": ["vgg16", "resnet18"]
                        },
                        name_template="{pruning}_{model}"
                    )
                    Creates 4 experiments: ocap_vgg16, ocap_resnet18, ln_structured_vgg16, ln_structured_resnet18
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))

            # Generate name from template
            if name_template:
                try:
                    # Handle list parameters in name template
                    name = name_template
                    for k, v in config.items():
                        if isinstance(v, list):
                            placeholder = "{" + k + "}"
                            name = name.replace(placeholder, f"{len(v)}items")
                        else:
                            placeholder = "{" + k + "}"
                            name = name.replace(placeholder, str(v))
                except Exception:
                    name = "_".join(f"{k}={v}" for k, v in config.items())
            else:
                name = "_".join(f"{k}={v}" for k, v in config.items())

            self.add_experiment(config, name)


    def run_all(self, dry_run: bool = True, continue_on_error: bool = True):
        """
        Executes all experiments.

        Args:
            dry_run (boolean): If True, only print commands without executing.
            continue_on_error (boolean): If True, continue with next experiment on failure.
        """
        print(f"\n%%%%%% {"=" * 80}")
        print(f"%%%%%% EXPERIMENT LAUNCHER")
        print(f"%%%%%% {"=" * 80}")
        print(f"%%%%%% Total experiments: {len(self.experiments)}")
        print(f"%%%%%% Mode: {"DRY RUN (preview only)" if dry_run else "EXECUTE"}")
        print(f"%%%%%% {"=" * 80}\n")

        results = []
        start_time = time.time()

        for i, exp in enumerate(self.experiments, 1):
            config = exp["config"]
            name = exp["name"]

            print(f"\n{"─" * 80}")
            print(f"%%%%%% [{i}/{len(self.experiments)}] {name}")
            print(f"{"─" * 80}")

            # Build command
            cmd = ["python", "main.py"]

            for key, value in config.items():
                print(f"{key} : {value}")
                if isinstance(value, list):
                    value_str = f"[{",".join(map(str, value))}]"
                elif isinstance(value, bool):
                    value_str = str(value).lower()
                else:
                    value_str = str(value)
                print(value_str)

                cmd.append(f"{key}={value_str}")

            # Add run name
            cmd.append(f"run_name={name}")

            print(f"%%%%%% Command: {" ".join(cmd)}")

            if not dry_run:
                exp_start = time.time()
                try:
                    sys.stdout.flush()  # Flush before subprocess
                    sys.stderr.flush()
                    subprocess.run(cmd, check=True)
                    sys.stdout.flush()  # Flush after subprocess
                    sys.stderr.flush()

                    exp_time = time.time() - exp_start
                    print(f"\n%%%%%% Completed in {exp_time:.1f}s")
                    results.append({"name": name, "status": "success", "time": exp_time})
                except subprocess.CalledProcessError as e:
                    exp_time = time.time() - exp_start
                    print(f"\nxxxxx Failed after {exp_time:.1f}s: {e}")
                    results.append({"name": name, "status": "failed", "time": exp_time})

                    if not continue_on_error:
                        print("\nxxxxx Stopping due to error.")
                        break
            else:
                print("%%%%%% (dry run - not executed)\n")
                results.append({"name": name, "status": "skipped", "time": 0})

        # Summary
        total_time = time.time() - start_time
        print(f"\n%%%%%% {"=" * 80}")
        print(f"%%%%%% SUMMARY")
        print(f"%%%%%% {"=" * 80}")
        print(f"%%%%%% Total experiments: {len(self.experiments)}")

        if not dry_run:
            success = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "failed")
            print(f"%%%%%% Successful: {success}")
            print(f"%%%%%% Failed: {failed}")
            print(f"%%%%%% Total time: {total_time / 60:.1f} minutes")

            if failed > 0:
                print(f"\n%%%%%% Failed experiments:")
                for r in results:
                    if r["status"] == "failed":
                        print(f"%%%%%%   - {r["name"]}")

        print(f"%%%%%% {"=" * 80}\n")


###################################
# --------- EXPERIMENTS --------- #
###################################

def experiment_kd(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - KD")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "use_knowledge_distillation": [True],
            "pruning": ["ln_structured"],
            "model": ["resnet18"],
            "selected_classes": [similar_classes, distinct_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="KD_{use_knowledge_distillation}_{model}_{pruning.pruning_ratio}_KD_" + timestamp
    )


def experiment_batchsize(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Batchsize")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 300,
        "selected_classes": [10, 20, 30, 40],
        "log_results": True,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured", "torchpruner"],
            "model": ["vgg16", "resnet18"],
            "pruning.pruning_ratio": [0.3, 0.5, 0.9],
            "training.batch_size_train": [128, 64],
        },
        name_template="BATCH_{pruning}_{training.batch_size_train}_" + timestamp
    )


def experiment_ocap_pruning_samples(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - OCAP Pruning Samples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 500,
        "selected_classes": [0, 111, 222, 333, 444, 555, 666, 777, 888, 999],
        "log_results": True,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ocap"],
            "model": ["resnet18"],
            "num_pruning_samples": [10, 20, 30, 50, 100, 200, 500],
            "pruning.pruning_ratio": [0.9, 0.93, 0.97, 0.99, 0.993, 0.997, 0.999, 0.9993,  0.9997,  0.9999],
        },
        name_template="OCAP_Pruning_Samples_PS{num_pruning_samples}_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


###################################
# -----  PRIMARY EXPERIMENTS ---- #
###################################

def experiment_unstructured_distinct(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Unstructured Distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["unstructured_magnitude"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [distinct_classes],
            "selected_classes": [distinct_classes],
            "pruning.pruning_ratio": [0.0, 0.33, 0.67, 0.9, 0.95, 0.97, 0.99],
        },
        name_template="UNS_DISTINCT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_unstructured_similar(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Unstructured Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["unstructured_magnitude"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [similar_classes],
            "pruning.pruning_ratio": [0.0, 0.33, 0.67, 0.9, 0.95, 0.97, 0.99],
        },
        name_template="UNS_SIMILAR_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_unstructured_amount(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Unstructured Amount")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["unstructured_magnitude"],
            "model": ["vgg16", "resnet18"],
            #"selected_classes": evenly_spaced_prefixes([3, 10, 30]),
            "selected_classes": evenly_spaced_prefixes([50]),
            "pruning.pruning_ratio": [0.0, 0.33, 0.67, 0.9, 0.95, 0.97, 0.99],
        },
        name_template="UNS_AMOUNT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_ln_struct_distinct(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - L1-Norm Distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [distinct_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="LNS_DISTINCT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_ln_struct_similar(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - L1-Norm Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [similar_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="LNS_SIMILAR_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_ln_struct_amount(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - L1-Norm Amount")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured"],
            "model": ["vgg16", "resnet18"],
            #"selected_classes": evenly_spaced_prefixes([3, 10, 30]),
            "selected_classes": evenly_spaced_prefixes([50]),
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="LNS_AMOUNT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_lrp_distinct(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - LRP Distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["lrp"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [distinct_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="LRP_DISTINCT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_lrp_similar(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - LRP Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["lrp"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [similar_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="LRP_SIMILAR_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_lrp_amount(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - LRP Amount")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["lrp"],
            "model": ["vgg16", "resnet18"],
            #"selected_classes": evenly_spaced_prefixes([3, 10, 30]),
            "selected_classes": evenly_spaced_prefixes([50]),
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="LRP_AMOUNT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_taylor_distinct(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Taylor Distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["torchpruner"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [distinct_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="TRP_DISTINCT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_taylor_similar(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Taylor Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["torchpruner"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [similar_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="TRP_SIMILAR_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_taylor_amount(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Taylor Amount")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["torchpruner"],
            "model": ["vgg16", "resnet18"],
            #"selected_classes": evenly_spaced_prefixes([3, 10, 30]),
            "selected_classes": evenly_spaced_prefixes([50]),
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="TRP_AMOUNT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_ocap_distinct(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - OCAP Distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 500,
        "num_pruning_samples": 25,
        "log_results": True,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ocap"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [distinct_classes],
            "pruning.pruning_ratio": [0.9, 0.93, 0.97, 0.99, 0.993, 0.997, 0.999, 0.9993,  0.9997,  0.9999],
        },
        name_template="OCAP_DISTINCT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_ocap_similar(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - OCAP Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 500,
        "num_pruning_samples": 25,
        "log_results": True,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ocap"],
            "model": ["vgg16", "resnet18"],
            "selected_classes": [similar_classes],
            "pruning.pruning_ratio": [0.9, 0.93, 0.97, 0.99, 0.993, 0.997, 0.999, 0.9993,  0.9997,  0.9999],
        },
        name_template="OCAP_SIMILAR_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_ocap_amount(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - OCAP Amount")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 500,
        "num_pruning_samples": 25,
        "log_results": True,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ocap"],
            "model": ["vgg16", "resnet18"],
            #"selected_classes": evenly_spaced_prefixes([3, 10, 30]),
            "selected_classes": evenly_spaced_prefixes([50]),
            "pruning.pruning_ratio": [0.9, 0.93, 0.97, 0.99, 0.993, 0.997, 0.999, 0.9993,  0.9997,  0.9999],
        },
        name_template="OCAP_AMOUNT_{pruning}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_layer_sparsity(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Layer Sparsity")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    pruning_configs_same = {
        "ocap": 0.9845,
        "ln_structured": 0.489,
        "lrp": 0.348,
        "torchpruner": 0.375,
        "unstructured_magnitude": 0.5,
    }

    pruning_configs_diff = {
        "ocap": 0.99,
        "ln_structured": 0.489,
        "lrp": 0.348,
        "torchpruner": 0.375,
        "unstructured_magnitude": 0.5,
    }

    for method, ratio in pruning_configs_same.items():
        launcher.add_experiment(
            config={
                "pruning": method,
                "model": "resnet18",
                "pruning.pruning_ratio": ratio,
                "selected_classes": similar_classes,
            },
            name=f"LAYER_SIMILAR_resnet18_{method}_{ratio}" + timestamp
        )

    for method, ratio in pruning_configs_diff.items():
        launcher.add_experiment(
            config={
                "pruning": method,
                "model": "resnet18",
                "pruning.pruning_ratio": ratio,
                "selected_classes": distinct_classes,
            },
            name=f"LAYER_DISTINCT_resnet18_{method}_{ratio}" + timestamp
        )


def experiment_pruning_speed(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Pruning Speed")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    pruning_configs_same = {
        "unstructured_magnitude": 0.5,
        "ocap": 0.9845,
        "ln_structured": 0.489,
        "lrp": 0.348,
        "torchpruner": 0.375,
    }

    pruning_configs_diff = {
        "unstructured_magnitude": 0.5,
        "ocap": 0.99,
        "ln_structured": 0.489,
        "lrp": 0.348,
        "torchpruner": 0.375,
    }

    for method, ratio in pruning_configs_same.items():
        launcher.add_experiment(
            config={
                "pruning": method,
                "model": "resnet18",
                "pruning.pruning_ratio": ratio,
                "selected_classes": similar_classes,
            },
            name=f"SPEED_resnet18_{method}_{ratio}" + timestamp
        )

    for method, ratio in pruning_configs_diff.items():
        launcher.add_experiment(
            config={
                "pruning": method,
                "model": "resnet18",
                "pruning.pruning_ratio": ratio,
                "selected_classes": distinct_classes,
            },
            name=f"SPEED_resnet18_{method}_{ratio}" + timestamp
        )

    for method, ratio in pruning_configs_same.items():
        launcher.add_experiment(
            config={
                "pruning": method,
                "model": "vgg16",
                "pruning.pruning_ratio": ratio,
                "selected_classes": similar_classes,
            },
            name=f"SPEED_vgg16_{method}_{ratio}" + timestamp
        )

    for method, ratio in pruning_configs_diff.items():
        launcher.add_experiment(
            config={
                "pruning": method,
                "model": "vgg16",
                "pruning.pruning_ratio": ratio,
                "selected_classes": distinct_classes,
            },
            name=f"SPEED_vgg16_{method}_{ratio}" + timestamp
        )


def experiment_random_similar(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Random Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["random"],
            "model": ["resnet18", "vgg16"],
            "selected_classes": [similar_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="RANDOM_SIMILAR_{pruning}_{pruning.pruning_ratio}" + timestamp
    )


def experiment_random_distinct(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Random Distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = standard_config

    launcher.add_sweep(
        param_grid={
            "pruning": ["random"],
            "model": ["resnet18", "vgg16"],
            "selected_classes": [distinct_classes],
            "pruning.pruning_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        },
        name_template="RANDOM_DISTINCT_{pruning}_{pruning.pruning_ratio}" + timestamp
    )


def experiment_sanity_check(launcher: ExperimentLauncher):
    print("%%%%%% Experiment - Sanity Check")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 500,
        "log_results": True,
        "training.batch_size_train": 32,
        "training.batch_size_test": 32,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured", "lrp", "ocap", "unstructured_magnitude", "torchpruner"],
            "model": ["resnet18", "vgg16"],
            "selected_classes": [distinct_classes, similar_classes],
            "pruning.pruning_ratio": [0.5],
        },
        name_template="SANITY_{pruning}_{pruning.pruning_ratio}_{model}_" + timestamp
    )


###################################
# ------------- MAIN ------------ #
###################################

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch multiple pruning experiments automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
        Examples:
          # Preview experiments
          python experiments.py --experiment 1
        
          # Execute experiments
          python experiments.py --experiment 1 --execute
        """
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
        help="Example experiment suite to run."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run experiments (default is dry run)."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue to next experiment if one fails."
    )

    args = parser.parse_args()

    # Create launcher
    launcher = ExperimentLauncher()

    # Experiment library
    experiments = {
        1: experiment_unstructured_distinct,
        2: experiment_unstructured_similar,
        3: experiment_unstructured_amount,

        4: experiment_ln_struct_distinct,
        5: experiment_ln_struct_similar,
        6: experiment_ln_struct_amount,

        7: experiment_lrp_distinct,
        8: experiment_lrp_similar,
        9: experiment_lrp_amount,

        10: experiment_taylor_distinct,
        11: experiment_taylor_similar,
        12: experiment_taylor_amount,

        13: experiment_ocap_distinct,
        14: experiment_ocap_similar,
        15: experiment_ocap_amount,

        16: experiment_random_distinct,
        17: experiment_random_similar,

        18: experiment_layer_sparsity,
        19: experiment_pruning_speed,

        20: experiment_sanity_check,

        21: experiment_kd,
        22: experiment_batchsize,
        23: experiment_ocap_pruning_samples,
    }


    if args.experiment not in experiments:
        print(f"xxxxx Error: Unknown experiment: {args.experiment}")
        sys.exit(1)

    # Setup experiments
    experiments[args.experiment](launcher)

    print("%%%%%% STARTING LAUNCHER")

    # Run
    launcher.run_all(
        dry_run=not args.execute,
        continue_on_error=args.continue_on_error
    )


if __name__ == "__main__":
    main()