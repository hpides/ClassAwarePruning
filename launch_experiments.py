import subprocess
import itertools
import time
import sys
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class ExperimentLauncher:
    """Launch multiple experiments with different configurations."""

    def __init__(self, base_config: Dict[str, Any] = None):
        """
        Initialize launcher with base configuration.

        Args:
            base_config: Configuration applied to all experiments
        """
        self.base_config = base_config or {}
        self.experiments = []

    def add_experiment(self, config: Dict[str, Any], name: str = None):
        """
        Add a single experiment configuration.

        Args:
            config: Dictionary of configuration parameters
            name: Optional name for the experiment
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
        Add a parameter sweep (Cartesian product of all parameters).

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            name_template: Template for run names, e.g., "{pruning}_{model}"
                          Can use any parameter name in curly braces

        Example:
            launcher.add_sweep(
                param_grid={
                    "pruning": ["ocap", "ln_structured"],
                    "model": ["vgg16", "resnet18"]
                },
                name_template="{pruning}_{model}"
            )
            # Creates 4 experiments: ocap_vgg16, ocap_resnet18, ln_structured_vgg16, ln_structured_resnet18
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

    def clear(self):
        """Clear all experiments."""
        self.experiments = []

    def run_all(self, dry_run: bool = True, continue_on_error: bool = True):
        """
        Execute all experiments.

        Args:
            dry_run: If True, only print commands without executing
            continue_on_error: If True, continue to next experiment on failure
        """
        print(f"\n{"=" * 80}")
        print(f"EXPERIMENT LAUNCHER")
        print(f"{"=" * 80}")
        print(f"Total experiments: {len(self.experiments)}")
        print(f"Mode: {"DRY RUN (preview only)" if dry_run else "EXECUTE"}")
        print(f"{"=" * 80}\n")

        results = []
        start_time = time.time()

        for i, exp in enumerate(self.experiments, 1):
            config = exp["config"]
            name = exp["name"]

            print(f"\n{"─" * 80}")
            print(f"[{i}/{len(self.experiments)}] {name}")
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

            print(f"Command: {" ".join(cmd)}")

            if not dry_run:
                exp_start = time.time()
                try:
                    # CHANGE THIS LINE - Remove capture_output, add flush
                    sys.stdout.flush()  # Flush before subprocess
                    sys.stderr.flush()

                    # Don't capture output - let it print directly to terminal
                    result = subprocess.run(cmd, check=True)

                    sys.stdout.flush()  # Flush after subprocess
                    sys.stderr.flush()

                    exp_time = time.time() - exp_start
                    print(f"\n✓ Completed in {exp_time:.1f}s")
                    results.append({"name": name, "status": "success", "time": exp_time})
                except subprocess.CalledProcessError as e:
                    exp_time = time.time() - exp_start
                    print(f"\n✗ Failed after {exp_time:.1f}s: {e}")
                    results.append({"name": name, "status": "failed", "time": exp_time})

                    if not continue_on_error:
                        print("\nStopping due to error.")
                        break
            else:
                print("(dry run - not executed)\n")
                results.append({"name": name, "status": "skipped", "time": 0})

        # Summary
        total_time = time.time() - start_time
        print(f"\n{"=" * 80}")
        print(f"SUMMARY")
        print(f"{"=" * 80}")
        print(f"Total experiments: {len(self.experiments)}")

        if not dry_run:
            success = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "failed")
            print(f"Successful: {success}")
            print(f"Failed: {failed}")
            print(f"Total time: {total_time / 60:.1f} minutes")

            if failed > 0:
                print(f"\nFailed experiments:")
                for r in results:
                    if r["status"] == "failed":
                        print(f"  - {r["name"]}")

        print(f"{"=" * 80}\n")


######################################################
###############       EXPERIMENTS       ##############
######################################################

def experiment_unstructured_vs_structured(launcher: ExperimentLauncher):
    print("\nExperiment - Unstructured vs structured comparison")

    launcher.base_config = {
        "model": "resnet18",
        "dataset": "imagenette",
        "selected_classes": [0, 1, 2],
        "pruning.pruning_ratio": [0.9],
        "log_results": True
    }

    # Unstructured methods
    for method in ["unstructured_magnitude", "unstructured_gradient", "unstructured_taylor", "unstructured_random"]:
        launcher.add_experiment(
            config={"pruning": method},
            name=f"{method}_baseline"
        )

    # Structured methods
    for method in ["ocap", "ln_structured"]:
        launcher.add_experiment(
            config={"pruning": method},
            name=f"{method}_comparison"
        )


def experiment_structured_similar_images(launcher: ExperimentLauncher):
    print("\nExperiment - Structured pruning with similar images")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "model": "resnet18",
        "dataset": "imagenette",
        #"dataset.subsample_ratio": 0.05,
        #"dataset.subsample_size_per_class": 2,
        #"selected_classes": [153, 200, 229, 231, 232, 242, 248], # some dogs
        "selected_classes": [4,6,8],  # some dogs
        "pruning.pruning_ratio": [0.8],
        "log_results": True
    }

    # Unstructured methods
    #for method in ["unstructured_magnitude", "unstructured_gradient", "unstructured_taylor", "unstructured_random"]:
    for method in ["unstructured_magnitude"]:
        launcher.add_experiment(
            config={"pruning": method},
            name=f"{method}_baseline_{timestamp}"
        )

    # Structured methods
    for method in ["ocap", "ln_structured"]:
        launcher.add_experiment(
            config={"pruning": method},
            name=f"{method}_comparison_{timestamp}"
        )


def experiment_playground(launcher: ExperimentLauncher):
    print("\nExperiment - Playground")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "model": "resnet18",
        "dataset": "imagenette",
        #"dataset.subsample_ratio": 0.05,
        #"dataset.subsample_size_per_class": 2,
        #"selected_classes": [153, 200, 229, 231, 232, 242, 248], # some dogs
        "selected_classes": [4,6,8],  # some dogs
        #"pruning.pruning_ratio": [0.8],
        "pruning.pruning_ratio": [0.8],
        "log_results": True
    }

    # Unstructured methods
    #for method in ["unstructured_magnitude", "unstructured_gradient", "unstructured_taylor", "unstructured_random"]:
    for method in ["ocap"]:
        #for model in ["vgg16", "resnet18", "resnet50", "resnet152", "mobilenetv2"]:
        #for model in ["resnet18", "resnet50"]:
        for model in ["resnet50"]:
            launcher.add_experiment(
                config={"pruning": method, "model": model},
                name=f"{model}_{method}_baseline_{timestamp}"
            )


def experiment_pruning_all(launcher: ExperimentLauncher):
    print("\nExperiment - Pruning all")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenette",
        #"dataset.subsample_ratio": 0.05,
        #"dataset.subsample_size_per_class": 2,
        #"selected_classes": [153, 200, 229, 231, 232, 242, 248], # some dogs
        "selected_classes": [4,6,8],  # some dogs
        #"pruning.pruning_ratio": [0.8],
        "log_results": True
    }

    for method in ["ocap", "ln_structured", "lrp", "cap", "torchpruner",
        "unstructured_magnitude", "unstructured_gradient", "unstructured_taylor", "unstructured_random"]:
    #for method in ["ocap"]:
        #for model in ["vgg16", "resnet18", "resnet50", "resnet152", "mobilenetv2"]:
        #for model in ["resnet18", "resnet50"]:
        for model in ["vgg16", "resnet18"]:
            for ratio in [0.3, 0.5, 0.8]:
                launcher.add_experiment(
                    config={"pruning": method, "model": model, "pruning.pruning_ratio": ratio,},
                    name=f"{model}_{method}_{ratio}_{timestamp}"
                )


def experiment_kd(launcher: ExperimentLauncher):
    print("\nExperiment - KD")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 100,
        "selected_classes": list(range(10)),
        #"use_knowledge_distillation": True,
        "pruning": "ocap",
        "log_results": True
    }

    launcher.add_sweep(
        param_grid={
            #"pruning": ["lrp", "torchpruner", "ocap", "ln_structured", "unstructured_magnitude"],
            "use_knowledge_distillation": [True, False],
            "model": ["vgg16", "resnet18"],
            "pruning.pruning_ratio": [0.5, 0.75, 0.9]
        },
        name_template="KD_{use_knowledge_distillation}_{model}_{pruning.pruning_ratio}_KD_" + timestamp
    )


def experiment_similar_images(launcher: ExperimentLauncher):
    print("\nExperiment - Similar")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 100,
        "selected_classes": [
            472,    # canoe
            484,    # catamaran
            554,    # fireboat
            576,    # gondola
            625,    # lifeboat
            628,    # liner, ocean liner
            724,    # pirate, pirate ship
            780,    # schooner
            814,    # speedboat
            871],   # trimaran
        "log_results": True
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["lrp", "torchpruner", "ocap", "ln_structured", "unstructured_magnitude"],
            "model": ["vgg16", "resnet18"],
            "pruning.pruning_ratio": [0.5, 0.75, 0.9],
        },
        name_template="SIM_{pruning}_{model}_{pruning.pruning_ratio}_" + timestamp
    )


def experiment_different_images(launcher: ExperimentLauncher):
    print("\nExperiment - Different")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 100,
        "selected_classes": [
            0,      # tench, Tinca tinca (fish)
            145,    # king penguin (bird)
            398,    # abacus (mathematical tool)
            483,    # castle (building)
            812,    # space shuttle (vehicle)
            924,    # guacamole (food)
            970,    # alp (geographical feature)
            985,    # daisy (flower)
            51,     # triceratops (extinct dinosaur)
            916],   # web site, website, internet site, site (digital concept)
        "log_results": True
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["lrp", "torchpruner", "ocap", "ln_structured", "unstructured_magnitude"],
            "model": ["vgg16", "resnet18"],
            "pruning.pruning_ratio": [0.5, 0.75, 0.9],
        },
        name_template="DIF_{pruning}_{model}_{pruning.pruning_ratio}_" + timestamp
    )

def experiment_different_amount_of_images(launcher: ExperimentLauncher):
    print("\nExperiment - Different amount of images")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 100,
        "log_results": True
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ocap", "ln_structured", "lrp", "torchpruner", "unstructured_magnitude"],
            "model": ["vgg16", "resnet18"],
            "pruning.pruning_ratio": [0.5, 0.75, 0.9],
            "selected_classes": [list(range(n)) for n in [3, 10, 30, 50]]
        },
        name_template="AMT_{pruning}_{model}_{pruning.pruning_ratio}_{selected_classes}_" + timestamp
    )


def experiment_debug(launcher: ExperimentLauncher):
    print("\nExperiment - Debug")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        #"training.train": True,
        #"dataset": "cifar10",
        #"training.use_pretrained_model": False,
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 100,
        #"dataset.subsample_ratio": 0.8,
        "selected_classes": [10,20,30,40],
        #"selected_classes": [0,1,2],
        "log_results": True,
        #"pruning.pruning_ratio": [0.3],
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured", "torchpruner", "lrp", "ocap"],
            "model": ["vgg16", "resnet18"],
            "pruning.pruning_ratio": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                      0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        },
        name_template="DEBUG_{pruning}_" + timestamp
    )


def experiment_all_small(launcher: ExperimentLauncher):
    print("\nExperiment - All small")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    launcher.base_config = {
        "dataset": "imagenet",
        "training.use_pretrained_model": True,  # If Imagenet
        "dataset.subsample_size_per_class": 300,
        "log_results": True,
    }

    launcher.add_sweep(
        param_grid={
            "pruning": ["ln_structured", "torchpruner", "lrp", "ocap", "unstructured_magnitude"],
            "model": ["resnet18", "vgg16"],
            "pruning.pruning_ratio": [0.2, 0.5, 0.9],
            "selected_classes": [
                [
                    0,      # tench, Tinca tinca (fish)
                    145,    # king penguin (bird)
                    398,    # abacus (mathematical tool)
                    483,    # castle (building)
                    812,    # space shuttle (vehicle)
                    924,    # guacamole (food)
                    970,    # alp (geographical feature)
                    985,    # daisy (flower)
                    51,     # triceratops (extinct dinosaur)
                    916],
                [
                    472,  # canoe
                    484,  # catamaran
                    554,  # fireboat
                    576,  # gondola
                    625,  # lifeboat
                    628,  # liner, ocean liner
                    724,  # pirate, pirate ship
                    780,  # schooner
                    814,  # speedboat
                    871],  # trimaran
                    list(range(3)),
                    list(range(20)),
                ],
        },
        name_template="DEBUG_{pruning}_" + timestamp
    )


def experiment_batchsize(launcher: ExperimentLauncher):
    print("\nExperiment - All small")

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
            "pruning": ["ln_structured", "lrp", "ocap", "unstructured_magnitude"],
            "model": ["vgg16"],
            "pruning.pruning_ratio": [0.5],
            "training.batch_size_train": [256, 128],
        },
        name_template="DEBUG_{pruning}_{training.batch_size_train}" + timestamp
    )


######################################################
##################       MAIN       ##################
######################################################

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch multiple pruning experiments automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview experiments
  python launch_experiments.py --experiment 1

  # Execute experiments
  python launch_experiments.py --experiment 1 --execute
        """
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
        help="Example experiment suite to run"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run experiments (default is dry run)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue to next experiment if one fails"
    )

    args = parser.parse_args()

    # Create launcher
    launcher = ExperimentLauncher()

    # Select example
    experiments = {
        1: experiment_unstructured_vs_structured,
        2: experiment_structured_similar_images,
        3: experiment_playground,
        4: experiment_pruning_all,
        5: experiment_kd,
        6: experiment_similar_images,
        7: experiment_different_images,
        8: experiment_different_amount_of_images,
        9: experiment_debug,
        10: experiment_all_small,
        11: experiment_batchsize,
    }

    if args.experiment not in experiments:
        print(f"Error: Unknown example {args.experiment}")
        print("Available examples: 1-6")
        sys.exit(1)

    # Setup experiments
    experiments[args.experiment](launcher)

    print("@@@@@ STARTING LAUNCHER")

    # Run
    launcher.run_all(
        dry_run=not args.execute,
        continue_on_error=args.continue_on_error
    )


if __name__ == "__main__":
    main()