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
                    name_params = {}
                    for k, v in config.items():
                        if isinstance(v, list):
                            name_params[k] = f"{len(v)}items"
                        else:
                            name_params[k] = v
                    name = name_template.format(**name_params)
                except KeyError:
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


def experiment_all(launcher: ExperimentLauncher):
    print("\nExample 6: Full Research Suite")
    print("Comprehensive experiment covering all research questions")

    try:
        from class_similarity_analysis import ClassSimilarityAnalyzer
        analyzer = ClassSimilarityAnalyzer()
    except ImportError:
        print("Error: class_similarity_analysis required for this example")
        return

    launcher.base_config = {
        "dataset": "imagenet",
        "model": "resnet18",
        "log_results": True,
        "training.retrain_after_pruning": True,
        "training.retrain_epochs": 5
    }

    # Part 1: Method comparison
    print("  - Method comparison (6 experiments)")
    for method in ["unstructured_magnitude", "unstructured_taylor", "ocap", "ln_structured", "lrp"]:
        launcher.add_experiment(
            config={
                "pruning": method,
                "pruning.pruning_ratio": [0.85],
                "selected_classes": [0, 1, 2]
            },
            name=f"part1_{method}_3classes"
        )

    # Part 2: Class count effect
    print("  - Class count analysis (5 experiments)")
    for count in [3, 10, 25, 50, 100]:
        classes = analyzer.get_diverse_classes(count)
        launcher.add_experiment(
            config={
                "pruning": "ocap",
                "pruning.pruning_ratio": [0.85],
                "selected_classes": classes
            },
            name=f"part2_count_{count}"
        )

    # Part 3: Similarity effect
    print("  - Similarity analysis (6 experiments)")
    for count in [25, 50]:
        for sim_type in ["similar", "diverse", "mixed"]:
            if sim_type == "similar":
                classes = analyzer.get_semantic_group("dogs", size=count)
            elif sim_type == "diverse":
                classes = analyzer.get_diverse_classes(count)
            else:
                classes = analyzer.get_mixed_similarity_classes(count, similar_ratio=0.5)

            launcher.add_experiment(
                config={
                    "pruning": "ocap",
                    "pruning.pruning_ratio": [0.85],
                    "selected_classes": classes
                },
                name=f"part3_{sim_type}_{count}"
            )

    print(f"\nTotal experiments: {len(launcher.experiments)}")


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
        3: experiment_all,
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