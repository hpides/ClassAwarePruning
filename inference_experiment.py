"""
Benchmark pre-pruned models in /models for:
  - Peak VRAM during inference  (weights + batch + all activations)
  - Peak RAM during inference   (CPU-side allocations via tracemalloc)
  - Batch inference speed
  - FLOPs for one forward pass

Both memory metrics are sampled at the hottest point of a real forward pass,
not around model loading, so they reflect true runtime footprint.

Results are saved to benchmark_results.csv
"""

import os
import re
import time
import csv
import ast
import tracemalloc
from datetime import datetime
import torch
from fvcore.nn import FlopCountAnalysis

from data_loader import dataloaderFactories

# ---------------------------------------------------------------------------
# Class index pools (same as in main.py)
# ---------------------------------------------------------------------------
similar_classes = [724, 628, 484, 576, 625, 814, 472, 780, 554, 871]
distinct_classes = [985, 483, 812, 924, 107, 398, 970, 51, 145, 916]

ALL_KNOWN_CLASSES = set(similar_classes + distinct_classes)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = "models"
OUTPUT_CSV = f"benchmark_results_{timestamp}.csv"
BATCH_SIZE  = 128
N_WARMUP    = 3   # warm-up batches before timing
N_TIMED     = 10  # batches to average timing over

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_classes_from_filename(filename: str) -> list[int]:
    """
    Extract the first five class indices from a filename like:
      resnet18__torchpruner__0.847__[985, 483, 812, 924, 107].pt
    Returns the list, e.g. [985, 483, 812, 924, 107].
    """
    match = re.search(r'\[([^\]]+)\]', filename)
    if not match:
        raise ValueError(f"Cannot parse class indices from filename: {filename}")
    classes = ast.literal_eval(f"[{match.group(1)}]")
    return list(classes)


def pick_selected_classes(file_classes: list[int]) -> list[int]:
    """
    Return similar_classes if all file_classes are in similar_classes,
    distinct_classes if they are in distinct_classes, otherwise fall back
    to distinct_classes as a safe default.
    """
    if all(c in similar_classes for c in file_classes):
        return similar_classes
    if all(c in distinct_classes for c in file_classes):
        return distinct_classes
    # Mixed or unknown — use file_classes directly as selected_classes
    return file_classes


def peak_vram_during_inference_mb(model, imgs, device) -> float:
    """
    Run one forward pass and return peak VRAM (MB) including weights,
    input batch, and all intermediate activations.
    reset_peak_memory_stats() is called immediately before the forward pass
    so we get a clean reading not contaminated by earlier allocations.
    """
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(imgs)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1e6


def peak_ram_during_inference_mb(model, imgs) -> float:
    """
    Run one forward pass and return peak CPU RAM (MB) for Python-side
    allocations using tracemalloc. This captures numpy / tensor copies
    that live on the CPU during the forward pass.
    """
    tracemalloc.start()
    tracemalloc.clear_traces()
    with torch.no_grad():
        _ = model(imgs)
    _, peak = tracemalloc.get_traced_memory()   # peak in bytes
    tracemalloc.stop()
    return peak / 1e6


# ---------------------------------------------------------------------------
# Main benchmarking loop
# ---------------------------------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model_files = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pt")
    ])
    print(f"Found {len(model_files)} model(s) in {MODELS_DIR}\n")

    fieldnames = [
        "filename",
        "selected_classes_pool",
        "peak_vram_inference_mb",
        "avg_vram_inference_mb",
        "peak_ram_inference_mb",
        "avg_batch_inference_ms",
        "flops_total",
        "flops_mflops",
    ]

    results = []
    skipped = []

    # Open CSV once, write header, then flush a row after each model
    csv_file = open(OUTPUT_CSV, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()

    for fname in model_files:
        if "ln_structured" not in fname:
            continue
        fpath = os.path.join(MODELS_DIR, fname)
        print(f"{'='*70}")
        print(f"Benchmarking: {fname}")

        # ---- Parse classes & build dataloader ----
        try:
            file_classes = parse_classes_from_filename(fname)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        selected_classes = pick_selected_classes(file_classes)
        pool_name = (
            "similar_classes" if selected_classes is similar_classes
            else "distinct_classes" if selected_classes is distinct_classes
            else "custom"
        )
        print(f"  Class pool: {pool_name} → {selected_classes}")

        dataloader_factory = dataloaderFactories["imagenet"](
            train_batch_size=BATCH_SIZE,
            test_batch_size=BATCH_SIZE,
            selected_classes=selected_classes,
            num_pruning_samples=None,
            use_data_augmentation=True,
            use_imagenet_labels=False,
            subsample_ratio=None,
            subsample_size_per_class=500,
        )
        _, _, test_loader = dataloader_factory.get_dataloaders()

        # ---- Load model ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            model = torch.load(fpath, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  [SKIP] Failed to load model: {e}")
            skipped.append((fname, str(e)))
            continue

        try:
            model.eval()

            # ---- FLOPs ----
            dummy = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                flops_analysis = FlopCountAnalysis(model, dummy)
                flops_analysis.unsupported_ops_warnings(False)
                flops_analysis.uncalled_modules_warnings(False)
                total_flops = flops_analysis.total()
            mflops = total_flops / 1e6
            print(f"  FLOPs (1 forward) : {mflops:.2f} MFLOPs")

            # ---- Batch inference speed + peak memory ----
            # We measure memory on the FIRST timed batch (post warm-up) so that
            # weights are already resident and we capture the true inference peak:
            #   VRAM = weights + input batch + all layer activations
            #   RAM  = CPU-side Python allocations during the forward pass
            data_iter = iter(test_loader)

            # warm-up: let CUDA/cuDNN settle, caches fill, etc.
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    try:
                        imgs, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(test_loader)
                        imgs, _ = next(data_iter)
                    imgs = imgs.to(device)
                    _ = model(imgs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Grab one batch for memory measurement
            try:
                mem_imgs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(test_loader)
                mem_imgs, _ = next(data_iter)
            mem_imgs = mem_imgs.to(device)

            # Peak VRAM during a real forward pass (weights + activations + batch)
            peak_vram_mb = peak_vram_during_inference_mb(model, mem_imgs, device)
            # Peak CPU RAM during the same forward pass
            peak_ram_mb  = peak_ram_during_inference_mb(model, mem_imgs)

            print(f"  Peak VRAM (inference): {peak_vram_mb:.1f} MB")
            print(f"  Peak RAM  (inference): {peak_ram_mb:.1f} MB")

            # Timing loop (separate from memory measurement to avoid tracemalloc overhead)
            batch_times = []
            vram_samples = []
            with torch.no_grad():
                for _ in range(N_TIMED):
                    try:
                        imgs, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(test_loader)
                        imgs, _ = next(data_iter)
                    imgs = imgs.to(device)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model(imgs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_times.append((time.perf_counter() - t0) * 1000)  # ms

                    # Sample live VRAM after each forward pass (post-synchronize, so accurate)
                    if torch.cuda.is_available():
                        vram_samples.append(torch.cuda.memory_allocated(device) / 1e6)

            avg_batch_ms = sum(batch_times) / len(batch_times)
            avg_vram_mb  = sum(vram_samples) / len(vram_samples) if vram_samples else 0.0
            print(f"  Avg batch inference: {avg_batch_ms:.2f} ms  (over {N_TIMED} batches, bs={BATCH_SIZE})")
            print(f"  Avg VRAM (inference): {avg_vram_mb:.1f} MB")

            row = {
                "filename": fname,
                "selected_classes_pool": pool_name,
                "peak_vram_inference_mb": round(peak_vram_mb, 2),
                "avg_vram_inference_mb": round(avg_vram_mb, 2),
                "peak_ram_inference_mb": round(peak_ram_mb, 2),
                "avg_batch_inference_ms": round(avg_batch_ms, 3),
                "flops_total": total_flops,
                "flops_mflops": round(mflops, 2),
            }
            results.append(row)
            csv_writer.writerow(row)
            csv_file.flush()
            print(f"  [OK] Row written to {OUTPUT_CSV}")

        except Exception as e:
            print(f"  [SKIP] Error during benchmarking: {e}")
            skipped.append((fname, str(e)))

        finally:
            # clean up before next model regardless of success/failure
            try:
                del model
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_file.close()

    print(f"\n{'='*70}")
    print(f"Done! Results saved to: {OUTPUT_CSV}")
    print(f"Benchmarked : {len(results)} model(s)")
    print(f"Skipped     : {len(skipped)} model(s)")
    if skipped:
        print("Skipped files:")
        for fname, reason in skipped:
            print(f"  - {fname}: {reason}")


if __name__ == "__main__":
    main()