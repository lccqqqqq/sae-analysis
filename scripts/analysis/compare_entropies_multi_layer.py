"""
Compare feature-influence vs token-vector entropies across all layers, in a
single pass (share model + data loading). This is the script cluster jobs
invoke.

Usage:
    python compare_entropies_multi_layer.py --preset pythia-70m --num-batches 50
    python compare_entropies_multi_layer.py --preset gpt2-small --layers 0 3 11 \
        --num-batches 50
"""

import argparse
import json
import platform
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from compare_entropies import (
    compare_entropies_for_batch,
    plot_batch_comparison,
)
from model_adapters import load_model
from presets import Preset, get_preset, site_for
from sae_adapters import SAEBundle, load_sae

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
BATCH_SIZE = 64


# --- Progress / heartbeat helpers ------------------------------------------

def _gpu_mem_str():
    if not torch.cuda.is_available():
        return ""
    cur = torch.cuda.memory_allocated() / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return f" gpu_mem={cur:.1f}G peak={peak:.1f}G"


def _heartbeat_loop(stop_event, t0, state, interval):
    while not stop_event.wait(interval):
        elapsed = time.time() - t0
        print(
            f"[heartbeat] t={elapsed:7.1f}s "
            f"batch={state.get('batch', 0)}/{state.get('total', 0)} "
            f"phase={state.get('phase', '-')}" + _gpu_mem_str(),
            flush=True,
        )


def _fmt_eta(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# --- Load all SAEs for the requested layer set ------------------------------

def load_all_saes(preset: Preset, layers: list[int]) -> dict[int, SAEBundle]:
    saes: dict[int, SAEBundle] = {}
    for layer_idx in layers:
        saes[layer_idx] = load_sae(preset, layer_idx, DEVICE)
        sae = saes[layer_idx]
        print(f"[INFO] L{layer_idx}: {sae.arch} n_latent={sae.n_latent}", flush=True)
    return saes


def process_batch_for_all_layers(
    model, saes: dict[int, SAEBundle], tokens, layers: list[int],
    preset: Preset, threshold: float,
):
    results_dict = {}
    all_features_by_layer = {L: set(range(saes[L].n_latent)) for L in layers}
    for layer_idx in layers:
        site = site_for(preset, layer_idx)
        try:
            results = compare_entropies_for_batch(
                model, saes[layer_idx], tokens, layer_idx,
                all_features_by_layer[layer_idx], site, preset, threshold,
            )
            results_dict[layer_idx] = results
        except Exception as e:
            print(f"[WARN] Error processing layer {layer_idx}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results_dict[layer_idx] = None
    return results_dict


# --- Main -------------------------------------------------------------------

def main(
    preset_name, layers=None, num_batches=10, random_batches=True,
    random_seed=None, threshold=None, log_every=1, heartbeat_interval=30.0,
    output_dir="data",
):
    preset = get_preset(preset_name)
    if layers is None:
        layers = preset.default_layers or list(range(preset.num_layers))
    threshold = threshold if threshold is not None else preset.threshold

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output layout:
    #   <output_dir>/<preset>/<timestamp>/
    #     entropy_comparison_<site>.pt       (one per layer)
    #     entropy_plots_<site>/batch_NNN.png (one dir per layer)
    #     bench.json                         (timing + host info)
    #     run_config.json                    (preset + CLI args + env)
    #   <output_dir>/<preset>/latest         (symlink -> <timestamp>)
    preset_dir = Path(output_dir) / preset.name
    run_dir = preset_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu/mps"
    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] Preset: {preset.name}  (model={preset.model_id})", flush=True)
    print(f"[INFO] Layers: {layers}", flush=True)
    print(f"[INFO] Threshold: {threshold}", flush=True)
    print(f"[INFO] Device: {DEVICE} ({gpu_name})  torch={torch.__version__}  "
          f"host={platform.node()}", flush=True)
    print(f"[INFO] Run directory: {run_dir.resolve()}", flush=True)
    print(f"{'='*60}", flush=True)

    # Write run_config.json up front so it exists even if the job aborts.
    run_config = {
        "preset": preset.name,
        "model_id": preset.model_id,
        "sae_loader": preset.sae_loader,
        "sae_arch": preset.sae_arch,
        "layers": list(layers),
        "threshold": threshold,
        "num_batches": num_batches,
        "batch_size": BATCH_SIZE,
        "random_batches": random_batches,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "host": platform.node(),
        "device": DEVICE,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model(preset, DEVICE)
    print(f"[INFO] Loading SAE weights for {len(layers)} layers...", flush=True)
    saes = load_all_saes(preset, layers)

    DATA_FILE = Path("wikitext-2-train.txt")
    if not DATA_FILE.exists():
        print("[ERROR] wikitext-2-train.txt not found."); sys.exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens: {total_tokens}", flush=True)

    print(f"[INFO] Processing {num_batches} batches of size {BATCH_SIZE}...", flush=True)
    if random_batches and random_seed is not None:
        random.seed(random_seed); np.random.seed(random_seed)
        print(f"[INFO] Random seed: {random_seed}", flush=True)

    max_start = total_tokens - BATCH_SIZE
    if max_start <= 0:
        print("[ERROR] Not enough tokens for batch size"); return
    if random_batches:
        starts = list(range(0, max_start + 1, BATCH_SIZE))
        if len(starts) < num_batches:
            starts = list(range(max_start + 1))
        batch_start_indices = random.sample(starts, min(num_batches, len(starts)))
        batch_start_indices.sort()
    else:
        batch_start_indices = [
            (b * (max_start // num_batches)) % max_start for b in range(num_batches)
        ]

    batch_results_by_layer = {L: [] for L in layers}
    per_batch_times = []

    hb_state = {"batch": 0, "total": num_batches, "phase": "init"}
    hb_stop = threading.Event()
    hb_thread = None
    if heartbeat_interval and heartbeat_interval > 0:
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(hb_stop, time.time(), hb_state, heartbeat_interval),
            daemon=True,
        )
        hb_thread.start()

    total_start_time = time.time()
    for batch_idx in range(num_batches):
        start_idx = batch_start_indices[batch_idx]
        chunk = tokens[start_idx: start_idx + BATCH_SIZE].to(DEVICE)

        hb_state["batch"] = batch_idx + 1
        hb_state["phase"] = "process_all_layers"
        t0 = time.time()

        results_dict = process_batch_for_all_layers(
            model, saes, chunk, layers, preset, threshold,
        )

        for L in layers:
            if results_dict[L] is not None:
                batch_results_by_layer[L].append({
                    "batch_idx": batch_idx, "start_idx": start_idx,
                    **results_dict[L],
                })

        dt = time.time() - t0
        per_batch_times.append(dt)
        hb_state["phase"] = "idle"

        if (batch_idx % max(1, log_every) == 0) or (batch_idx == num_batches - 1):
            avg = float(np.mean(per_batch_times))
            eta = (num_batches - (batch_idx + 1)) * avg
            bits = []
            for L in layers:
                r = results_dict.get(L)
                if r is None:
                    bits.append(f"L{L}:ERR")
                else:
                    fH = (float(np.mean(list(r["feature_entropies"].values())))
                          if r["feature_entropies"] else float("nan"))
                    bits.append(
                        f"L{L}:n={r['num_active_features']},"
                        f"fH={fH:.2f},tH={float(r['token_vector_entropy']):.2f}"
                    )
            print(
                f"[batch {batch_idx+1:>4}/{num_batches} start={start_idx:>7}] "
                f"dt={dt:5.2f}s avg={avg:5.2f}s eta={_fmt_eta(eta)}"
                f"{_gpu_mem_str()} | " + " ".join(bits),
                flush=True,
            )

    total_elapsed = time.time() - total_start_time
    hb_stop.set()
    if hb_thread is not None:
        hb_thread.join(timeout=2.0)

    print(flush=True)
    print(f"[INFO] Total: {total_elapsed:.2f}s  "
          f"avg {total_elapsed/num_batches:.2f}s/batch  "
          f"{total_elapsed/(num_batches*len(layers)):.2f}s/batch/layer", flush=True)

    all_output_files = {}
    for L in layers:
        site = site_for(preset, L)
        batch_results = batch_results_by_layer[L]
        if not batch_results:
            print(f"[WARN] L{L}: no batches processed"); continue

        plots_dir = run_dir / f"entropy_plots_{site}"
        for br in batch_results:
            plot_batch_comparison(br, br["batch_idx"], site, plots_dir)

        batch_index = {
            "site": site, "preset": preset.name, "timestamp": timestamp,
            "num_batches": len(batch_results), "batch_size": BATCH_SIZE,
            "random_batches": random_batches, "random_seed": random_seed,
            "batches": [
                {
                    "batch_idx": br["batch_idx"],
                    "start_idx": br["start_idx"],
                    "plot_file": f"batch_{br['batch_idx']:03d}.png",
                    "num_active_features": br["num_active_features"],
                    "token_vector_entropy": float(br["token_vector_entropy"]),
                    "avg_feature_entropy": (
                        float(np.mean(list(br["feature_entropies"].values())))
                        if br["num_active_features"] > 0 else None
                    ),
                }
                for br in batch_results
            ],
        }
        with open(plots_dir / "batch_index.json", "w") as f:
            json.dump(batch_index, f, indent=2)

        output_file = run_dir / f"entropy_comparison_{site}.pt"
        summary_feat = [
            np.mean(list(br["feature_entropies"].values()))
            for br in batch_results if br["num_active_features"] > 0
        ]
        torch.save({
            "batch_results": batch_results,
            "summary": {
                "site": site, "preset": preset.name, "timestamp": timestamp,
                "layer": L, "num_batches": len(batch_results),
                "mean_feature_entropy": float(np.mean(summary_feat)) if summary_feat else None,
                "mean_token_vector_entropy": float(np.mean(
                    [br["token_vector_entropy"] for br in batch_results]
                )),
            },
            "config": {
                "preset": preset.name, "threshold": threshold,
                "batch_size": BATCH_SIZE,
                "total_features": saes[L].n_latent,
                "random_batches": random_batches, "random_seed": random_seed,
                "sae_source": saes[L].source, "sae_arch": saes[L].arch,
            },
            "plots_dir": str(plots_dir),
            "batch_start_indices": [br["start_idx"] for br in batch_results],
        }, output_file)
        all_output_files[L] = output_file
        print(f"[INFO] L{L}: saved {output_file}", flush=True)

    bench_file = run_dir / "bench.json"
    with open(bench_file, "w") as f:
        json.dump({
            "timestamp": timestamp, "host": platform.node(), "device": DEVICE,
            "gpu_name": gpu_name, "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "preset": preset.name, "layers": list(layers),
            "num_batches": num_batches, "batch_size": BATCH_SIZE,
            "threshold": threshold, "random_batches": random_batches,
            "random_seed": random_seed,
            "total_elapsed_s": total_elapsed,
            "mean_batch_s": float(np.mean(per_batch_times)) if per_batch_times else None,
            "std_batch_s": float(np.std(per_batch_times)) if per_batch_times else None,
            "peak_gpu_mem_gib": (torch.cuda.max_memory_allocated() / 1024 ** 3
                                 if torch.cuda.is_available() else None),
            "output_files": {int(k): str(v.relative_to(run_dir))
                             for k, v in all_output_files.items()},
        }, f, indent=2)
    print(f"[INFO] Bench: {bench_file}", flush=True)

    # Update the 'latest' symlink in the preset directory to point at this run.
    latest_link = preset_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(timestamp, target_is_directory=True)
        print(f"[INFO] Latest -> {latest_link} -> {timestamp}", flush=True)
    except OSError as e:
        print(f"[WARN] Could not update latest symlink: {e}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Preset: {preset.name}  batches={num_batches}  layers={len(layers)}",
          flush=True)
    print(f"Total time: {total_elapsed:.2f}s", flush=True)
    print(f"Run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare entropies for multiple layers (preset-aware)"
    )
    parser.add_argument("--preset", type=str, required=True,
                        help="One of presets.list_presets() (e.g. pythia-70m, gpt2-small)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to process; default = all preset layers")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--sequential-batches", action="store_false",
                        dest="random_batches")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override preset default activation threshold")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    try:
        main(
            preset_name=args.preset, layers=args.layers,
            num_batches=args.num_batches, random_batches=args.random_batches,
            random_seed=args.random_seed, threshold=args.threshold,
            log_every=args.log_every, heartbeat_interval=args.heartbeat_interval,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
