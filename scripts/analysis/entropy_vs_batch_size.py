"""
Study entropy as a function of batch size.

For a randomly chosen large batch (e.g., 128 tokens), take sub-batches of
increasing size (8, 16, 24, ..., 128) that all end with the same last token.
For each sub-batch size, compute leading features and their entropies.

Usage:
    python entropy_vs_batch_size.py --preset pythia-70m --layer 3 \\
        --max-batch-size 128 --min-batch-size 8 --step 8
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

matplotlib.use("Agg")

from data_loader import load_wikitext_train_text
from feature_token_influence import process_batch_with_influence
from model_adapters import get_layer, load_model
from presets import Preset, get_preset, site_for
from sae_adapters import SAEBundle, load_sae

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
MAX_BATCH_SIZE = 128
MIN_BATCH_SIZE = 8
BATCH_SIZE_STEP = 8


# --- Entropy helpers --------------------------------------------------------

def compute_feature_entropy(influence_distribution):
    eps = 1e-12
    P = (influence_distribution + eps) / (np.sum(influence_distribution) + eps)
    return scipy.stats.entropy(P, base=2)


# --- Per-sub-batch hot path -------------------------------------------------

def compute_entropy_for_sub_batch(
    model, sae: SAEBundle, sub_batch_tokens, layer_idx, all_features,
    preset: Preset, threshold: float,
):
    """Per-feature influence + entropies + last-position activations for one sub-batch."""
    feature_influences = process_batch_with_influence(
        model, sae, sub_batch_tokens, layer_idx, all_features, threshold, preset,
    )

    # Last-position activations (no grad) for plot colouring / legend ranking.
    input_ids = sub_batch_tokens.unsqueeze(0)
    layer = get_layer(model, preset, layer_idx)
    activations = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        activations.append(output[0] if isinstance(output, tuple) else output)

    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(input_ids)
        resid = activations[0]
        feats = sae.encode(resid)
        last_pos_feats = feats[0, -1, :].cpu().numpy()
        feature_activations = {
            feat_idx: float(last_pos_feats[feat_idx])
            for feat_idx in feature_influences.keys()
        }
    finally:
        handle.remove()

    feature_entropies = {
        feat_idx: compute_feature_entropy(dist)
        for feat_idx, dist in feature_influences.items()
    }

    return {
        "feature_entropies": feature_entropies,
        "feature_activations": feature_activations,
        "feature_influences": feature_influences,
        "num_active_features": len(feature_entropies),
    }


# --- Plotting ---------------------------------------------------------------

def plot_entropy_vs_batch_size(results_by_batch_size, site, output_dir):
    all_feature_indices = set()
    for result in results_by_batch_size.values():
        all_feature_indices.update(result["feature_entropies"].keys())
    all_feature_indices = sorted(all_feature_indices)
    batch_sizes = sorted(results_by_batch_size.keys())

    n_features = len(all_feature_indices)
    if n_features > 0:
        if n_features <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_features))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            while len(colors) < n_features:
                colors = np.vstack([colors, plt.cm.tab20(np.linspace(0, 1, 20))])
            colors = colors[:n_features]
        feature_color_map = {f: colors[i] for i, f in enumerate(all_feature_indices)}
    else:
        feature_color_map = {}

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for feat_idx in all_feature_indices:
        entropies = []
        sizes = []
        for bs in batch_sizes:
            r = results_by_batch_size[bs]
            if feat_idx in r["feature_entropies"]:
                entropies.append(r["feature_entropies"][feat_idx])
                sizes.append(bs)
        if entropies:
            ax.plot(sizes, entropies, "o-", color=feature_color_map[feat_idx],
                    linewidth=2, markersize=6, alpha=0.7)

    # Maximal-entropy reference: log2(n) for a uniform distribution.
    ref_x = np.array(batch_sizes)
    ax.plot(ref_x, np.log2(ref_x), "k--", linewidth=2,
            label="Maximal Entropy (log₂(n))", alpha=0.8)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title(f"{site} - Feature Entropy vs Batch Size",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if all_feature_indices:
        feature_total_act = {
            f: sum(results_by_batch_size[bs]["feature_activations"].get(f, 0.0)
                   for bs in batch_sizes)
            for f in all_feature_indices
        }
        top10 = {f for f, _ in sorted(feature_total_act.items(),
                                      key=lambda x: x[1], reverse=True)[:10]}
        legend_elems = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=feature_color_map[f], markersize=8,
                       markeredgecolor="black", markeredgewidth=0.5,
                       linewidth=2, label=f"Feature {f}")
            for f in all_feature_indices if f in top10
        ]
        legend_elems.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                       linewidth=2, label="Maximal Entropy (log₂(n))"))
        if legend_elems:
            ax.legend(handles=legend_elems, fontsize=9, loc="best", ncol=2)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "entropy_vs_batch_size.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# --- Main -------------------------------------------------------------------

def main(preset_name="pythia-70m", layer_idx=3, max_batch_size=None,
         min_batch_size=None, step=None, random_seed=None, threshold=None,
         output_dir="."):
    preset = get_preset(preset_name)
    site = site_for(preset, layer_idx)
    threshold = threshold if threshold is not None else preset.threshold
    max_batch_size = max_batch_size if max_batch_size is not None else MAX_BATCH_SIZE
    min_batch_size = min_batch_size if min_batch_size is not None else MIN_BATCH_SIZE
    step = step if step is not None else BATCH_SIZE_STEP

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[INFO] entropy_vs_batch_size: preset={preset.name} site={site}")
    print(f"[INFO] Threshold: {threshold}")
    print(f"{'='*60}")

    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    print(f"[INFO] SAE: arch={sae.arch} n_latent={sae.n_latent}")
    all_features = set(range(sae.n_latent))

    text = load_wikitext_train_text()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens: {total_tokens}")

    max_start = total_tokens - max_batch_size
    if max_start <= 0:
        print(f"[ERROR] Not enough tokens for batch size {max_batch_size}")
        return

    if random_seed is not None:
        random.seed(random_seed); np.random.seed(random_seed)
        print(f"[INFO] Random seed: {random_seed}")

    start_idx = random.randint(0, max_start)
    large_batch = tokens[start_idx: start_idx + max_batch_size].to(DEVICE)
    print(f"[INFO] Large batch: start_idx={start_idx} size={max_batch_size}")

    sub_batch_sizes = list(range(min_batch_size, max_batch_size + 1, step))
    if max_batch_size not in sub_batch_sizes:
        sub_batch_sizes.append(max_batch_size)
    sub_batch_sizes.sort()
    print(f"[INFO] Sub-batch sizes: {sub_batch_sizes}")

    results_by_batch_size = {}
    for bs in sub_batch_sizes:
        sub_batch = large_batch[-bs:].clone()
        print(f"\n[INFO] Sub-batch size {bs} "
              f"(positions {max_batch_size - bs}..{max_batch_size - 1})...")
        try:
            result = compute_entropy_for_sub_batch(
                model, sae, sub_batch, layer_idx, all_features, preset, threshold,
            )
            results_by_batch_size[bs] = result
            n = result["num_active_features"]
            print(f"  active={n}", end="")
            if n:
                print(f"  avg_fH={np.mean(list(result['feature_entropies'].values())):.4f}")
            else:
                print()
        except Exception as e:
            print(f"[WARN] sub-batch {bs}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not results_by_batch_size:
        print("[ERROR] No sub-batch sizes processed."); return

    plots_dir = out_root / f"entropy_vs_batch_size_{site}_{timestamp}"
    plot_path = plot_entropy_vs_batch_size(results_by_batch_size, site, plots_dir)
    print(f"[INFO] Plot: {plot_path}")

    # Serialize
    serializable = {}
    for bs, r in results_by_batch_size.items():
        serializable[bs] = {
            "feature_entropies": {int(k): float(v) for k, v in r["feature_entropies"].items()},
            "feature_activations": {int(k): float(v) for k, v in r["feature_activations"].items()},
            "feature_influences": {
                int(k): (v.cpu().numpy().tolist() if isinstance(v, torch.Tensor)
                         else v.tolist())
                for k, v in r["feature_influences"].items()
            },
            "num_active_features": r["num_active_features"],
        }

    output_file = out_root / f"entropy_vs_batch_size_{site}_{timestamp}.pt"
    torch.save({
        "results_by_batch_size": serializable,
        "summary": {
            "preset": preset.name, "site": site, "layer": layer_idx,
            "timestamp": timestamp,
            "max_batch_size": max_batch_size, "min_batch_size": min_batch_size,
            "step": step, "sub_batch_sizes": sub_batch_sizes,
            "start_idx": start_idx,
        },
        "config": {
            "preset": preset.name, "threshold": threshold,
            "random_seed": random_seed, "total_features": sae.n_latent,
            "sae_source": sae.source, "sae_arch": sae.arch,
        },
        "plots_dir": str(plots_dir),
    }, output_file)
    print(f"[INFO] Saved {output_file}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"Sub-batch sizes processed: {sorted(results_by_batch_size.keys())}")
    seen = set()
    for r in results_by_batch_size.values():
        seen.update(r["feature_entropies"].keys())
    print(f"Unique active features across all sizes: {len(seen)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Study entropy as a function of (sub-)batch size",
    )
    parser.add_argument("--preset", type=str, default="pythia-70m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--min-batch-size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    try:
        main(preset_name=args.preset, layer_idx=args.layer,
             max_batch_size=args.max_batch_size, min_batch_size=args.min_batch_size,
             step=args.step, random_seed=args.random_seed,
             threshold=args.threshold, output_dir=args.output_dir)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
