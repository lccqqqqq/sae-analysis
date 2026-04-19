"""
Compare entropy between per-feature influences and the token-vector influence.

For each batch:
  1. Per-feature influence J_a(t')  ->  H_feat_a = -sum P log P
  2. Token-vector influence R(t')   ->  H_token
Conjecture: H_token > <H_feat_a>. I.e. individual SAE features form a
"low-entropy decomposition" of the residual-stream direction they reconstruct.

Usage:
    python compare_entropies.py --preset pythia-70m --layer 3 --num-batches 10
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

from feature_token_influence import process_batch_with_influence
from model_adapters import get_layer, load_model
from presets import Preset, get_preset, site_for
from sae_adapters import SAEBundle, load_sae
from token_vector_influence import process_batch_with_token_influence

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
BATCH_SIZE = 64
NUM_BATCHES = 10


# --- Entropy helpers --------------------------------------------------------

def compute_feature_entropy(influence_distribution):
    eps = 1e-12
    P = (influence_distribution + eps) / (np.sum(influence_distribution) + eps)
    return scipy.stats.entropy(P, base=2)


def normalize_influence(influence_distribution):
    eps = 1e-12
    return (influence_distribution + eps) / (np.sum(influence_distribution) + eps)


# --- Per-batch entropy comparison (the hot path) ---------------------------

def compare_entropies_for_batch(
    model, sae: SAEBundle, tokens, layer_idx, all_features, site,
    preset: Preset, threshold: float,
):
    """Run feature + token-vector influence for one batch, return both entropies.

    Both inner functions swap the embedding layer + hooks and restore them in
    their own finally blocks, so sequential calls are safe.
    """
    # Per-feature influence (gradient per active feature).
    feature_influences = process_batch_with_influence(
        model, sae, tokens, layer_idx, all_features, threshold, preset,
    )

    # Pull last-position activations (no grad) for coloring / diagnostics.
    input_ids = tokens.unsqueeze(0)
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

    R_values, token_vector_entropy = process_batch_with_token_influence(
        model, tokens, layer_idx, preset,
    )

    return {
        "feature_entropies": feature_entropies,
        "token_vector_entropy": token_vector_entropy,
        "num_active_features": len(feature_entropies),
        "feature_influences": feature_influences,
        "feature_activations": feature_activations,
        "token_vector_influence": R_values,
    }


# --- Plotting ---------------------------------------------------------------

def plot_batch_comparison(batch_result, batch_idx, site, output_dir):
    """Two-panel figure: left = per-token probability traces, right = entropy scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    feature_influences = batch_result["feature_influences"]
    token_vector_influence = batch_result["token_vector_influence"]
    feature_entropies = batch_result["feature_entropies"]
    token_vector_entropy = batch_result["token_vector_entropy"]
    feature_activations = batch_result.get("feature_activations", {})

    seq_len = len(token_vector_influence)
    positions = np.arange(seq_len)
    ax1.plot(positions, normalize_influence(token_vector_influence),
             "k--", linewidth=2, label="Token Vector", alpha=0.7)

    sorted_feats = sorted(feature_influences.keys())
    if feature_activations and sorted_feats:
        ranked = sorted(
            [(i, feature_activations.get(i, 0.0)) for i in sorted_feats],
            key=lambda x: x[1], reverse=True,
        )
        top = [i for i, _ in ranked[:5]]
    else:
        top = sorted_feats[:5]

    n_top = len(top)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))[:n_top] if n_top else []
    top_color = {idx: palette[i] for i, idx in enumerate(top)}

    # Cap how many grey lines we draw — matplotlib chokes on >>1k traces.
    GREY_CAP = 200
    grey_pool = [i for i in sorted_feats if i not in top_color]
    if len(grey_pool) > GREY_CAP:
        step = len(grey_pool) // GREY_CAP + 1
        grey_pool = grey_pool[::step][:GREY_CAP]
    grey_set = set(grey_pool)

    for feat_idx in sorted_feats:
        if feat_idx not in top_color and feat_idx not in grey_set:
            continue
        feat_prob = normalize_influence(feature_influences[feat_idx])
        if feat_idx in top_color:
            ax1.plot(positions, feat_prob, "-", linewidth=1.5,
                     label=f"Feature {feat_idx}", color=top_color[feat_idx], alpha=0.7)
        else:
            ax1.plot(positions, feat_prob, "-", linewidth=1.0,
                     color="lightgrey", alpha=0.5)

    ax1.set_xlabel("Token Position"); ax1.set_ylabel("Probability")
    ax1.set_title("Influence probability distributions")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.set_xlim(0, seq_len - 1)

    if feature_entropies:
        # Vectorize the grey scatter so we don't call ax2.scatter() 25k times.
        grey_x, grey_y = [], []
        for idx in sorted_feats:
            x = feature_activations.get(idx, idx) if feature_activations else idx
            y = feature_entropies[idx]
            if idx in top_color:
                ax2.scatter([x], [y], s=50, color=top_color[idx],
                            edgecolors="black", linewidth=0.5, zorder=3, alpha=0.7)
            else:
                grey_x.append(x); grey_y.append(y)
        if grey_x:
            ax2.scatter(grey_x, grey_y, s=20, color="lightgrey",
                        edgecolors="none", zorder=2, alpha=0.4)
        ax2.axhline(token_vector_entropy, color="red", linestyle="--",
                    linewidth=2, alpha=0.8)
        legend = [plt.Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=top_color[i], markersize=8,
                             markeredgecolor="black", markeredgewidth=0.5,
                             label=f"Feature {i}") for i in top]
        legend.append(plt.Line2D([0], [0], color="red", linestyle="--", linewidth=2,
                                 label=f"Token Vector ({token_vector_entropy:.3f})"))
        ax2.legend(handles=legend, fontsize=9, loc="best")
    else:
        ax2.axhline(token_vector_entropy, color="red", linestyle="--",
                    linewidth=2, label=f"Token Vector ({token_vector_entropy:.3f})")
        ax2.legend(fontsize=9)

    ax2.set_xlabel("Activation" if feature_activations else "Feature Index")
    ax2.set_ylabel("Entropy (bits)")
    ax2.set_title("Entropy vs activation")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{site} - Batch {batch_idx + 1}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"batch_{batch_idx:03d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# --- Main (single layer) ----------------------------------------------------

def main(preset_name="pythia-70m", layer_idx=3, num_batches=None,
         random_batches=True, random_seed=None, threshold=None,
         output_dir="."):
    preset = get_preset(preset_name)
    site = site_for(preset, layer_idx)
    threshold = threshold if threshold is not None else preset.threshold
    num_batches = num_batches if num_batches is not None else NUM_BATCHES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[INFO] Compare entropies: preset={preset.name} site={site}")
    print(f"[INFO] Threshold: {threshold}")
    print(f"{'='*60}")

    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    print(f"[INFO] SAE: arch={sae.arch} n_latent={sae.n_latent}")

    all_features = set(range(sae.n_latent))

    DATA_FILE = Path("wikitext-2-train.txt")
    if not DATA_FILE.exists():
        print("[ERROR] wikitext-2-train.txt not found."); sys.exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens: {total_tokens}")

    if random_batches and random_seed is not None:
        random.seed(random_seed); np.random.seed(random_seed)

    max_start = total_tokens - BATCH_SIZE
    if random_batches:
        starts = list(range(0, max_start + 1, BATCH_SIZE))
        starts = random.sample(starts, min(num_batches, len(starts)))
        starts.sort()
    else:
        starts = [(b * (max_start // num_batches)) % max_start for b in range(num_batches)]

    batch_results = []
    for batch_idx, start_idx in enumerate(starts):
        chunk = tokens[start_idx: start_idx + BATCH_SIZE].to(DEVICE)
        try:
            r = compare_entropies_for_batch(
                model, sae, chunk, layer_idx, all_features, site, preset, threshold,
            )
            batch_results.append({"batch_idx": batch_idx, "start_idx": start_idx, **r})
            print(f"[batch {batch_idx+1}/{num_batches}] active={r['num_active_features']} "
                  f"tH={r['token_vector_entropy']:.3f}")
        except Exception as e:
            print(f"[WARN] batch {batch_idx}: {e}")
            import traceback; traceback.print_exc()

    if not batch_results:
        print("[ERROR] No batches processed."); return

    plots_dir = out_root / f"entropy_plots_{site}_{timestamp}"
    for br in batch_results:
        plot_batch_comparison(br, br["batch_idx"], site, plots_dir)

    all_feat_H = [h for br in batch_results for h in br["feature_entropies"].values()]
    token_H = [br["token_vector_entropy"] for br in batch_results]
    mean_feat = float(np.mean(all_feat_H)) if all_feat_H else None
    mean_tok = float(np.mean(token_H))

    output_file = out_root / f"entropy_comparison_{site}_{timestamp}.pt"
    torch.save({
        "batch_results": batch_results,
        "summary": {
            "site": site, "preset": preset.name, "layer": layer_idx,
            "timestamp": timestamp, "num_batches": len(batch_results),
            "mean_feature_entropy": mean_feat,
            "mean_token_vector_entropy": mean_tok,
            "entropy_difference": (mean_tok - mean_feat) if mean_feat is not None else None,
        },
        "config": {
            "preset": preset.name, "threshold": threshold,
            "batch_size": BATCH_SIZE, "total_features": sae.n_latent,
            "random_batches": random_batches, "random_seed": random_seed,
            "sae_source": sae.source, "sae_arch": sae.arch,
        },
        "plots_dir": str(plots_dir),
        "batch_start_indices": [br["start_idx"] for br in batch_results],
    }, output_file)
    print(f"[INFO] Saved results to {output_file}")
    if mean_feat is not None:
        print(f"[INFO] mean_feat_H={mean_feat:.4f}  mean_token_H={mean_tok:.4f}  "
              f"diff={mean_tok - mean_feat:+.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="pythia-70m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--sequential-batches", action="store_false", dest="random_batches")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    try:
        main(preset_name=args.preset, layer_idx=args.layer,
             num_batches=args.num_batches, random_batches=args.random_batches,
             random_seed=args.random_seed, threshold=args.threshold,
             output_dir=args.output_dir)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
