"""
Study entropy as a function of batch size.

This module analyzes how feature entropy changes as batch size increases.
For a randomly chosen large batch (e.g., 128 tokens), we take sub-batches
of increasing size (8, 16, 24, ..., 128) that all end with the same last token.
For each sub-batch size, we compute leading features and their entropies.

Usage:
    python entropy_vs_batch_size.py --site resid_out_layer3 --max-batch-size 128 --min-batch-size 8 --step 8
"""

from token_vector_influence import (
    process_batch_with_token_influence
)
from feature_token_influence import (
    load_sae, get_sae_weights, get_feature_activations,
    process_batch_with_influence
)
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import numpy as np
import scipy.stats
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import random
import json
from datetime import datetime
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

# Import functions from existing modules


THRESHOLD = 0.2

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_BATCH_SIZE = 128  # Size of the large batch to start with
MIN_BATCH_SIZE = 8    # Minimum sub-batch size
BATCH_SIZE_STEP = 8   # Step size for sub-batch sizes


def compute_feature_entropy(influence_distribution):
    """
    Compute entropy of a feature's influence distribution.

    Args:
        influence_distribution: [seq_len] array of J(t,z|t') values

    Returns:
        entropy: Shannon entropy in bits (base 2)
    """
    eps = 1e-12
    J_sum = np.sum(influence_distribution) + eps
    P = (influence_distribution + eps) / J_sum

    # Compute entropy: H = -∑ P log P
    entropy = scipy.stats.entropy(P, base=2)
    return entropy


def normalize_influence(influence_distribution):
    """
    Normalize influence distribution to get probability distribution.

    Args:
        influence_distribution: [seq_len] array of influence values

    Returns:
        probability: [seq_len] array of normalized probabilities
    """
    eps = 1e-12
    total = np.sum(influence_distribution) + eps
    probability = (influence_distribution + eps) / total
    return probability


def compute_entropy_for_sub_batch(
    model, tokenizer, sae_weights, sub_batch_tokens, layer_idx,
    all_features, site
):
    """
    Compute feature entropies for a sub-batch.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        sae_weights: SAE weights dictionary
        sub_batch_tokens: Token IDs [sub_batch_size]
        layer_idx: Layer index to analyze
        all_features: Set of all feature indices to check
        site: Site string for identification

    Returns:
        results: Dict with:
            - feature_entropies: Dict mapping feature_idx -> entropy
            - feature_activations: Dict mapping feature_idx -> activation (at last token)
            - feature_influences: Dict mapping feature_idx -> influence distribution
            - num_active_features: Number of features above threshold
    """
    # Compute feature influences
    feature_influences = process_batch_with_influence(
        model, tokenizer, sae_weights, sub_batch_tokens, layer_idx,
        all_features, THRESHOLD
    )

    # Get activations for the last token position
    input_ids = sub_batch_tokens.unsqueeze(0)  # [1, seq_len]
    layer = model.gpt_neox.layers[layer_idx]
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0])
        else:
            activations.append(output)

    handle = layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = model(input_ids)
        resid = activations[0]  # [1, seq_len, d_model]
        feats = get_feature_activations(
            resid, sae_weights)  # [1, seq_len, n_latent]
        last_pos_feats = feats[0, -1, :].cpu().numpy()  # [n_latent]

        # Store activations only for active features
        feature_activations = {feat_idx: float(last_pos_feats[feat_idx])
                               for feat_idx in feature_influences.keys()}
    finally:
        handle.remove()

    # Compute entropy for each active feature
    feature_entropies = {}
    for feat_idx, influence_dist in feature_influences.items():
        entropy = compute_feature_entropy(influence_dist)
        feature_entropies[feat_idx] = entropy

    return {
        "feature_entropies": feature_entropies,
        "feature_activations": feature_activations,
        "feature_influences": feature_influences,
        "num_active_features": len(feature_entropies)
    }


def plot_entropy_vs_batch_size(results_by_batch_size, site, output_dir):
    """
    Plot leading feature entropy vs batch size.

    Args:
        results_by_batch_size: Dict mapping batch_size -> result dict
        site: Site string for title
        output_dir: Directory to save the figure
    """
    # Collect all features that appear at any batch size
    all_feature_indices = set()
    for result in results_by_batch_size.values():
        all_feature_indices.update(result['feature_entropies'].keys())

    all_feature_indices = sorted(all_feature_indices)
    batch_sizes = sorted(results_by_batch_size.keys())

    # Create color map for features (consistent across batch sizes)
    n_features = len(all_feature_indices)
    if n_features > 0:
        # Use tab20 colormap for up to 20 features, then cycle
        if n_features <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_features))
        else:
            # For more features, use a larger colormap or cycle
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            # Extend by cycling
            while len(colors) < n_features:
                colors = np.vstack(
                    [colors, plt.cm.tab20(np.linspace(0, 1, 20))])
            colors = colors[:n_features]

        feature_color_map = {feat_idx: colors[i]
                             for i, feat_idx in enumerate(all_feature_indices)}
    else:
        feature_color_map = {}

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot entropy vs batch size for each feature
    for feat_idx in all_feature_indices:
        entropies = []
        batch_sizes_with_feature = []

        for batch_size in batch_sizes:
            result = results_by_batch_size[batch_size]
            if feat_idx in result['feature_entropies']:
                entropies.append(result['feature_entropies'][feat_idx])
                batch_sizes_with_feature.append(batch_size)

        if len(entropies) > 0:
            color = feature_color_map[feat_idx]
            ax.plot(batch_sizes_with_feature, entropies, 'o-',
                    color=color, label=f'Feature {feat_idx}',
                    linewidth=2, markersize=6, alpha=0.7)

    # Add maximal entropy line (log2(batch_size) for uniform distribution)
    max_entropy_batch_sizes = np.array(batch_sizes)
    max_entropies = np.log2(max_entropy_batch_sizes)
    ax.plot(max_entropy_batch_sizes, max_entropies, 'k--',
            linewidth=2, label='Maximal Entropy (log₂(n))', alpha=0.8)

    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title(f'{site} - Feature Entropy vs Batch Size',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend (limit to top features by total activation across all batch sizes)
    if len(all_feature_indices) > 0:
        # Compute total activation for each feature across all batch sizes
        feature_total_activations = {}
        for feat_idx in all_feature_indices:
            total_act = 0.0
            for batch_size in batch_sizes:
                result = results_by_batch_size[batch_size]
                if feat_idx in result['feature_activations']:
                    total_act += result['feature_activations'][feat_idx]
            feature_total_activations[feat_idx] = total_act

        # Get top 10 features by total activation
        top_features = sorted(feature_total_activations.items(),
                              key=lambda x: x[1], reverse=True)[:10]
        top_feature_indices = {idx for idx, _ in top_features}

        # Create legend with only top features
        legend_elements = []
        for feat_idx in all_feature_indices:
            if feat_idx in top_feature_indices:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=feature_color_map[feat_idx],
                               markersize=8, markeredgecolor='black',
                               markeredgewidth=0.5, linewidth=2,
                               label=f'Feature {feat_idx}')
                )

        # Add maximal entropy to legend
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                       label='Maximal Entropy (log₂(n))')
        )

        if len(legend_elements) > 0:
            ax.legend(handles=legend_elements, fontsize=9, loc='best', ncol=2)

    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'entropy_vs_batch_size.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main(site=None, max_batch_size=None, min_batch_size=None, step=None, random_seed=None):
    """
    Main function to study entropy as a function of batch size.

    Args:
        site: Site string like "resid_out_layer3" (if None, uses default)
        max_batch_size: Maximum batch size (if None, uses MAX_BATCH_SIZE)
        min_batch_size: Minimum batch size (if None, uses MIN_BATCH_SIZE)
        step: Step size for batch sizes (if None, uses BATCH_SIZE_STEP)
        random_seed: Random seed for reproducibility
    """
    if site is None:
        site = "resid_out_layer3"

    if max_batch_size is None:
        max_batch_size = MAX_BATCH_SIZE

    if min_batch_size is None:
        min_batch_size = MIN_BATCH_SIZE

    if step is None:
        step = BATCH_SIZE_STEP

    # Generate unique ID based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = timestamp

    base_dir = Path("dictionaries/pythia-70m-deduped") / site

    print(f"\n{'='*60}")
    print(f"[INFO] Studying entropy vs batch size for {site}")
    print(f"{'='*60}")

    # 1. Setup
    run_dir = None
    for p in base_dir.iterdir():
        if p.is_dir() and (p / "ae.pt").exists():
            run_dir = p
            break
    if not run_dir:
        raise FileNotFoundError(f"No run directory found in {base_dir}")

    print("[INFO] Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    d_model = model.config.hidden_size

    sae_sd = load_sae(run_dir)
    sae_weights = get_sae_weights(sae_sd, d_model)

    # Get total number of features
    n_latent = sae_weights["dec_w"].shape[1]
    all_features = set(range(n_latent))
    print(
        f"[INFO] Will process all {n_latent} features (filtered by activation threshold per batch)")

    # 2. Load data
    DATA_FILE = Path("wikitext-2-train.txt")
    if not DATA_FILE.exists():
        print("[ERROR] wikitext-2-train.txt not found.")
        sys.exit(1)

    print(f"[INFO] Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[INFO] Tokenizing text...")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_tokens = tokens.shape[0]
    print(f"[INFO] Total tokens available: {total_tokens}")

    # 3. Select a random large batch
    layer_idx = int(site.rsplit("layer", 1)[-1])

    max_start = total_tokens - max_batch_size
    if max_start <= 0:
        print(f"[ERROR] Not enough tokens for batch size {max_batch_size}")
        return

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"[INFO] Random seed: {random_seed}")

    # Randomly select a starting position for the large batch
    start_idx = random.randint(0, max_start)
    large_batch = tokens[start_idx: start_idx + max_batch_size].to(DEVICE)

    print(
        f"\n[INFO] Selected large batch: start_idx={start_idx}, size={max_batch_size}")
    print(f"[INFO] Layer: {layer_idx}, Threshold: {THRESHOLD}")

    # 4. Generate sub-batch sizes
    sub_batch_sizes = list(range(min_batch_size, max_batch_size + 1, step))
    if max_batch_size not in sub_batch_sizes:
        sub_batch_sizes.append(max_batch_size)
    sub_batch_sizes.sort()

    print(f"\n[INFO] Processing sub-batch sizes: {sub_batch_sizes}")

    # 5. Process each sub-batch size
    results_by_batch_size = {}

    for batch_size in sub_batch_sizes:
        # Take sub-batch ending with the same last token: batch[-n:]
        sub_batch = large_batch[-batch_size:].clone()

        print(
            f"\n[INFO] Processing batch size {batch_size} (sub-batch from position {max_batch_size - batch_size} to {max_batch_size - 1})...")

        try:
            result = compute_entropy_for_sub_batch(
                model, tokenizer, sae_weights, sub_batch, layer_idx,
                all_features, site
            )

            results_by_batch_size[batch_size] = result

            print(f"  Active features: {result['num_active_features']}")
            if result['num_active_features'] > 0:
                avg_feat_entropy = np.mean(
                    list(result['feature_entropies'].values()))
                print(f"  Avg feature entropy: {avg_feat_entropy:.4f} bits")

        except Exception as e:
            print(f"[WARN] Error processing batch size {batch_size}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results_by_batch_size) == 0:
        print("[ERROR] No batch sizes processed successfully.")
        return

    # 6. Create plot
    print(f"\n[INFO] Creating plot...")
    plots_dir = Path(f"entropy_vs_batch_size_{site}_{unique_id}")
    plots_dir.mkdir(exist_ok=True)
    print(f"[INFO] Plots directory: {plots_dir}")

    plot_path = plot_entropy_vs_batch_size(
        results_by_batch_size, site, plots_dir)
    print(f"  Saved plot: {plot_path}")

    # 7. Save data
    output_file = Path(f"entropy_vs_batch_size_{site}_{unique_id}.pt")

    # Convert numpy arrays to lists for serialization
    serializable_results = {}
    for batch_size, result in results_by_batch_size.items():
        serializable_results[batch_size] = {
            'feature_entropies': {k: float(v) for k, v in result['feature_entropies'].items()},
            'feature_activations': {k: float(v) for k, v in result['feature_activations'].items()},
            'feature_influences': {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v.tolist()
                                   for k, v in result['feature_influences'].items()},
            'num_active_features': result['num_active_features']
        }

    output_data = {
        "results_by_batch_size": serializable_results,
        "summary": {
            "site": site,
            "unique_id": unique_id,
            "timestamp": timestamp,
            "layer": layer_idx,
            "max_batch_size": max_batch_size,
            "min_batch_size": min_batch_size,
            "step": step,
            "sub_batch_sizes": sub_batch_sizes,
            "start_idx": start_idx,
        },
        "config": {
            "threshold": THRESHOLD,
            "random_seed": random_seed,
            "total_features": n_latent,
        },
        "plots_dir": str(plots_dir),
    }

    torch.save(output_data, output_file)
    print(f"\n[INFO] Saved results to {output_file}")

    # 8. Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Processed {len(results_by_batch_size)} batch sizes")
    print(f"Batch sizes: {sorted(results_by_batch_size.keys())}")

    # Count unique features across all batch sizes
    all_features_seen = set()
    for result in results_by_batch_size.values():
        all_features_seen.update(result['feature_entropies'].keys())
    print(f"Total unique features seen: {len(all_features_seen)}")

    # For each batch size, show number of active features
    print(f"\nActive features per batch size:")
    print(f"{'Batch Size':<12} | {'Active Features':<16}")
    print("-" * 30)
    for batch_size in sorted(results_by_batch_size.keys()):
        num_active = results_by_batch_size[batch_size]['num_active_features']
        print(f"{batch_size:<12} | {num_active:<16}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Study entropy as a function of batch size")
    parser.add_argument("--site", type=str, default=None,
                        help="Site like 'resid_out_layer3'")
    parser.add_argument("--max-batch-size", type=int,
                        default=None, help="Maximum batch size (default: 128)")
    parser.add_argument("--min-batch-size", type=int,
                        default=None, help="Minimum batch size (default: 8)")
    parser.add_argument("--step", type=int, default=None,
                        help="Step size for batch sizes (default: 8)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    try:
        main(site=args.site, max_batch_size=args.max_batch_size,
             min_batch_size=args.min_batch_size, step=args.step,
             random_seed=args.random_seed)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
