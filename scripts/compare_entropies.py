"""
Compare entropy between feature influences and token vector influences.

This module compares the entropy of feature influences vs token vector influences
for the same batches and layers, testing the hypothesis that features have lower
entropy (are more structured/localized) than the token vector.

For each batch:
1. Computes influence distribution J(t,z|t') for each active leading feature
2. Normalizes to get probability: P(t'|t,z) = J(t,z|t') / sum J(t,z|t')
3. Computes entropy: H_feat = -∑ P log P (in bits, base 2)
4. Computes token vector influence R(t,z|t') = ||D_{νμ}||^2
5. Normalizes to get probability: Q(t') = R(t,z|t') / sum R(t,z|t')
6. Computes entropy: H_token = -∑ Q log Q

The conjecture is that H_token > H_feat (average), indicating that features
provide a "low entropy decomposition" of the token vector.

Usage:
    python compare_entropies.py --site resid_out_layer3 --num-batches 10
"""

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
from feature_token_influence import (
    load_sae, get_sae_weights, get_feature_activations,
    process_batch_with_influence
)

THRESHOLD = 0.2

from token_vector_influence import (
    process_batch_with_token_influence
)

# Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64
NUM_BATCHES = 10  # Default number of batches for comparison


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


def plot_batch_comparison(batch_result, batch_idx, site, output_dir):
    """
    Create a two-panel figure comparing feature and token vector entropies.
    
    Left panel: Probability distributions vs token position (one curve per feature + token vector)
    Right panel: Feature entropies as points, token entropy as dashed line
    
    Args:
        batch_result: Dict with feature_entropies, token_vector_entropy, 
                     feature_influences, token_vector_influence
        batch_idx: Batch index for labeling
        site: Site string for title
        output_dir: Directory to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get data
    feature_influences = batch_result['feature_influences']
    token_vector_influence = batch_result['token_vector_influence']
    feature_entropies = batch_result['feature_entropies']
    token_vector_entropy = batch_result['token_vector_entropy']
    feature_activations = batch_result.get('feature_activations', {})
    
    seq_len = len(token_vector_influence)
    token_positions = np.arange(seq_len)
    
    # Left panel: Probability distributions
    # Normalize token vector influence
    token_prob = normalize_influence(token_vector_influence)
    ax1.plot(token_positions, token_prob, 'k--', linewidth=2, label='Token Vector', alpha=0.7)
    
    # Normalize and plot each feature's influence
    # Sort features for consistent ordering and color assignment
    sorted_feat_indices = sorted(feature_influences.keys())
    n_features = len(sorted_feat_indices)
    
    # Get top features by activation for legend (same as right panel)
    top_feat_indices_for_legend = set()
    top_feat_list = []
    if feature_activations and n_features > 0:
        feat_activation_pairs = [(idx, feature_activations.get(idx, 0.0)) for idx in sorted_feat_indices]
        top_features = sorted(feat_activation_pairs, key=lambda x: x[1], reverse=True)[:5]
        top_feat_indices_for_legend = {idx for idx, _ in top_features}
        top_feat_list = [idx for idx, _ in top_features]  # Preserve order
    elif n_features > 0:
        # Fallback: use first 5 if no activations
        top_feat_list = sorted_feat_indices[:5]
        top_feat_indices_for_legend = set(top_feat_list)
    
    # Assign colors to top features (different color for each)
    top_feat_color_map = {}
    if len(top_feat_list) > 0:
        # Use tab10 or tab20 colormap for top features
        n_top = len(top_feat_list)
        top_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_top] if n_top <= 10 else plt.cm.tab20(np.linspace(0, 1, 20))[:n_top]
        for i, feat_idx in enumerate(top_feat_list):
            top_feat_color_map[feat_idx] = top_colors[i]
    
    # Plot all features, but only show top ones in legend
    if n_features > 0:
        for feat_idx in sorted_feat_indices:
            influence_dist = feature_influences[feat_idx]
            feat_prob = normalize_influence(influence_dist)
            
            # Use distinct color for top features, grey for others
            if feat_idx in top_feat_indices_for_legend:
                color = top_feat_color_map[feat_idx]
                label = f'Feature {feat_idx}'
                linewidth = 1.5
            else:
                color = 'lightgrey'
                label = None
                linewidth = 1.0
            
            ax1.plot(token_positions, feat_prob, '-', linewidth=linewidth, 
                    label=label, color=color, alpha=0.7)
    
    ax1.set_xlabel('Token Position', fontsize=11)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Influence Probability Distributions', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    ax1.set_xlim(0, seq_len - 1)
    
    # Right panel: Entropy comparison
    if len(feature_entropies) > 0:
        feat_indices = sorted_feat_indices  # Use same ordering as left panel
        feat_entropy_values = [feature_entropies[idx] for idx in feat_indices]
        
        # Get activation values for coloring
        if feature_activations:
            activation_values = [feature_activations.get(idx, 0.0) for idx in feat_indices]
            activation_array = np.array(activation_values)
            # Use reversed Reds colormap: high activation = red, low = light red/grey
            # Create custom colormap that goes from light grey/red to full red
            # Get top 5 features by activation for legend (same as left panel)
            feat_activation_pairs = [(idx, feature_activations.get(idx, 0.0)) for idx in feat_indices]
            top_features = sorted(feat_activation_pairs, key=lambda x: x[1], reverse=True)[:5]
            top_feat_indices = {idx for idx, _ in top_features}
            top_feat_list_right = [idx for idx, _ in top_features]  # Preserve order
            
            # Assign colors to top features (same as left panel)
            top_feat_color_map_right = {}
            n_top = len(top_feat_list_right)
            if n_top > 0:
                top_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_top] if n_top <= 10 else plt.cm.tab20(np.linspace(0, 1, 20))[:n_top]
                for i, feat_idx in enumerate(top_feat_list_right):
                    top_feat_color_map_right[feat_idx] = top_colors[i]
            
            # Plot points with different colors for top features, grey for others
            for i, idx in enumerate(feat_indices):
                act_val = activation_values[i]
                ent_val = feat_entropy_values[i]
                if idx in top_feat_indices:
                    ax2.scatter([act_val], [ent_val], s=50, alpha=0.6, 
                               color=top_feat_color_map_right[idx], edgecolors='black', linewidth=0.5, zorder=3)
                else:
                    ax2.scatter([act_val], [ent_val], s=30, alpha=0.4, 
                               color='lightgrey', edgecolors='black', linewidth=0.3, zorder=2)
        else:
            # Fallback: if no activations, use feature index on x-axis
            top_feat_list_right = feat_indices[:5]  # Fallback to first 5
            top_feat_indices = set(top_feat_list_right)
            
            # Assign colors to top features
            top_feat_color_map_right = {}
            n_top = len(top_feat_list_right)
            if n_top > 0:
                top_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_top] if n_top <= 10 else plt.cm.tab20(np.linspace(0, 1, 20))[:n_top]
                for i, feat_idx in enumerate(top_feat_list_right):
                    top_feat_color_map_right[feat_idx] = top_colors[i]
            
            # Plot points with different colors for top features, grey for others
            for i, idx in enumerate(feat_indices):
                if idx in top_feat_indices:
                    ax2.scatter([idx], [feat_entropy_values[i]], s=50, alpha=0.6, 
                               color=top_feat_color_map_right[idx], edgecolors='black', linewidth=0.5, zorder=3)
                else:
                    ax2.scatter([idx], [feat_entropy_values[i]], s=30, alpha=0.4, 
                               color='lightgrey', edgecolors='black', linewidth=0.3, zorder=2)
        
        # Create legend with only top 5 features (using same colors as left panel)
        legend_elements = []
        if feature_activations:
            # Use the color map we created above
            for idx in top_feat_list_right:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                  markerfacecolor=top_feat_color_map_right[idx], markersize=8,
                                                  markeredgecolor='black', markeredgewidth=0.5,
                                                  label=f'Feature {idx}'))
        else:
            # Use the color map we created above (already defined in else block above)
            for idx in top_feat_list_right:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=top_feat_color_map_right[idx], markersize=8,
                                                  markeredgecolor='black', markeredgewidth=0.5,
                                                  label=f'Feature {idx}'))
        
        # Add token vector to legend
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                                          label=f'Token Vector ({token_vector_entropy:.3f})'))
        ax2.legend(handles=legend_elements, fontsize=9, loc='best')
        
        # Plot token vector entropy as dashed horizontal line
        ax2.axhline(y=token_vector_entropy, color='red', linestyle='--', 
                   linewidth=2, zorder=2, alpha=0.8)
        
        # Set limits with some padding
        y_min = min(min(feat_entropy_values), token_vector_entropy) * 0.95
        y_max = max(max(feat_entropy_values), token_vector_entropy) * 1.05
        ax2.set_ylim(y_min, y_max)
        
        # Set x-axis limits if using activations
        if feature_activations and len(activation_array) > 0:
            x_min = activation_array.min() * 0.95 if activation_array.min() > 0 else activation_array.min() * 1.05
            x_max = activation_array.max() * 1.05
            ax2.set_xlim(x_min, x_max)
    else:
        # No active features, just show token entropy
        ax2.axhline(y=token_vector_entropy, color='red', linestyle='--', 
                   linewidth=2, label=f'Token Vector ({token_vector_entropy:.3f})', 
                   zorder=2, alpha=0.8)
        ax2.set_ylim(0, token_vector_entropy * 1.2)
        ax2.legend(fontsize=9, loc='best')
    
    if feature_activations:
        ax2.set_xlabel('Activation', fontsize=11)
    else:
        ax2.set_xlabel('Feature Index', fontsize=11)
    ax2.set_ylabel('Entropy (bits)', fontsize=11)
    ax2.set_title('Entropy vs Activation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{site} - Batch {batch_idx + 1}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'batch_{batch_idx:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def compare_entropies_for_batch(
    model, tokenizer, sae_weights, tokens, layer_idx, 
    all_features, site
):
    """
    Compare feature and token vector entropies for a single batch.
    
    Note: Both process_batch_with_influence and process_batch_with_token_influence
    modify the model state (embedding layer and hooks), but they restore it in their
    finally blocks, so calling them sequentially should be safe.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        sae_weights: SAE weights dictionary
        tokens: Token IDs [batch_size]
        layer_idx: Layer index to analyze
        all_features: Set of all feature indices to check (filtered by activation threshold per batch)
        site: Site string for identification
    
    Returns:
        results: Dict with:
            - feature_entropies: Dict mapping feature_idx -> entropy
            - token_vector_entropy: Scalar entropy for token vector
            - num_active_features: Number of features above threshold
            - feature_influences: Dict mapping feature_idx -> influence distribution
            - token_vector_influence: Token vector influence distribution
    """
    # Compute feature influences
    # This modifies model state but restores it in finally block
    feature_influences = process_batch_with_influence(
        model, tokenizer, sae_weights, tokens, layer_idx, 
        all_features, THRESHOLD
    )
    
    # Get activations for the last token position to use for coloring
    # We need a forward pass to get activations (without gradients)
    input_ids = tokens.unsqueeze(0)  # [1, seq_len]
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
        feats = get_feature_activations(resid, sae_weights)  # [1, seq_len, n_latent]
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
    
    # Compute token vector influence and entropy
    # This also modifies model state but restores it in finally block
    R_values, token_vector_entropy = process_batch_with_token_influence(
        model, tokenizer, tokens, layer_idx
    )
    
    return {
        "feature_entropies": feature_entropies,
        "token_vector_entropy": token_vector_entropy,
        "num_active_features": len(feature_entropies),
        "feature_influences": feature_influences,
        "feature_activations": feature_activations,
        "token_vector_influence": R_values
    }


def main(site=None, num_batches=None, random_batches=True, random_seed=None):
    """
    Main function to compare entropies across multiple batches.
    
    Args:
        site: Site string like "resid_out_layer3" (if None, uses default)
        num_batches: Number of batches to process (if None, uses NUM_BATCHES)
        random_batches: If True, select batches randomly from the corpus
        random_seed: Random seed for reproducibility (if None, uses system default)
    """
    if site is None:
        site = "resid_out_layer3"
    
    if num_batches is None:
        num_batches = NUM_BATCHES
    
    # Generate unique ID based on timestamp to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = timestamp
    
    base_dir = Path("dictionaries/pythia-70m-deduped") / site
    
    print(f"\n{'='*60}")
    print(f"[INFO] Comparing entropies for {site}")
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
    
    # Get total number of features (all features will be considered, filtered by activation per batch)
    n_latent = sae_weights["dec_w"].shape[1]
    all_features = set(range(n_latent))
    print(f"[INFO] Will process all {n_latent} features (filtered by activation threshold per batch)")
    
    # 3. Load data
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
    
    # 4. Process batches
    layer_idx = int(site.rsplit("layer", 1)[-1])
    
    print(f"\n[INFO] Processing {num_batches} batches of size {BATCH_SIZE}...")
    print(f"[INFO] Layer: {layer_idx}, Threshold: {THRESHOLD}")
    if random_batches:
        print(f"[INFO] Batch selection: Random")
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            print(f"[INFO] Random seed: {random_seed}")
    else:
        print(f"[INFO] Batch selection: Sequential (from beginning)")
    
    batch_results = []
    
    # Generate batch start indices
    max_start = total_tokens - BATCH_SIZE
    if max_start <= 0:
        print("[ERROR] Not enough tokens for batch size")
        return
    
    if random_batches:
        # Randomly select batch start positions (without replacement)
        available_starts = list(range(0, max_start + 1, BATCH_SIZE))  # Stagger by BATCH_SIZE to avoid overlap
        if len(available_starts) < num_batches:
            # If not enough non-overlapping positions, allow any position
            available_starts = list(range(max_start + 1))
        batch_start_indices = random.sample(available_starts, min(num_batches, len(available_starts)))
        batch_start_indices.sort()  # Sort for easier reading
    else:
        # Sequential selection from beginning
        batch_start_indices = [(batch_idx * (max_start // num_batches)) % max_start 
                              for batch_idx in range(num_batches)]
    
    for batch_idx in range(num_batches):
        start_idx = batch_start_indices[batch_idx]
        chunk = tokens[start_idx : start_idx + BATCH_SIZE].to(DEVICE)
        
        print(f"\n[INFO] Processing batch {batch_idx + 1}/{num_batches} (start_idx={start_idx})...")
        
        try:
            results = compare_entropies_for_batch(
                model, tokenizer, sae_weights, chunk, layer_idx,
                all_features, site
            )
            
            batch_results.append({
                "batch_idx": batch_idx,
                "start_idx": start_idx,
                **results
            })
            
            print(f"  Active features: {results['num_active_features']}")
            if results['num_active_features'] > 0:
                avg_feat_entropy = np.mean(list(results['feature_entropies'].values()))
                print(f"  Avg feature entropy: {avg_feat_entropy:.4f} bits")
            print(f"  Token vector entropy: {results['token_vector_entropy']:.4f} bits")
            
        except Exception as e:
            print(f"[WARN] Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(batch_results) == 0:
        print("[ERROR] No batches processed successfully.")
        return
    
    # 5. Create plots for each batch
    print(f"\n[INFO] Creating plots...")
    plots_dir = Path(f"entropy_plots_{site}_{unique_id}")
    plots_dir.mkdir(exist_ok=True)
    print(f"[INFO] Plots directory: {plots_dir}")
    
    plot_paths = []
    for br in batch_results:
        batch_idx = br['batch_idx']
        plot_path = plot_batch_comparison(br, batch_idx, site, plots_dir)
        plot_paths.append(plot_path)
        print(f"  Saved plot for batch {batch_idx + 1}: {plot_path}")
    
    print(f"[INFO] Saved {len(plot_paths)} plots to {plots_dir}/")
    
    # Save batch index file for easy reference
    batch_index_file = plots_dir / "batch_index.json"
    batch_index_data = {
        "site": site,
        "unique_id": unique_id,
        "timestamp": timestamp,
        "num_batches": len(batch_results),
        "batch_size": BATCH_SIZE,
        "random_batches": random_batches,
        "random_seed": random_seed,
        "batches": [
            {
                "batch_idx": br['batch_idx'],
                "start_idx": br['start_idx'],
                "plot_file": f"batch_{br['batch_idx']:03d}.png",
                "num_active_features": br['num_active_features'],
                "token_vector_entropy": float(br['token_vector_entropy']),
                "avg_feature_entropy": float(np.mean(list(br['feature_entropies'].values()))) if br['num_active_features'] > 0 else None
            }
            for br in batch_results
        ]
    }
    
    with open(batch_index_file, 'w') as f:
        json.dump(batch_index_data, f, indent=2)
    print(f"[INFO] Saved batch index to {batch_index_file}")
    
    # Also save a simple text file for quick reference
    batch_index_txt = plots_dir / "batch_index.txt"
    with open(batch_index_txt, 'w') as f:
        f.write(f"Batch Index for {site}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Unique ID: {unique_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of batches: {len(batch_results)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Random selection: {random_batches}\n")
        if random_seed is not None:
            f.write(f"Random seed: {random_seed}\n")
        f.write("\n")
        f.write(f"{'Batch':<8} | {'Start Index':<12} | {'Active Features':<16} | {'Token Entropy':<14} | {'Avg Feat Entropy':<16}\n")
        f.write("-" * 80 + "\n")
        for br in batch_results:
            batch_idx = br['batch_idx']
            start_idx = br['start_idx']
            num_active = br['num_active_features']
            token_ent = br['token_vector_entropy']
            avg_feat_ent = np.mean(list(br['feature_entropies'].values())) if num_active > 0 else np.nan
            avg_feat_str = f"{avg_feat_ent:.4f}" if not np.isnan(avg_feat_ent) else "N/A"
            f.write(f"{batch_idx:<8} | {start_idx:<12} | {num_active:<16} | {token_ent:<14.4f} | {avg_feat_str:<16}\n")
    print(f"[INFO] Saved batch index (text) to {batch_index_txt}")
    
    # 6. Aggregate and analyze results
    print(f"\n{'='*60}")
    print("Entropy Comparison Results")
    print(f"{'='*60}")
    
    # Collect all feature entropies
    all_feature_entropies = []
    token_vector_entropies = []
    num_active_per_batch = []
    
    for br in batch_results:
        if br['num_active_features'] > 0:
            all_feature_entropies.extend(br['feature_entropies'].values())
        token_vector_entropies.append(br['token_vector_entropy'])
        num_active_per_batch.append(br['num_active_features'])
    
    # Statistics
    if all_feature_entropies:
        mean_feat_entropy = np.mean(all_feature_entropies)
        std_feat_entropy = np.std(all_feature_entropies)
        min_feat_entropy = np.min(all_feature_entropies)
        max_feat_entropy = np.max(all_feature_entropies)
    else:
        mean_feat_entropy = std_feat_entropy = min_feat_entropy = max_feat_entropy = np.nan
    
    mean_token_entropy = np.mean(token_vector_entropies)
    std_token_entropy = np.std(token_vector_entropies)
    min_token_entropy = np.min(token_vector_entropies)
    max_token_entropy = np.max(token_vector_entropies)
    
    print(f"\nFeature Entropies (over {len(all_feature_entropies)} active features):")
    print(f"  Mean: {mean_feat_entropy:.4f} ± {std_feat_entropy:.4f} bits")
    print(f"  Range: [{min_feat_entropy:.4f}, {max_feat_entropy:.4f}] bits")
    
    print(f"\nToken Vector Entropies (over {len(token_vector_entropies)} batches):")
    print(f"  Mean: {mean_token_entropy:.4f} ± {std_token_entropy:.4f} bits")
    print(f"  Range: [{min_token_entropy:.4f}, {max_token_entropy:.4f}] bits")
    
    if not np.isnan(mean_feat_entropy):
        entropy_diff = mean_token_entropy - mean_feat_entropy
        print(f"\nDifference (Token Vector - Feature Average):")
        print(f"  {entropy_diff:.4f} bits")
        if entropy_diff > 0:
            print(f"  → Token vector has higher entropy (less localized)")
        else:
            print(f"  → Features have higher entropy (less localized)")
    
    print(f"\nAverage active features per batch: {np.mean(num_active_per_batch):.1f}")
    
    # Per-batch comparison
    print(f"\n{'='*60}")
    print("Per-Batch Comparison:")
    print(f"{'='*60}")
    print(f"{'Batch':<8} | {'Active':<8} | {'Feat Entropy (avg)':<20} | {'Token Entropy':<15} | {'Diff':<10}")
    print("-" * 75)
    
    for br in batch_results:
        batch_idx = br['batch_idx']
        num_active = br['num_active_features']
        
        if num_active > 0:
            avg_feat = np.mean(list(br['feature_entropies'].values()))
        else:
            avg_feat = np.nan
        
        token_ent = br['token_vector_entropy']
        diff = token_ent - avg_feat if not np.isnan(avg_feat) else np.nan
        
        feat_str = f"{avg_feat:.4f}" if not np.isnan(avg_feat) else "N/A"
        diff_str = f"{diff:.4f}" if not np.isnan(diff) else "N/A"
        
        print(f"{batch_idx:<8} | {num_active:<8} | {feat_str:<20} | {token_ent:<15.4f} | {diff_str:<10}")
    
    # Save results
    output_file = Path(f"entropy_comparison_{site}_{unique_id}.pt")
    output_data = {
        "batch_results": batch_results,
        "summary": {
            "site": site,
            "unique_id": unique_id,
            "timestamp": timestamp,
            "layer": layer_idx,
            "num_batches": len(batch_results),
            "mean_feature_entropy": float(mean_feat_entropy) if not np.isnan(mean_feat_entropy) else None,
            "std_feature_entropy": float(std_feat_entropy) if not np.isnan(std_feat_entropy) else None,
            "mean_token_vector_entropy": float(mean_token_entropy),
            "std_token_vector_entropy": float(std_token_entropy),
            "entropy_difference": float(entropy_diff) if not np.isnan(mean_feat_entropy) else None,
        },
        "config": {
            "threshold": THRESHOLD,
            "batch_size": BATCH_SIZE,
            "total_features": n_latent,
            "random_batches": random_batches,
            "random_seed": random_seed,
        },
        "plots_dir": str(plots_dir),
        "batch_start_indices": [br['start_idx'] for br in batch_results]  # Easy access to batch positions
    }
    
    torch.save(output_data, output_file)
    print(f"\n[INFO] Saved results to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare feature and token vector entropies")
    parser.add_argument("--site", type=str, default=None, help="Site like 'resid_out_layer3'")
    parser.add_argument("--num-batches", type=int, default=None, help="Number of batches to process")
    parser.add_argument("--sequential-batches", action="store_false", dest="random_batches",
                       help="Select batches sequentially from beginning instead of randomly (default: random)")
    parser.add_argument("--random-seed", type=int, default=None,
                       help="Random seed for reproducible random batch selection (default: random batches)")
    
    args = parser.parse_args()
    
    try:
        main(site=args.site, num_batches=args.num_batches,
             random_batches=args.random_batches, random_seed=args.random_seed)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

