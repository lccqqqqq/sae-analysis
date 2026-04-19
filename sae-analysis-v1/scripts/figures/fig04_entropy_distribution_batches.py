"""
Figure 4 — Per-batch entropy distributions for top features at layer 3.
Paper: Fig. 4, caption: "Entropy distributions across batches for top features at
layer 3. Mean entropies range from 1.915 bits (Feature 6191) to 4.325 bits (Feature 20709)."

Prerequisite data:
    feature_token_influence_resid_out_layer3.pt
    (produced by: python scripts/analysis/feature_token_influence.py)

Output:
    paper/figures/entropy_distribution_batches.png
"""

import torch
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "feature_token_influence_resid_out_layer3.pt"
OUT_FILE = ROOT / "paper/figures/entropy_distribution_batches.png"

N_FEATURES = 6  # Number of top features to show


def compute_entropy(influence_dist):
    eps = 1e-12
    p = (np.array(influence_dist) + eps)
    p /= p.sum()
    return stats.entropy(p, base=2)


def main():
    data = torch.load(DATA_FILE, map_location="cpu", weights_only=False)
    feature_influences = data["feature_influences"]

    # Select top features by number of batches
    ranked = sorted(
        [(idx, fd["num_samples"]) for idx, fd in feature_influences.items()
         if "all_influences" in fd],
        key=lambda x: x[1], reverse=True
    )
    features_to_plot = [idx for idx, _ in ranked[:N_FEATURES]]

    n_cols = 3
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat_idx in enumerate(features_to_plot):
        fd = feature_influences[feat_idx]
        entropies = [compute_entropy(inf) for inf in fd["all_influences"]]
        entropies = np.array(entropies)
        mean_e = entropies.mean()

        ax = axes[i]
        ax.hist(entropies, bins=30, color="skyblue", edgecolor="black", alpha=0.8)
        ax.axvline(mean_e, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_e:.3f} bits")
        ax.set_xlabel("Entropy (bits)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Feature {feat_idx}  (N={len(entropies)})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(features_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-batch entropy distributions — layer 3 top features", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
