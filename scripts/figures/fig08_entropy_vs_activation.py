"""
Figure 8 — Averaged entropy versus activation probability for 500 leading features at layer 5.
Paper: Fig. 8, caption: "Averaged entropy versus activation probability for 500 leading
features at layer 5. The mean entropy is 4.054 bits."

Prerequisite data:
    feature_token_influence_resid_out_layer5.pt
    feature_sparsity_data_resid_out_layer5.pt
    (produced by: python scripts/analysis/feature_sparsity.py
                  python scripts/analysis/feature_token_influence.py)

Output:
    paper/figures/entropy_vs_activation.png
"""

import torch
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = "resid_out_layer5"
INFLUENCE_FILE = ROOT / f"feature_token_influence_{SITE}.pt"
SPARSITY_FILE = ROOT / f"feature_sparsity_data_{SITE}.pt"
OUT_FILE = ROOT / "paper/figures/entropy_vs_activation.png"


def compute_entropy(influence_dist):
    eps = 1e-12
    p = np.array(influence_dist, dtype=float) + eps
    p /= p.sum()
    return stats.entropy(p, base=2)


def main():
    influence_data = torch.load(INFLUENCE_FILE, map_location="cpu", weights_only=False)
    sparsity_data = torch.load(SPARSITY_FILE, map_location="cpu", weights_only=False)

    feature_influences = influence_data["feature_influences"]
    feature_counts = sparsity_data["feature_counts"].cpu().numpy()
    total_tokens = sparsity_data.get("total_tokens", len(feature_counts))

    avg_entropies, activation_probs = [], []
    for feat_idx, fd in feature_influences.items():
        if "all_influences" not in fd:
            continue
        ents = [compute_entropy(inf) for inf in fd["all_influences"]]
        avg_entropies.append(float(np.mean(ents)))
        act_prob = float(feature_counts[feat_idx]) / total_tokens if feat_idx < len(feature_counts) else 0.0
        activation_probs.append(act_prob)

    avg_entropies = np.array(avg_entropies)
    activation_probs = np.array(activation_probs)
    mean_ent = avg_entropies.mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(activation_probs, avg_entropies, alpha=0.5, s=20,
               edgecolors="black", linewidth=0.4, color="steelblue")
    ax.set_xlabel("Activation probability", fontsize=12)
    ax.set_ylabel("Mean entropy (bits)", fontsize=12)
    ax.set_title(f"Entropy vs activation probability — {SITE}\n"
                 f"N={len(avg_entropies)}, mean entropy = {mean_ent:.3f} bits", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
