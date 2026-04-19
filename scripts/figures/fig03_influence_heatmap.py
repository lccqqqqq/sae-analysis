"""
Figure 3 — Influence distribution J_a(t,n|t') for Feature 531 across batches at layer 3.
Paper: Fig. 3, caption: "Influence distribution J_a(t,n|t') for Feature 531 across
45 batches at layer 3."

Prerequisite data:
    feature_token_influence_resid_out_layer3.pt
    (produced by: python scripts/analysis/feature_token_influence.py)

Output:
    paper/figures/influence_heatmap_0.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "feature_token_influence_resid_out_layer3.pt"
OUT_FILE = ROOT / "paper/figures/influence_heatmap_0.png"

FEATURE_IDX = 531  # Feature shown in the paper


def main():
    data = torch.load(DATA_FILE, map_location="cpu", weights_only=False)
    feature_influences = data["feature_influences"]

    if FEATURE_IDX not in feature_influences:
        available = sorted(feature_influences.keys())
        print(f"Feature {FEATURE_IDX} not found. Available: {available[:10]} ...")
        feat_idx = available[0]
        print(f"Using feature {feat_idx} instead.")
    else:
        feat_idx = FEATURE_IDX

    fdata = feature_influences[feat_idx]
    all_influences = [np.array(inf) for inf in fdata["all_influences"]]
    n_batches = len(all_influences)

    # Pad/trim to common length
    seq_len = max(len(x) for x in all_influences)
    matrix = np.zeros((n_batches, seq_len))
    for i, inf in enumerate(all_influences):
        matrix[i, :len(inf)] = inf

    mean_inf = matrix.mean(axis=0)
    std_inf = matrix.std(axis=0)
    positions = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(positions, mean_inf, color="steelblue", linewidth=1.5, label="Mean influence")
    ax.fill_between(positions, mean_inf - std_inf, mean_inf + std_inf,
                    alpha=0.25, color="steelblue", label="±1 std dev")
    ax.set_xlabel("Input token position (0 = furthest, end = feature position)", fontsize=11)
    ax.set_ylabel("Influence $J_a(t,n \\mid t')$", fontsize=11)
    ax.set_title(f"Feature {feat_idx} — influence distribution over {n_batches} batches (layer 3)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
