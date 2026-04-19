"""
Figure 1 — Distribution of unique tokens activating each feature (layer 0).
Paper: Fig. 1, caption: "Distribution of the number of unique tokens activating each
feature (layer 0). The vertical axis is on a logarithmic scale."

Prerequisite data:
    feature_sparsity_data_resid_out_layer0.pt
    (produced by: python scripts/analysis/feature_sparsity.py)

Output:
    paper/figures/unique_tokens_histogram.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "feature_sparsity_data_resid_out_layer0.pt"
OUT_FILE = ROOT / "paper/figures/unique_tokens_histogram.png"


def main():
    data = torch.load(DATA_FILE, map_location="cpu", weights_only=False)
    feature_token_counts = data["feature_token_counts"]  # list of Counter

    unique_counts = [len(c) for c in feature_token_counts if len(c) > 0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(0, np.log10(max(unique_counts)), 50)
    ax.hist(unique_counts, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of unique activating tokens", fontsize=12)
    ax.set_ylabel("Number of features", fontsize=12)
    ax.set_title("Feature activation breadth — layer 0", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
