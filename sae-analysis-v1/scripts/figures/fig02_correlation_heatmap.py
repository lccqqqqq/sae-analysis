"""
Figure 2 — Feature correlation matrix heatmap for leading features at layer 3.
Paper: Fig. 2, caption: "Feature correlation matrix C_ab for leading features at
layer 3. The matrix is approximately diagonal."

Prerequisite data:
    correlation_matrix_resid_out_layer3.pt
    (produced by: python scripts/analysis/compute_correlations.py)

Output:
    paper/figures/correlation_heatmap.png
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "correlation_matrix_resid_out_layer3.pt"
OUT_FILE = ROOT / "paper/figures/correlation_heatmap.png"

N_FEATURES = 100  # Show top-N leading features in the heatmap


def main():
    data = torch.load(DATA_FILE, map_location="cpu", weights_only=False)
    cov = data["covariance_matrix"]  # (n_leading, n_leading) tensor

    n = min(N_FEATURES, cov.shape[0])
    sub = cov[:n, :n].numpy()

    vmax = np.percentile(np.abs(sub), 99)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Covariance $C_{ab}$")
    ax.set_xlabel(f"Feature index (top {n})", fontsize=11)
    ax.set_ylabel(f"Feature index (top {n})", fontsize=11)
    ax.set_title("Feature correlation matrix — layer 3", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
