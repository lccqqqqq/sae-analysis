"""
Figure 6 — Histograms of average feature entropy for leading features across layers 0–4.
Paper: Fig. 6, caption: "Histograms of average feature entropy for leading features
across layers 0–4. The distribution shifts to higher entropy at deeper layers."

Prerequisite data:
    entropy_comparison_resid_out_layer{0..4}_<timestamp>.pt  (one file per layer)
    (produced by: python scripts/analysis/compare_entropies_multi_layer.py --layers 0 1 2 3 4)

Output:
    paper/figures/entropy_vs_activation_multilayer.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "paper/figures/entropy_vs_activation_multilayer.png"

LAYERS = [0, 1, 2, 3, 4]


def find_latest(layer):
    pattern = str(ROOT / f"entropy_comparison_resid_out_layer{layer}_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No entropy_comparison file for layer {layer}. "
                                f"Run: python scripts/analysis/compare_entropies_multi_layer.py "
                                f"--layers {' '.join(map(str, LAYERS))}")
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def main():
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(3.5 * len(LAYERS), 4), sharey=False)

    for ax, layer in zip(axes, LAYERS):
        path = find_latest(layer)
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  Layer {layer}: {Path(path).name}")

        # Collect mean entropy per feature across all batches
        feat_ent_lists = defaultdict(list)
        for br in data["batch_results"]:
            for feat_idx, ent in br.get("feature_entropies", {}).items():
                feat_ent_lists[feat_idx].append(ent)
        mean_ents = [np.mean(v) for v in feat_ent_lists.values()]

        ax.hist(mean_ents, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Mean entropy (bits)", fontsize=10)
        ax.set_ylabel("Number of features", fontsize=10)
        ax.set_title(f"Layer {layer}\n(N={len(mean_ents)})", fontsize=11, fontweight="bold")
        ax.axvline(np.mean(mean_ents), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(mean_ents):.2f} bits")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature entropy distribution across layers", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
