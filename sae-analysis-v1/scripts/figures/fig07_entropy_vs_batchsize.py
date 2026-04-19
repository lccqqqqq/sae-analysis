"""
Figure 7 — Feature entropy versus context length (batch size) for layer 2.
Paper: Fig. 7, caption: "Feature entropy versus context length (batch size) for layer 2.
Each curve corresponds to one leading feature. The dashed line shows the maximum
possible entropy log2(n). Feature entropies saturate well below the maximum."

Prerequisite data:
    entropy_vs_batch_size_resid_out_layer2_<timestamp>.pt
    (produced by: python scripts/analysis/entropy_vs_batch_size.py --site resid_out_layer2)

Output:
    paper/figures/entropy_vs_batchsize.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = "resid_out_layer2"
OUT_FILE = ROOT / "paper/figures/entropy_vs_batchsize.png"

N_LEGEND = 10  # Number of features to show in legend


def find_latest():
    pattern = str(ROOT / f"entropy_vs_batch_size_{SITE}_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No entropy_vs_batch_size file found for {SITE}.\n"
            f"Run: python scripts/analysis/entropy_vs_batch_size.py --site {SITE}"
        )
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def main():
    path = find_latest()
    data = torch.load(path, map_location="cpu", weights_only=False)
    results_by_batch_size = data["results_by_batch_size"]
    batch_sizes = sorted(results_by_batch_size.keys())

    # Collect all features
    all_features = set()
    for r in results_by_batch_size.values():
        all_features.update(r["feature_entropies"].keys())

    # Rank features by total activation across batch sizes
    feat_total_act = {f: sum(results_by_batch_size[bs].get("feature_activations", {}).get(f, 0)
                             for bs in batch_sizes)
                      for f in all_features}
    top_features = {f for f, _ in
                    sorted(feat_total_act.items(), key=lambda x: x[1], reverse=True)[:N_LEGEND]}

    colors = plt.cm.tab10(np.linspace(0, 1, min(N_LEGEND, 10)))
    color_map = {f: colors[i % 10] for i, f in enumerate(
        sorted(top_features, key=lambda x: feat_total_act[x], reverse=True))}

    fig, ax = plt.subplots(figsize=(8, 5))

    for feat_idx in all_features:
        xs, ys = [], []
        for bs in batch_sizes:
            ent = results_by_batch_size[bs]["feature_entropies"].get(feat_idx)
            if ent is not None:
                xs.append(bs)
                ys.append(ent)
        if xs:
            is_top = feat_idx in top_features
            ax.plot(xs, ys, "o-",
                    color=color_map.get(feat_idx, "gray"),
                    linewidth=1.5 if is_top else 0.8,
                    markersize=5 if is_top else 3,
                    alpha=0.85 if is_top else 0.3,
                    label=f"Feature {feat_idx}" if is_top else None,
                    zorder=3 if is_top else 1)

    # Maximum entropy line
    bs_arr = np.array(batch_sizes)
    ax.plot(bs_arr, np.log2(bs_arr), "k--", linewidth=2, label="Max entropy log₂(n)", zorder=4)

    ax.set_xlabel("Context length (tokens)", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title(f"Feature entropy vs context length — {SITE}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
