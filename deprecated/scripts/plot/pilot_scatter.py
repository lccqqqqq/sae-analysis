"""Two-panel pilot figure: H_pos vs feature level.

Reads the CSV produced by scripts/analysis/feature_pilot_panel.py and
emits the central figure of the pilot:

  Left panel  — categorical tier on the x-axis (token / phrase / concept
                / abstract), H_pos_mean on y. One point per feature, with
                an error bar of H_pos_std and a feature-id label.
  Right panel — H_vocab on the x-axis (continuous unsupervised proxy for
                feature level), H_pos_mean on y. Point colour by tier,
                size by activation density.

The dashed reference line on both panels is log2(context_len), the
maximal positional entropy attainable for the gradient-influence
distribution.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

TIER_ORDER = ["token", "phrase", "concept", "abstract"]
TIER_COLORS = {
    "token":    "#1f77b4",
    "phrase":   "#2ca02c",
    "concept":  "#ff7f0e",
    "abstract": "#d62728",
}


def _density_to_size(density: float | None, lo: float = 1e-5, hi: float = 1e-1) -> float:
    if density is None or density <= 0 or math.isnan(density):
        return 30.0
    # log-scale density into [30, 220].
    x = (math.log10(density) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    x = max(0.0, min(1.0, x))
    return 30.0 + 190.0 * x


def plot_pilot(csv_path: Path, output_path: Path, context_len: int):
    df = pd.read_csv(csv_path)
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    df = df.sort_values("tier")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    log2_T = math.log2(context_len)

    # Left: categorical tier
    tier_to_x = {t: i for i, t in enumerate(TIER_ORDER)}
    rng = np.random.default_rng(0)
    for _, row in df.iterrows():
        x = tier_to_x[row["tier"]] + rng.uniform(-0.12, 0.12)
        y = row["H_pos_mean"]
        yerr = row["H_pos_std"]
        ax_l.errorbar(
            x, y, yerr=yerr, fmt="o", capsize=3,
            color=TIER_COLORS[row["tier"]], markersize=8,
            markeredgecolor="black", markeredgewidth=0.5,
        )
        ax_l.annotate(
            f"{int(row['feature_id'])}", xy=(x, y),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    ax_l.axhline(log2_T, color="black", linestyle="--", alpha=0.6,
                 label=f"log2(T)={log2_T:.2f}")
    ax_l.set_xticks(list(tier_to_x.values()))
    ax_l.set_xticklabels(TIER_ORDER)
    ax_l.set_xlabel("user-assigned tier", fontsize=12)
    ax_l.set_ylabel("H_pos (bits, mean ± std over events)", fontsize=12)
    ax_l.set_title("H_pos vs categorical tier", fontsize=12, fontweight="bold")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=9)

    # Right: continuous H_vocab
    for tier in TIER_ORDER:
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        sizes = [_density_to_size(d) for d in sub["density"].tolist()]
        ax_r.scatter(
            sub["H_vocab"], sub["H_pos_mean"],
            s=sizes, c=TIER_COLORS[tier], alpha=0.85,
            edgecolors="black", linewidths=0.5, label=tier,
        )
    for _, row in df.iterrows():
        if pd.notna(row["H_vocab"]):
            ax_r.annotate(
                f"{int(row['feature_id'])}",
                xy=(row["H_vocab"], row["H_pos_mean"]),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
            )
    ax_r.axhline(log2_T, color="black", linestyle="--", alpha=0.6,
                 label=f"log2(T)={log2_T:.2f}")
    ax_r.set_xlabel("H_vocab (bits) — continuous abstraction proxy", fontsize=12)
    ax_r.set_ylabel("H_pos (bits, mean over events)", fontsize=12)
    ax_r.set_title("H_pos vs H_vocab; size = log10 density",
                   fontsize=12, fontweight="bold")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=9, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True,
                    help="CSV from feature_pilot_panel.py")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--context-len", type=int, default=64,
                    help="Context length used in the pilot (for the log2 ref line).")
    args = ap.parse_args()

    output = args.output or args.csv.with_name(args.csv.stem + "_scatter.png")
    plot_pilot(args.csv, output, context_len=args.context_len)


if __name__ == "__main__":
    main()
