"""Per-feature view: continuum ranking + concrete J(t') profile contrast.

Tier averages hide most of the signal in this dataset. Two outputs:

  1. ``feature_continuum_ctx<NNN>.png``
     Horizontal bar plot of mean H_pos per feature at the chosen ctx_len,
     sorted ascending. One bar per feature, coloured by tier. Each bar
     annotated with the per-feature std. Reveals that the ordering is a
     continuum, not four discrete tiers.

  2. ``feature_profile_contrast_ctx<NNN>.png``
     Two-panel figure showing the average normalised J_a(t') profile for
     a *most-localised* feature and a *most-spread* feature side by side.
     Defaults to feature 11372 (Iran token) vs 13603 (epistemic hedging
     abstract), the two extremes of the continuum at CTX_LEN=128. The
     x-axis is "distance from peak" (the activating-token position
     anchored at 0, preceding tokens negative); y-axis is mean
     normalised J / J_max with ±1 std envelope.

Use cases:
  - Make the "individual feature ranking is sharper than tier means"
    point with one figure.
  - Show why 11372 and 13603 are the two extremes — their J profiles
    look qualitatively different.
"""

from __future__ import annotations

import argparse
import json
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


def load_csvs(expt_dir: Path, ctx_len: int) -> pd.DataFrame:
    frames = []
    for tier in TIER_ORDER:
        path = expt_dir / f"ctx{ctx_len:03d}_{tier}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no ctx{ctx_len:03d}_*.csv under {expt_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    return df


def load_influences(expt_dir: Path, ctx_len: int) -> dict[tuple[int, int], np.ndarray]:
    """Return {(feature_id, top_act_rank): J_vec} from per-tier NPZs."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for tier in TIER_ORDER:
        npz_path = expt_dir / f"influences_ctx{ctx_len:03d}_{tier}.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        for k in z.files:
            fid_str, rank_str = k.split("_")
            out[(int(fid_str), int(rank_str))] = z[k]
    return out


def plot_continuum(df: pd.DataFrame, ctx_len: int, output: Path,
                   manifest: dict, descriptions: dict[int, str]):
    ok = df[df["status"] == "ok"]
    agg = (ok.groupby(["tier", "feature_id"], observed=True)
             .agg(mean=("H_pos_bits", "mean"),
                  std=("H_pos_bits", "std"),
                  n=("H_pos_bits", "size"))
             .reset_index()
             .sort_values("mean").reset_index(drop=True))

    n = len(agg)
    # Adaptive row height: tighter packing for large feature counts.
    # Below ~30 features keep the original spacious layout with descriptions;
    # above that, drop descriptions, shrink the row, and shrink the font.
    if n <= 30:
        row_h, fontsize, edge_lw, err_lw, show_desc = 0.32, 9, 0.5, 1.0, True
    elif n <= 80:
        row_h, fontsize, edge_lw, err_lw, show_desc = 0.16, 7, 0.3, 0.6, False
    elif n <= 200:
        row_h, fontsize, edge_lw, err_lw, show_desc = 0.10, 5, 0.0, 0.3, False
    else:
        row_h, fontsize, edge_lw, err_lw, show_desc = 0.08, 4, 0.0, 0.25, False

    fig_h = max(5.0, row_h * n + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_h))
    ys = np.arange(n)
    colors = [TIER_COLORS[t] for t in agg["tier"]]
    ax.barh(ys, agg["mean"], xerr=agg["std"], color=colors,
            edgecolor="black", linewidth=edge_lw, alpha=0.85,
            error_kw={"linewidth": err_lw, "ecolor": "black",
                      "capsize": 0 if n > 80 else 3})
    labels = []
    for _, r in agg.iterrows():
        if show_desc:
            d = descriptions.get(int(r["feature_id"]), "")
            if len(d) > 50:
                d = d[:47] + "…"
            labels.append(f"{int(r['feature_id'])}  ({r['tier'][0]})  {d}")
        else:
            labels.append(f"{int(r['feature_id'])} ({r['tier'][0]})")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.invert_yaxis()
    ax.axvline(np.log2(ctx_len), color="black", linestyle="--", alpha=0.55,
               label=f"log2(ctx_len)={np.log2(ctx_len):.2f}")
    ax.set_xlabel("H_pos (bits)  —  mean ± std across events", fontsize=11)
    ax.set_title(f"Per-feature H_pos continuum at CTX_LEN={ctx_len}  "
                 f"({manifest.get('preset')} layer {manifest.get('layer')})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t], alpha=0.85)
               for t in TIER_ORDER]
    ax.legend(handles + [ax.lines[-1]],
              TIER_ORDER + [f"log2({ctx_len})"], loc="lower right", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")


def plot_profile_contrast(
    influences: dict[tuple[int, int], np.ndarray],
    ctx_len: int,
    feat_low: int, feat_high: int,
    desc_low: str, desc_high: str,
    df: pd.DataFrame,
    output: Path,
    manifest: dict,
):
    """Average normalised J(t') profile for two features, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)

    rel_x = np.arange(-(ctx_len - 1), 1)  # -(ctx_len-1) ... 0 (peak at 0)

    for ax, fid, desc, color in (
        (axes[0], feat_low, desc_low, "#1f77b4"),
        (axes[1], feat_high, desc_high, "#d62728"),
    ):
        events = sorted([k for k in influences if k[0] == fid], key=lambda k: k[1])
        if not events:
            ax.text(0.5, 0.5, f"no events for feature {fid}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        # Normalise each event by its own max so profiles are comparable.
        stack = []
        for k in events:
            J = influences[k]
            if J.shape[0] != ctx_len:
                continue
            jmax = J.max() if J.max() > 0 else 1.0
            stack.append(J / jmax)
        if not stack:
            continue
        arr = np.stack(stack, axis=0)        # [n_events, ctx_len]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(rel_x, mean, "-", color=color, linewidth=2,
                label=f"mean across {arr.shape[0]} events")
        ax.fill_between(rel_x, np.maximum(mean - std, 0.0), mean + std,
                        color=color, alpha=0.20, label="±1 std")
        # H_pos summary on the figure.
        ok = df[(df["feature_id"] == fid) & (df["status"] == "ok")
                & (df["ctx_len"] == ctx_len)]
        if len(ok):
            mh = ok["H_pos_bits"].mean()
            sh = ok["H_pos_bits"].std()
            ax.text(0.02, 0.96,
                    f"H_pos = {mh:.2f} ± {sh:.2f} bits  (n={len(ok)})",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="grey", boxstyle="round,pad=0.3"))
        ax.set_title(f"feature {fid} — {desc}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("position relative to activating token (0 = peak)",
                      fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.axvline(0, color="black", linestyle=":", alpha=0.6)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9, loc="lower left")

    axes[0].set_ylabel("J(t') / max J  (log scale)", fontsize=10)
    fig.suptitle(
        f"Per-position influence profile at CTX_LEN={ctx_len}  "
        f"({manifest.get('preset')} layer {manifest.get('layer')})",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt-dir", type=Path, required=True)
    ap.add_argument("--ctx-len", type=int, default=128)
    ap.add_argument("--low-feature", type=int, default=11372,
                    help="Most-localised feature to plot (default 11372 / Iran).")
    ap.add_argument("--high-feature", type=int, default=13603,
                    help="Most-spread feature to plot (default 13603 / epistemic).")
    ap.add_argument("--continuum-only", action="store_true")
    ap.add_argument("--profile-only", action="store_true")
    args = ap.parse_args()

    expt_dir = args.expt_dir.resolve()
    ctx_len = int(args.ctx_len)
    manifest = json.loads((expt_dir / "manifest.json").read_text())
    descriptions = {int(e["feature_id"]): e.get("description", "")
                    for e in manifest.get("features", [])}

    df = load_csvs(expt_dir, ctx_len)

    if not args.profile_only:
        plot_continuum(
            df, ctx_len,
            expt_dir / "figures" / f"feature_continuum_ctx{ctx_len:03d}.png",
            manifest, descriptions,
        )

    if not args.continuum_only:
        influences = load_influences(expt_dir, ctx_len)
        desc_low = descriptions.get(args.low_feature,
                                    f"feature {args.low_feature}")
        desc_high = descriptions.get(args.high_feature,
                                     f"feature {args.high_feature}")
        plot_profile_contrast(
            influences, ctx_len,
            args.low_feature, args.high_feature, desc_low, desc_high,
            df,
            expt_dir / "figures"
            / f"feature_profile_contrast_ctx{ctx_len:03d}.png",
            manifest,
        )


if __name__ == "__main__":
    main()
