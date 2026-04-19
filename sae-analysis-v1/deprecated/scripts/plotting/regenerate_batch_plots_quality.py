"""Trial plotter: rank features by cross-batch activation quality.

Clones scripts/plotting/regenerate_batch_plots.py but replaces the right
panel's x-axis (raw activation at last position, single-batch) with a
cross-batch quality score computed across all batches in the loaded .pt:

    mu_on(a)    = mean(activation) over batches where feature a fired
    sigma_on(a) = std (population) over those batches; NaN when num_active < 2
    margin(a)   = mu_on(a) / tau      (tau = config.threshold)
    cv(a)       = sigma_on(a) / mu_on(a)

Top-5 features are ranked by `margin` (or `cv` via --rank-by), and the same
new top-5 drives the left panel's colored curves.

Usage:
    python scripts/plotting/regenerate_batch_plots_quality.py \
        --pt-file data/entropy_comparison_resid_out_layer5_20260414_053350.pt \
        --num-batches 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from regenerate_batch_plots import (  # type: ignore
    normalize_influence,
    draw_token_heatmap_rows,
)


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12


def compute_quality_stats(batch_results, tau):
    """Return dict feat_idx -> {'num_active', 'mu_on', 'sigma_on',
    'margin', 'cv'} using cross-batch activation magnitudes."""
    on_vals: dict[int, list[float]] = {}
    for br in batch_results:
        acts = br.get("feature_activations") or {}
        for fidx, a in acts.items():
            on_vals.setdefault(int(fidx), []).append(float(a))

    stats = {}
    for fidx, vals in on_vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        n = arr.size
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=0)) if n >= 2 else float("nan")
        margin = mu / tau if tau > 0 else float("nan")
        cv = (sigma / mu) if (mu > 0 and np.isfinite(sigma)) else float("nan")
        stats[fidx] = {
            "num_active": n,
            "mu_on": mu,
            "sigma_on": sigma,
            "margin": margin,
            "cv": cv,
        }
    return stats


def pick_top_by_quality(feat_ids_this_batch, quality_stats, rank_by="margin",
                        n_top=5):
    """From features active in this batch, pick top-n by the chosen score.

    `margin` ranked descending. `cv` ranked ascending (low dispersion = best).
    Features with num_active < 2 are placed last for `cv` ranking (NaN cv).
    """
    scored = []
    for fidx in feat_ids_this_batch:
        s = quality_stats.get(fidx)
        if s is None:
            continue
        if rank_by == "margin":
            key = s["margin"] if np.isfinite(s["margin"]) else -np.inf
            scored.append((key, fidx))
            reverse = True
        elif rank_by == "cv":
            key = s["cv"] if np.isfinite(s["cv"]) else np.inf
            scored.append((key, fidx))
            reverse = False
        else:
            raise ValueError(f"Unknown rank_by: {rank_by}")
    scored.sort(key=lambda x: x[0], reverse=reverse)
    top_ids = [fidx for _, fidx in scored[:n_top]]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(top_ids)]
    color_map = {idx: colors[i] for i, idx in enumerate(top_ids)}
    return top_ids, color_map


def plot_batch_quality(batch_result, site, batch_tokens, tokenizer,
                       output_path, quality_stats, tau,
                       rank_by="margin", size_by_cv=True, explanations=None):
    feature_influences = batch_result["feature_influences"]
    token_vector_influence = np.asarray(batch_result["token_vector_influence"])
    feature_entropies = batch_result["feature_entropies"]
    token_vector_entropy = batch_result["token_vector_entropy"]
    feature_activations = batch_result.get("feature_activations", {})
    start_idx = batch_result["start_idx"]
    batch_idx = batch_result["batch_idx"]

    sorted_feat_ids = sorted(feature_influences.keys())
    top_ids, color_map = pick_top_by_quality(
        sorted_feat_ids, quality_stats, rank_by=rank_by, n_top=5)
    top_set = set(top_ids)

    seq_len = len(token_vector_influence)
    positions = np.arange(seq_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 11))

    # --- Left panel: influence probability distributions
    for feat_idx in sorted_feat_ids:
        if feat_idx in top_set:
            continue
        prob = normalize_influence(feature_influences[feat_idx])
        ax1.plot(positions, prob, color="lightgrey", alpha=0.35,
                 linewidth=0.8, zorder=1)
    tok_prob = normalize_influence(token_vector_influence)
    ax1.plot(positions, tok_prob, "k--", linewidth=2.0, alpha=0.9,
             zorder=2, label="Token Vector")
    for feat_idx in top_ids:
        prob = normalize_influence(feature_influences[feat_idx])
        s = quality_stats[feat_idx]
        ax1.plot(positions, prob, "-", color=color_map[feat_idx],
                 linewidth=1.8, alpha=0.95, zorder=3,
                 label=f"F{feat_idx} (μ/τ={s['margin']:.1f}, CV={s['cv']:.2f})")
    ax1.set_xlabel("Token Position", fontsize=11)
    ax1.set_ylabel("Probability", fontsize=11)
    ax1.set_title(f"Influence Probability Distributions "
                  f"(top-5 by {rank_by})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, seq_len - 1)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.85,
               ncol=1, handlelength=1.5)

    # --- Right panel: entropy vs cross-batch margin (log x)
    xs, ys, sizes = [], [], []
    for feat_idx in sorted_feat_ids:
        if feat_idx in top_set:
            continue
        s = quality_stats.get(feat_idx)
        if s is None or not np.isfinite(s["margin"]) or s["margin"] <= 0:
            continue
        xs.append(s["margin"])
        ys.append(feature_entropies[feat_idx])
        if size_by_cv and np.isfinite(s["cv"]):
            sizes.append(25.0 / (1.0 + s["cv"]))
        else:
            sizes.append(18.0)
    if xs:
        ax2.scatter(xs, ys, s=sizes, color="lightgrey", alpha=0.3,
                    edgecolors="none", zorder=1)

    ax2.axhline(y=token_vector_entropy, color="red", linestyle="--",
                linewidth=2, zorder=2, alpha=0.85,
                label=f"Token Vector ({token_vector_entropy:.3f})")
    ax2.axvline(x=2.0, color="black", linestyle=":", linewidth=1.2,
                zorder=2, alpha=0.6, label="margin = 2 (solid-on cutoff)")

    for feat_idx in top_ids:
        s = quality_stats[feat_idx]
        if not np.isfinite(s["margin"]) or s["margin"] <= 0:
            continue
        sz = 90.0 / (1.0 + (s["cv"] if np.isfinite(s["cv"]) else 0.0)) \
            if size_by_cv else 80.0
        ax2.scatter([s["margin"]], [feature_entropies[feat_idx]],
                    s=sz, color=color_map[feat_idx],
                    alpha=0.95, edgecolors="black", linewidth=0.6, zorder=3,
                    label=f"F{feat_idx} (μ/τ={s['margin']:.1f}, "
                          f"CV={s['cv']:.2f}, n={s['num_active']})")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Cross-batch margin $\mu_{\rm on}/\tau$  (log)",
                   fontsize=11)
    ax2.set_ylabel("Entropy (bits)", fontsize=11)
    ax2.set_title(f"Entropy vs Activation Quality "
                  f"(τ={tau:.2g}, rank by {rank_by})",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=7, loc="best", framealpha=0.85, handlelength=1.5)

    fig.suptitle(f"{site} - Batch {batch_idx + 1}  (start_idx={start_idx}) "
                 f"[quality-ranked trial]",
                 fontsize=13, fontweight="bold")

    # Token heatmap rows (same as original)
    token_strings = []
    for tid in batch_tokens:
        s = tokenizer.decode([int(tid)])
        if s.startswith(" "):
            s = "·" + s[1:]
        if s == "":
            s = "␀"
        token_strings.append(s)

    rows = [{
        "label": "TokVec",
        "color": (0.15, 0.15, 0.15, 1.0),
        "probs": normalize_influence(token_vector_influence),
        "description": "token vector reconstruction (baseline)",
    }]
    layer_key = f"layer_{site.rsplit('layer', 1)[-1]}"
    for feat_idx in top_ids:
        c = color_map[feat_idx]
        if len(c) == 3:
            c = (c[0], c[1], c[2], 1.0)
        qs = quality_stats[feat_idx]
        desc = (f"μ/τ={qs['margin']:.2f}  CV={qs['cv']:.2f}  "
                f"n_active={qs['num_active']}  "
                f"act_last={feature_activations.get(feat_idx, 0.0):.2f}")
        if explanations and layer_key in explanations:
            feat_entry = explanations[layer_key].get(str(feat_idx)) or \
                         explanations[layer_key].get(int(feat_idx))
            if feat_entry and feat_entry.get("explanations"):
                raw_desc = feat_entry["explanations"][0].get("description") or ""
                extra = raw_desc.strip()
                if len(extra) > 80:
                    extra = extra[:77] + "..."
                desc = f"{desc}  |  {extra}"
        rows.append({
            "label": f"F{feat_idx}",
            "color": c,
            "probs": normalize_influence(feature_influences[feat_idx]),
            "description": desc,
        })

    plt.subplots_adjust(bottom=0.62, top=0.95, left=0.07, right=0.98, wspace=0.22)
    draw_token_heatmap_rows(fig, token_strings, rows,
                            y_start=0.56, row_height=0.028, wrap_per_line=32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-file", type=Path, required=True,
                        help="Path to entropy_comparison_*.pt")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: plots/"
                             "entropy_plots_quality_<site>_<ts>/)")
    parser.add_argument("--num-batches", type=int, default=10,
                        help="Render first N batches (default: 10)")
    parser.add_argument("--rank-by", choices=["margin", "cv"], default="margin")
    parser.add_argument("--no-size-by-cv", action="store_true",
                        help="Disable CV-based point sizing")
    parser.add_argument("--explanations", type=Path,
                        default=ROOT / "data" / "neuronpedia_explanations.json")
    args = parser.parse_args()

    pt_file = args.pt_file.resolve()
    if not pt_file.exists():
        raise FileNotFoundError(pt_file)

    print(f"[INFO] Loading {pt_file}")
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    batch_results = data["batch_results"]
    cfg = data.get("config", {})
    summary = data.get("summary", {})
    tau = float(cfg.get("threshold", 0.2))
    site = summary.get("site") or pt_file.stem.split("_202")[0].replace(
        "entropy_comparison_", "")
    ts = summary.get("timestamp") or pt_file.stem.rsplit("_", 2)[-2] + "_" + \
        pt_file.stem.rsplit("_", 1)[-1]

    print(f"[INFO] site={site} tau={tau}  batches_available={len(batch_results)}")

    out_dir = args.out_dir or (ROOT / "plots" /
                               f"entropy_plots_quality_{site}_{ts}")
    print(f"[INFO] Output dir: {out_dir}")

    print("[INFO] Computing cross-batch quality stats...")
    quality = compute_quality_stats(batch_results, tau)
    margins = np.array([s["margin"] for s in quality.values()
                        if np.isfinite(s["margin"])])
    print(f"  {len(quality)} unique features seen across all batches")
    if margins.size:
        print(f"  margin (μ_on/τ):  min={margins.min():.2f}  "
              f"median={np.median(margins):.2f}  max={margins.max():.2f}")
        frac_graze = float(np.mean(margins < 2.0))
        print(f"  fraction with margin < 2 (grazers): {frac_graze:.2%}")

    explanations = None
    if args.explanations and args.explanations.exists():
        with open(args.explanations) as f:
            explanations = json.load(f)

    print(f"[INFO] Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    corpus_path = ROOT / "wikitext-2-train.txt"
    text = corpus_path.read_text(encoding="utf-8")
    all_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    n = min(args.num_batches, len(batch_results))
    for bi in range(n):
        br = batch_results[bi]
        start_idx = br["start_idx"]
        batch_tokens = all_tokens[start_idx: start_idx + 64]
        out_path = out_dir / f"batch_{bi:03d}.png"
        plot_batch_quality(br, site, batch_tokens, tokenizer, out_path,
                           quality_stats=quality, tau=tau,
                           rank_by=args.rank_by,
                           size_by_cv=not args.no_size_by_cv,
                           explanations=explanations)
        print(f"  batch {bi:03d} -> {out_path}")

    print(f"[INFO] Wrote {n} plots to {out_dir}")


if __name__ == "__main__":
    main()
