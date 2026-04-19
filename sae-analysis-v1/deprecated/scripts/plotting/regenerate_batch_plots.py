"""Regenerate per-batch diagnostic plots with z-order fix + decoded-text caption.

Reads the entropy_comparison_*.pt data written by
scripts/analysis/compare_entropies_multi_layer.py and re-renders each
batch's two-panel diagnostic PNG so that:

  1. Colored top-5 curves/points sit visibly on top of the grey bulk
     (explicit zorder).
  2. A caption below the figure shows the decoded 64-token batch text
     plus the top-3 most-influential input positions by token-vector
     influence, with the actual tokens.

Does NOT rerun the gradient computation -- purely re-plots from cached data.

Usage:
    python scripts/plotting/regenerate_batch_plots.py \
        --timestamp 20260414_053350 --layers 0 1 2 3 4 5
"""

import argparse
import json
from pathlib import Path
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from transformers import AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12


def normalize_influence(influence):
    total = np.sum(influence) + EPS
    return (np.asarray(influence) + EPS) / total


def find_data_file(data_dir: Path, layer: int, timestamp: str | None) -> Path:
    if timestamp:
        p = data_dir / f"entropy_comparison_resid_out_layer{layer}_{timestamp}.pt"
        if not p.exists():
            raise FileNotFoundError(f"No data file: {p}")
        return p
    pattern = str(data_dir / f"entropy_comparison_resid_out_layer{layer}_*.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No data files matching: {pattern}")
    return Path(max(matches, key=lambda f: Path(f).stat().st_mtime))


def draw_token_heatmap_rows(fig, token_strings, rows, y_start=0.30,
                            row_height=0.032, wrap_per_line=32):
    """Draw multiple rows of token text, each row color-shaded per-token.

    Each row has (optionally) a description line above the token line(s):
        F12345 • 'autointerp string describing the feature'
        [colored tokens line 1]
        [colored tokens line 2]

    rows : list of dicts with keys:
        label        : str
        color        : RGBA tuple
        probs        : np.ndarray length = len(token_strings), normalized influences
        description  : optional str (autointerp from Neuronpedia)
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_width_px = fig.get_size_inches()[0] * fig.dpi

    left_margin = 0.04
    label_width = 0.05
    text_area_left = left_margin + label_width
    text_area_right = 0.995
    fontsize = 7.0

    n = len(token_strings)
    n_lines_per_row = int(np.ceil(n / wrap_per_line))

    y_cursor = y_start
    for row in rows:
        probs = row["probs"]
        max_prob = max(probs.max(), 1e-9)
        desc = row.get("description")

        # Description line (if present) above the token block
        if desc:
            fig.text(left_margin, y_cursor, f"{row['label']}  {desc}",
                     fontsize=fontsize + 0.5,
                     color=row["color"], family="sans-serif",
                     fontweight="bold", ha="left", va="bottom")
            y_cursor -= row_height * 0.9  # leave some gap before tokens
        else:
            # Fall back to label on first token line
            row_top_y = y_cursor
            label_y = row_top_y - (n_lines_per_row - 1) * row_height / 2.0
            fig.text(left_margin, label_y, row["label"],
                     fontsize=fontsize + 1,
                     color=row["color"], family="monospace",
                     fontweight="bold", ha="left", va="center")

        # Token heatmap rows
        row_top_y = y_cursor
        for line_i in range(n_lines_per_row):
            y = row_top_y - line_i * row_height
            x = text_area_left if not desc else left_margin
            start = line_i * wrap_per_line
            end = min(start + wrap_per_line, n)
            for j in range(start, end):
                tok = token_strings[j]
                raw = float(probs[j] / max_prob)
                intensity = raw ** 0.6
                bg_rgba = (
                    1 - intensity * (1 - row["color"][0]),
                    1 - intensity * (1 - row["color"][1]),
                    1 - intensity * (1 - row["color"][2]),
                    0.95,
                )
                text_color = "black" if intensity < 0.55 else "white"
                t = fig.text(x, y, tok, fontsize=fontsize, family="monospace",
                             ha="left", va="center", color=text_color,
                             bbox=dict(boxstyle="square,pad=0.08",
                                       facecolor=bg_rgba, edgecolor="none"))
                bbox_px = t.get_window_extent(renderer=renderer)
                width_fig = bbox_px.width / fig_width_px + 0.0015
                x += width_fig
                if x > text_area_right:
                    break

        y_cursor -= n_lines_per_row * row_height + 0.012


def pick_top_features(feature_activations: dict, n_top: int = 5):
    if not feature_activations:
        return [], {}
    top = sorted(feature_activations.items(), key=lambda x: x[1], reverse=True)[:n_top]
    top_ids = [idx for idx, _ in top]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(top_ids)]
    color_map = {idx: colors[i] for i, idx in enumerate(top_ids)}
    return top_ids, color_map


def plot_batch(batch_result, site, batch_tokens, tokenizer, output_path: Path,
               explanations=None):
    feature_influences = batch_result["feature_influences"]
    token_vector_influence = batch_result["token_vector_influence"]
    feature_entropies = batch_result["feature_entropies"]
    token_vector_entropy = batch_result["token_vector_entropy"]
    feature_activations = batch_result.get("feature_activations", {})
    start_idx = batch_result["start_idx"]
    batch_idx = batch_result["batch_idx"]

    sorted_feat_ids = sorted(feature_influences.keys())
    top_ids, color_map = pick_top_features(feature_activations, n_top=5)
    top_set = set(top_ids)

    token_vector_influence = np.asarray(token_vector_influence)
    seq_len = len(token_vector_influence)
    positions = np.arange(seq_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 11))

    # --- Left panel: influence probability distributions
    # zorder=1: grey bulk
    for feat_idx in sorted_feat_ids:
        if feat_idx in top_set:
            continue
        prob = normalize_influence(feature_influences[feat_idx])
        ax1.plot(positions, prob, color="lightgrey", alpha=0.35, linewidth=0.8, zorder=1)

    # zorder=2: token vector
    tok_prob = normalize_influence(token_vector_influence)
    ax1.plot(positions, tok_prob, "k--", linewidth=2.0, alpha=0.9,
             zorder=2, label="Token Vector")

    # zorder=3: top-5 colored features
    for feat_idx in top_ids:
        prob = normalize_influence(feature_influences[feat_idx])
        ax1.plot(positions, prob, "-", color=color_map[feat_idx],
                 linewidth=1.8, alpha=0.95, zorder=3,
                 label=f"Feature {feat_idx}")

    ax1.set_xlabel("Token Position", fontsize=11)
    ax1.set_ylabel("Probability", fontsize=11)
    ax1.set_title("Influence Probability Distributions", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, seq_len - 1)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.85,
               ncol=1, handlelength=1.5)

    # --- Right panel: entropy vs activation
    feat_ids_for_scatter = sorted_feat_ids
    xs, ys, colors_list = [], [], []
    for feat_idx in feat_ids_for_scatter:
        if feat_idx in top_set:
            continue
        act = feature_activations.get(feat_idx, 0.0)
        ent = feature_entropies[feat_idx]
        xs.append(act)
        ys.append(ent)
    if xs:
        ax2.scatter(xs, ys, s=25, color="lightgrey", alpha=0.25,
                    edgecolors="none", zorder=1)

    ax2.axhline(y=token_vector_entropy, color="red", linestyle="--",
                linewidth=2, zorder=2, alpha=0.85,
                label=f"Token Vector ({token_vector_entropy:.3f})")

    for feat_idx in top_ids:
        act = feature_activations.get(feat_idx, 0.0)
        ent = feature_entropies[feat_idx]
        ax2.scatter([act], [ent], s=70, color=color_map[feat_idx],
                    alpha=0.9, edgecolors="black", linewidth=0.6, zorder=3,
                    label=f"Feature {feat_idx}")

    if feature_activations:
        ax2.set_xlabel("Activation", fontsize=11)
    else:
        ax2.set_xlabel("Feature Index", fontsize=11)
    ax2.set_ylabel("Entropy (bits)", fontsize=11)
    ax2.set_title("Entropy vs Activation", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="best", framealpha=0.85, handlelength=1.5)

    fig.suptitle(f"{site} - Batch {batch_idx + 1}  (start_idx={start_idx})",
                 fontsize=13, fontweight="bold")

    # Per-token display strings: strip whitespace for layout, show underscore for leading-space
    token_strings = []
    for tid in batch_tokens:
        s = tokenizer.decode([int(tid)])
        # Replace leading space with a visible glyph
        if s.startswith(" "):
            s = "·" + s[1:]
        if s == "":
            s = "␀"
        token_strings.append(s)

    # Build one row per top-feature, plus one for token vector
    rows = []
    tok_probs = normalize_influence(token_vector_influence)
    rows.append({
        "label": "TokVec",
        "color": (0.15, 0.15, 0.15, 1.0),  # dark grey/near-black
        "probs": tok_probs,
        "description": "token vector reconstruction (baseline)",
    })
    layer_key = f"layer_{site.rsplit('layer', 1)[-1]}"
    for feat_idx in top_ids:
        c = color_map[feat_idx]
        if len(c) == 3:
            c = (c[0], c[1], c[2], 1.0)
        desc = None
        if explanations and layer_key in explanations:
            feat_entry = explanations[layer_key].get(str(feat_idx)) or \
                         explanations[layer_key].get(int(feat_idx))
            if feat_entry and feat_entry.get("explanations"):
                raw_desc = feat_entry["explanations"][0].get("description") or ""
                desc = raw_desc.strip()
                if len(desc) > 110:
                    desc = desc[:107] + "..."
        rows.append({
            "label": f"F{feat_idx}",
            "color": c,
            "probs": normalize_influence(feature_influences[feat_idx]),
            "description": desc or "(no autointerp)",
        })

    # With descriptions above each row, need more vertical space.
    # 6 rows × (1 desc line + 2 token lines) = 18 lines → figsize (14, 10)
    # Leave plots occupying top ~38% of figure.
    plt.subplots_adjust(bottom=0.62, top=0.95, left=0.07, right=0.98, wspace=0.22)

    draw_token_heatmap_rows(fig, token_strings, rows,
                            y_start=0.56, row_height=0.028,
                            wrap_per_line=32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)  # no bbox_inches='tight' so our fig.text positions stay
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Timestamp suffix (default: most recent per layer)")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--batches", type=int, nargs="+", default=None,
                        help="Subset of batch indices (default: all)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--plots-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--explanations", type=Path,
                        default=ROOT / "data" / "neuronpedia_explanations.json",
                        help="JSON with Neuronpedia explanations. Pass empty to disable.")
    args = parser.parse_args()

    explanations = None
    if args.explanations and args.explanations.exists():
        with open(args.explanations) as f:
            explanations = json.load(f)
        total = sum(len(v) for v in explanations.values())
        print(f"[INFO] Loaded {total} feature explanations from {args.explanations}")
    else:
        print("[WARN] No explanations file found; rows will not be annotated.")

    print(f"[INFO] Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    corpus_path = ROOT / "wikitext-2-train.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    print(f"[INFO] Tokenizing corpus {corpus_path}...")
    text = corpus_path.read_text(encoding="utf-8")
    all_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    print(f"[INFO] Corpus has {all_tokens.shape[0]:,} tokens")

    total_plots = 0
    for layer in args.layers:
        site = f"resid_out_layer{layer}"
        data_file = find_data_file(args.data_dir, layer, args.timestamp)
        print(f"\n[INFO] Layer {layer}: loading {data_file.name}")
        data = torch.load(data_file, map_location="cpu", weights_only=False)
        batch_results = data["batch_results"]

        timestamp = data["summary"].get("timestamp") or data_file.stem.split("_")[-1]
        plots_subdir = args.plots_dir / f"entropy_plots_{site}_{timestamp}"

        batch_indices = args.batches if args.batches else list(range(len(batch_results)))
        for bi in batch_indices:
            if bi >= len(batch_results):
                print(f"  [WARN] batch_idx {bi} out of range ({len(batch_results)})")
                continue
            br = batch_results[bi]
            start_idx = br["start_idx"]
            batch_tokens = all_tokens[start_idx: start_idx + 64]
            out_path = plots_subdir / f"batch_{bi:03d}.png"
            plot_batch(br, site, batch_tokens, tokenizer, out_path,
                       explanations=explanations)
            total_plots += 1
            if (bi + 1) % 10 == 0 or bi == batch_indices[-1]:
                print(f"  Batch {bi+1}/{len(batch_indices)} -> {out_path}")

    print(f"\n[INFO] Regenerated {total_plots} plots total.")


if __name__ == "__main__":
    main()
