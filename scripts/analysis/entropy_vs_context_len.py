"""
Study entropy as a function of context length.

For a randomly chosen large window (e.g., 128 tokens), take sub-windows of
increasing context length (8, 16, 24, ..., 128) that all end with the same
last token. For each sub-context-length, compute leading features and their
entropies.

Usage:
    python entropy_vs_context_len.py --preset pythia-70m --layer 3 \\
        --max-context-len 128 --min-context-len 8 --step 8
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

matplotlib.use("Agg")

from data_loader import load_wikitext_train_text
from feature_token_influence import process_batch_with_influence
from model_adapters import get_layer, load_model
from presets import Preset, get_preset, site_for
from sae_adapters import SAEBundle, load_sae

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
MAX_CONTEXT_LEN = 128
MIN_CONTEXT_LEN = 8
CONTEXT_LEN_STEP = 8


# --- Entropy helpers --------------------------------------------------------

def compute_feature_entropy(influence_distribution):
    eps = 1e-12
    P = (influence_distribution + eps) / (np.sum(influence_distribution) + eps)
    return scipy.stats.entropy(P, base=2)


# --- Per-sub-batch hot path -------------------------------------------------

def compute_entropy_for_sub_batch(
    model, sae: SAEBundle, sub_batch_tokens, layer_idx, all_features,
    preset: Preset, threshold: float,
):
    """Per-feature influence + entropies + last-position activations for one sub-batch."""
    feature_influences = process_batch_with_influence(
        model, sae, sub_batch_tokens, layer_idx, all_features, threshold, preset,
    )

    # Last-position activations (no grad) for plot colouring / legend ranking.
    input_ids = sub_batch_tokens.unsqueeze(0)
    layer = get_layer(model, preset, layer_idx)
    activations = []

    def hook_fn(module, inputs, output):  # noqa: ARG001
        activations.append(output[0] if isinstance(output, tuple) else output)

    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(input_ids)
        resid = activations[0]
        feats = sae.encode(resid)
        last_pos_feats = feats[0, -1, :].cpu().numpy()
        feature_activations = {
            feat_idx: float(last_pos_feats[feat_idx])
            for feat_idx in feature_influences.keys()
        }
    finally:
        handle.remove()

    feature_entropies = {
        feat_idx: compute_feature_entropy(dist)
        for feat_idx, dist in feature_influences.items()
    }

    return {
        "feature_entropies": feature_entropies,
        "feature_activations": feature_activations,
        "feature_influences": feature_influences,
        "num_active_features": len(feature_entropies),
    }


# --- Plotting ---------------------------------------------------------------

def plot_entropy_vs_context_len(results_by_context_len, site, output_dir,
                                 top_n=10, prompt_token_strs=None,
                                 grey_cap=200):
    """Plot per-feature entropy curves.

    Top-N features (by ∑ last-position activation across context lengths)
    are drawn in distinct colours; all other active features are drawn
    behind them in light grey at low alpha so the population context is
    visible without overwhelming the plot. To keep matplotlib responsive
    when many features are active, the grey background is subsampled to
    `grey_cap` features (default 200). Active feature counts and the
    sub-sampling are reported in the title.

    If `prompt_token_strs` is provided (list of decoded tokens, oldest→
    query), a panel below the entropy axis displays them with boundary
    markers and the query token highlighted.
    """
    all_feature_indices = set()
    for result in results_by_context_len.values():
        all_feature_indices.update(result["feature_entropies"].keys())
    all_feature_indices = sorted(all_feature_indices)
    context_lens = sorted(results_by_context_len.keys())
    n_total = len(all_feature_indices)

    # Rank features by total activation (summed over all context lengths)
    # BEFORE plotting, so we only draw the top N.
    feature_total_act = {
        f: sum(results_by_context_len[bs]["feature_activations"].get(f, 0.0)
               for bs in context_lens)
        for f in all_feature_indices
    }
    top_features = [f for f, _ in sorted(feature_total_act.items(),
                                         key=lambda x: x[1], reverse=True)[:top_n]]
    n_top = len(top_features)
    palette = plt.cm.tab10(np.linspace(0, 1, max(n_top, 1)))[:n_top]
    feature_color_map = {f: palette[i] for i, f in enumerate(top_features)}

    if prompt_token_strs:
        # Pre-wrap the token string into lines so we can size the figure
        # height to accommodate them. Tokens are rendered with │ boundary
        # markers (one cell = "│tok"); we greedy-pack cells onto a line
        # until a target character width is reached.
        def _show(s):
            return s.replace("Ġ", "·").replace("▁", "·").replace("\n", "↵")
        TARGET_LINE_CHARS = 110
        cells = [f"│{_show(t)}" for t in prompt_token_strs] + ["│"]
        wrapped_lines = []
        cur = ""
        for cell in cells:
            if cur and len(cur) + len(cell) > TARGET_LINE_CHARS:
                wrapped_lines.append(cur)
                cur = cell
            else:
                cur += cell
        if cur:
            wrapped_lines.append(cur)
        n_lines = len(wrapped_lines)
        # ~0.22 inches per text line + ~1 inch for headers/query line.
        tok_panel_in = 1.0 + 0.22 * n_lines
        plot_in = 8.0
        fig = plt.figure(figsize=(12, plot_in + tok_panel_in))
        gs = fig.add_gridspec(2, 1,
                              height_ratios=[plot_in, tok_panel_in],
                              hspace=0.30)
        ax = fig.add_subplot(gs[0])
        ax_tok = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax_tok = None
        wrapped_lines = None

    # Grey background: every active feature that's NOT in top-N. Subsample
    # if there are more than grey_cap of them so matplotlib doesn't choke
    # on thousands of low-alpha lines (layer 5 of pythia-70m can have ~700+
    # active features at one context length).
    top_set = set(top_features)
    grey_pool = [f for f in all_feature_indices if f not in top_set]
    grey_subsampled = False
    if len(grey_pool) > grey_cap:
        step = len(grey_pool) // grey_cap + 1
        grey_pool = grey_pool[::step][:grey_cap]
        grey_subsampled = True

    for feat_idx in grey_pool:
        entropies, sizes = [], []
        for bs in context_lens:
            r = results_by_context_len[bs]
            if feat_idx in r["feature_entropies"]:
                entropies.append(r["feature_entropies"][feat_idx])
                sizes.append(bs)
        if entropies:
            ax.plot(sizes, entropies, "-", color="lightgrey",
                    linewidth=1.0, alpha=0.6, zorder=1)

    # Top-N features on top, in colour with markers.
    for feat_idx in top_features:
        entropies, sizes = [], []
        for bs in context_lens:
            r = results_by_context_len[bs]
            if feat_idx in r["feature_entropies"]:
                entropies.append(r["feature_entropies"][feat_idx])
                sizes.append(bs)
        if entropies:
            ax.plot(sizes, entropies, "o-", color=feature_color_map[feat_idx],
                    linewidth=2, markersize=6, alpha=0.85, zorder=3)

    # Maximal-entropy reference: log2(n) for a uniform distribution.
    ref_x = np.array(context_lens)
    ax.plot(ref_x, np.log2(ref_x), "k--", linewidth=2, alpha=0.8, zorder=2)

    ax.set_xlabel("Context length", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    grey_note = (f"; grey: {len(grey_pool)} of {n_total - n_top} others"
                 + (" (subsampled)" if grey_subsampled else ""))
    ax.set_title(f"{site} — coloured: top {n_top} of {n_total} active features"
                 f"{grey_note}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=feature_color_map[f], markersize=8,
                   markeredgecolor="black", markeredgewidth=0.5,
                   linewidth=2, label=f"Feature {f}")
        for f in top_features
    ]
    legend_elems.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                   linewidth=2, label="Maximal Entropy (log₂(n))"))
    if legend_elems:
        ax.legend(handles=legend_elems, fontsize=9, loc="best", ncol=2)

    # Token panel: pre-wrapped lines (computed above) + a separate
    # bold red "query →" line for the last token.
    if ax_tok is not None and prompt_token_strs:
        ax_tok.axis("off")
        n_tok = len(prompt_token_strs)
        last = _show(prompt_token_strs[-1]) if prompt_token_strs else ""
        # Header (bold), then wrapped token block (monospace), then query
        # token highlight on its own line at the bottom.
        ax_tok.text(
            0.0, 1.0,
            f"Prompt window — {n_tok} tokens "
            f"(oldest left → query right; · = leading space, ↵ = newline)",
            ha="left", va="top", fontsize=10, fontweight="bold",
            transform=ax_tok.transAxes,
        )
        # Place the wrapped tokens just under the header; matplotlib handles
        # line spacing inside a multi-line string.
        ax_tok.text(
            0.0, 0.90, "\n".join(wrapped_lines),
            ha="left", va="top", fontsize=8, family="monospace",
            linespacing=1.4, transform=ax_tok.transAxes,
        )
        ax_tok.text(
            0.0, -0.02, f"query → │{last}│",
            ha="left", va="top", fontsize=9, family="monospace",
            fontweight="bold", color="#b22222",
            transform=ax_tok.transAxes,
        )

    if ax_tok is None:
        plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "entropy_vs_context_len.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# --- Main -------------------------------------------------------------------

def main(preset_name="pythia-70m", layer_idx=3, max_context_len=None,
         min_context_len=None, step=None, random_seed=None, threshold=None,
         output_dir="."):
    preset = get_preset(preset_name)
    site = site_for(preset, layer_idx)
    threshold = threshold if threshold is not None else preset.threshold
    max_context_len = max_context_len if max_context_len is not None else MAX_CONTEXT_LEN
    min_context_len = min_context_len if min_context_len is not None else MIN_CONTEXT_LEN
    step = step if step is not None else CONTEXT_LEN_STEP

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[INFO] entropy_vs_context_len: preset={preset.name} site={site}")
    print(f"[INFO] Threshold: {threshold}")
    print(f"{'='*60}")

    model, tokenizer = load_model(preset, DEVICE)
    sae = load_sae(preset, layer_idx, DEVICE)
    print(f"[INFO] SAE: arch={sae.arch} n_latent={sae.n_latent}")
    all_features = set(range(sae.n_latent))

    # Sample in CHARACTER space, not token space, so the same --random-seed
    # gives the same source-text window across models with different
    # tokenizers. Each preset tokenizes the shared window with its own
    # tokenizer and we take the LAST max_context_len tokens.
    text = load_wikitext_train_text()
    total_chars = len(text)
    # Generous upper bound on chars-per-token for English text. WikiText with
    # BPE / SentencePiece typically yields ~3-5 chars/token; 12x is
    # comfortably safe across all six presets.
    char_budget = max_context_len * 12
    if total_chars < char_budget:
        print(f"[ERROR] Corpus has {total_chars} chars; need {char_budget}")
        return

    if random_seed is not None:
        random.seed(random_seed); np.random.seed(random_seed)
        print(f"[INFO] Random seed: {random_seed}")

    char_start = random.randint(0, total_chars - char_budget)
    source_text = text[char_start: char_start + char_budget]
    print(f"[INFO] Char window: [{char_start}, {char_start + char_budget})  "
          f"({char_budget} chars)")

    tokens = tokenizer(source_text, return_tensors="pt")["input_ids"][0]
    n_tokens = tokens.shape[0]
    if n_tokens < max_context_len:
        print(f"[ERROR] Tokenizer produced {n_tokens} tokens from {char_budget} "
              f"chars; need >= {max_context_len}. Increase char_budget multiplier.")
        return
    large_batch = tokens[-max_context_len:].to(DEVICE)
    # Capture the actual prompt-window tokens (IDs + decoded strings) for
    # later persistence and for the figure's token panel.
    prompt_token_ids = large_batch.cpu().tolist()
    prompt_token_strs = tokenizer.convert_ids_to_tokens(prompt_token_ids)
    print(f"[INFO] Tokenized {n_tokens} tokens; using last {max_context_len}")
    print(f"[INFO] Query token: {prompt_token_strs[-1]!r}")

    sub_context_lens = list(range(min_context_len, max_context_len + 1, step))
    if max_context_len not in sub_context_lens:
        sub_context_lens.append(max_context_len)
    sub_context_lens.sort()
    print(f"[INFO] Sub-context lengths: {sub_context_lens}")

    results_by_context_len = {}
    for bs in sub_context_lens:
        sub_batch = large_batch[-bs:].clone()
        print(f"\n[INFO] Sub-context length {bs} "
              f"(positions {max_context_len - bs}..{max_context_len - 1})...")
        try:
            result = compute_entropy_for_sub_batch(
                model, sae, sub_batch, layer_idx, all_features, preset, threshold,
            )
            results_by_context_len[bs] = result
            n = result["num_active_features"]
            print(f"  active={n}", end="")
            if n:
                print(f"  avg_fH={np.mean(list(result['feature_entropies'].values())):.4f}")
            else:
                print()
        except Exception as e:
            print(f"[WARN] sub-batch {bs}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not results_by_context_len:
        print("[ERROR] No sub-context lengths processed."); return

    plots_dir = out_root / f"entropy_vs_context_len_{site}_{timestamp}"
    plot_path = plot_entropy_vs_context_len(
        results_by_context_len, site, plots_dir,
        prompt_token_strs=prompt_token_strs,
    )
    print(f"[INFO] Plot: {plot_path}")

    # Serialize
    serializable = {}
    for bs, r in results_by_context_len.items():
        serializable[bs] = {
            "feature_entropies": {int(k): float(v) for k, v in r["feature_entropies"].items()},
            "feature_activations": {int(k): float(v) for k, v in r["feature_activations"].items()},
            "feature_influences": {
                int(k): (v.cpu().numpy().tolist() if isinstance(v, torch.Tensor)
                         else v.tolist())
                for k, v in r["feature_influences"].items()
            },
            "num_active_features": r["num_active_features"],
        }

    output_file = out_root / f"entropy_vs_context_len_{site}_{timestamp}.pt"
    torch.save({
        "results_by_context_len": serializable,
        "summary": {
            "preset": preset.name, "site": site, "layer": layer_idx,
            "timestamp": timestamp,
            "max_context_len": max_context_len, "min_context_len": min_context_len,
            "step": step, "sub_context_lens": sub_context_lens,
            "char_start": char_start, "char_budget": char_budget,
            "n_tokens_total": n_tokens,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_strs": prompt_token_strs,
        },
        "config": {
            "preset": preset.name, "threshold": threshold,
            "random_seed": random_seed, "total_features": sae.n_latent,
            "sae_source": sae.source, "sae_arch": sae.arch,
        },
        "plots_dir": str(plots_dir),
    }, output_file)
    print(f"[INFO] Saved {output_file}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"Sub-context lengths processed: {sorted(results_by_context_len.keys())}")
    seen = set()
    for r in results_by_context_len.values():
        seen.update(r["feature_entropies"].keys())
    print(f"Unique active features across all sizes: {len(seen)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Study entropy as a function of (sub-)context length",
    )
    parser.add_argument("--preset", type=str, default="pythia-70m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--max-context-len", type=int, default=None)
    parser.add_argument("--min-context-len", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    try:
        main(preset_name=args.preset, layer_idx=args.layer,
             max_context_len=args.max_context_len, min_context_len=args.min_context_len,
             step=args.step, random_seed=args.random_seed,
             threshold=args.threshold, output_dir=args.output_dir)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
