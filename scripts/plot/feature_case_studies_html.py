"""Per-feature case-study HTML report with token-level gradient heatmaps.

For each of a small list of cherry-picked feature ids, render one card with:

  - feature_id, heuristic tier, optional recall_alt, auto-interp
  - mean ± std H_pos  (n events used)
  - inline H_pos histogram across the feature's events
  - the top-3 max-activating contexts, each shown as the 128-token slice
    that was actually fed to the gradient pipeline. Each token is rendered
    with a background colour proportional to that position's gradient norm
    J_a(t'), per-context-normalised so the comparison within one context is
    fair. The activating-token itself is highlighted with a red border.

The slice is taken from the virtual ``corpus_tokens.pt`` produced by the
Neuronpedia adapter, indexed at ``[abs_peak - ctx_len + 1 : abs_peak + 1]``
exactly like ``topact_entropy.py`` did. This preserves any cross-event
``<bos>`` markers so the artefact is faithful to the actual gradient
computation.

Outputs (all under ``--output-dir``):

  - manifest.json                 the feature spec and pointers to source
  - feature_case_studies.html     the visualisation

Usage:
    python scripts/plot/feature_case_studies_html.py \\
        --source-expt-dir data/cherry_picked_feature_entropy/latest_neuronpedia_full \\
        --feature-ids 6658 4248 2000 15408 12612 3282 \\
        --output-dir data/feature_case_studies/<TIMESTAMP>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from presets import get_preset  # noqa: E402

TIER_COLORS = {
    "token":    "#1f77b4",
    "phrase":   "#2ca02c",
    "concept":  "#ff7f0e",
    "abstract": "#d62728",
}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def per_feature_histogram(values: np.ndarray, ctx_len: int,
                           color: str) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.0))
    bins = np.linspace(0.0, float(np.log2(ctx_len)), 25)
    ax.hist(values, bins=bins, color=color, edgecolor="black",
            linewidth=0.4, alpha=0.85)
    ax.axvline(values.mean(), color="black", linewidth=1.0)
    ax.set_xlim(0, np.log2(ctx_len))
    ax.set_xlabel("H_pos (bits)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_yticks([])
    return fig_to_b64(fig)


CMAP_NAME = "Greens"


def color_for(value_norm: float, cmap_name: str = CMAP_NAME) -> str:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    r, g, b, _ = cmap(float(value_norm))
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def text_color_for_bg(value_norm: float) -> str:
    """White text on dark cells; black on light cells.

    Threshold tuned for the ``Greens`` colormap, where everything past
    ~0.65 is dark enough that white text wins for legibility.
    """
    return "white" if value_norm > 0.65 else "#222"


def colorbar_css(cmap_name: str = CMAP_NAME, n_stops: int = 7) -> str:
    """Return a CSS linear-gradient string approximating ``cmap_name``."""
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    parts = []
    for i in range(n_stops):
        t = i / (n_stops - 1)
        r, g, b, _ = cmap(t)
        parts.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) "
                     f"{int(t*100)}%")
    return "linear-gradient(to right, " + ", ".join(parts) + ")"


def render_token_strip(token_ids: list[int], j_values: np.ndarray,
                        peak_pos: int, tokenizer) -> str:
    """Decode token IDs and emit HTML spans coloured by J value.

    Colour scale is **sqrt**: ``sqrt(J) / sqrt(J_max) = sqrt(J/J_max)``.
    Since ``J = ||∂f/∂x||²`` is the squared gradient norm, sqrt(J) is
    the gradient norm itself — i.e., the colour is linear in the
    gradient norm. Less aggressive compression than log1p; preserves
    the peak as dominant while keeping mid-magnitude tokens visible.
    """
    decoded = tokenizer.convert_ids_to_tokens(token_ids)
    j_max = float(j_values.max()) if j_values.size else 1.0
    if j_max <= 0:
        j_norm = np.zeros_like(j_values, dtype=float)
    else:
        # Per-context sqrt normalisation: sqrt(J) / sqrt(J_max).
        j_norm = np.sqrt(np.clip(j_values, 0.0, None) / j_max)
    pieces = []
    for i, (tid, t) in enumerate(zip(token_ids, decoded)):
        norm = float(j_norm[i])
        bg = color_for(norm)
        fg = text_color_for_bg(norm)
        # Render whitespace markers; escape HTML.
        shown = (t.replace("▁", "·").replace("Ġ", "·")
                 .replace("\n", "↵"))
        shown = (shown.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;"))
        border = ("border: 2px solid #b22222;" if i == peak_pos
                  else "border: 1px solid #ddd;")
        title = f"pos={i}  J={float(j_values[i]):.4g}  tok_id={int(tid)}"
        pieces.append(
            f"<span class='tok' style='background:{bg};color:{fg};{border}' "
            f"title='{title}'>{shown}</span>"
        )
    return "".join(pieces)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-expt-dir", type=Path, required=True,
                    help="Where to pull features from (e.g. "
                         "data/cherry_picked_feature_entropy/latest_neuronpedia_full)")
    ap.add_argument("--feature-ids", type=int, nargs="+", required=True)
    ap.add_argument("--ctx-len", type=int, default=128)
    ap.add_argument("--top-n-contexts", type=int, default=5)
    ap.add_argument("--max-visible-tokens", type=int, default=16,
                    help="Display-only truncation: only show the last N "
                         "tokens of each context, ending at the activating "
                         "position. The H_pos values in the histogram were "
                         "computed on the full ctx_len-token slice; this "
                         "flag changes only the visualisation strip.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="New per-experiment dir to write into.")
    args = ap.parse_args()

    src = args.source_expt_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((src / "manifest.json").read_text())
    desc_by_id = {int(e["feature_id"]): e.get("description", "")
                  for e in manifest.get("features", [])}
    tier_by_id = {int(e["feature_id"]): e["tier"]
                  for e in manifest.get("features", [])}
    layer = int(manifest["layer"])
    preset_name = manifest["preset"]
    sae_id = manifest["sae_id"]

    # CSVs (all tiers — we don't know which tier each requested fid is in)
    frames = []
    for path in sorted(src.glob(f"ctx{args.ctx_len:03d}_*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"no ctx{args.ctx_len:03d}_*.csv under {src}")
    df = pd.concat(frames, ignore_index=True)
    ok = df[df["status"] == "ok"].copy()

    # Influences NPZs (one per tier)
    influences: dict[str, np.ndarray] = {}
    for path in sorted(src.glob(f"influences_ctx{args.ctx_len:03d}_*.npz")):
        z = np.load(path)
        for k in z.files:
            influences[k] = z[k]

    # Top contexts JSON
    top_ctx = json.loads(
        (src / f"top_contexts_layer{layer}.json").read_text())

    corpus_tokens = torch.load(src / "corpus_tokens.pt").to(torch.long)

    # Per-feature recall_alt if available (for header info)
    quality_csv = src / "feature_quality_stats.csv"
    quality = (pd.read_csv(quality_csv).set_index("feature_id")
               if quality_csv.exists() else None)

    # Tokenizer
    preset = get_preset(preset_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(preset.model_id)

    # Build cards
    cards_html: list[str] = []
    cards_meta: list[dict] = []
    for fid in args.feature_ids:
        sub = ok[ok["feature_id"] == fid].copy()
        if sub.empty:
            print(f"[WARN] feature {fid} has no ok events in {src}",
                  flush=True)
            continue
        tier = tier_by_id.get(fid, str(sub["tier"].iloc[0]))
        desc = desc_by_id.get(fid, "")
        recall = (None if quality is None or fid not in quality.index
                  else quality.loc[fid].get("recall_alt"))
        recall_html = (f"&middot; recall_alt={float(recall):.2f}"
                       if recall is not None and not pd.isna(recall) else "")
        h_vals = sub["H_pos_bits"].values
        mean = float(h_vals.mean())
        std = float(h_vals.std()) if len(h_vals) > 1 else 0.0
        hist_b64 = per_feature_histogram(h_vals, args.ctx_len,
                                          TIER_COLORS[tier])

        # Top-N events by activation strength.
        top_events = (sub.sort_values("activation_value", ascending=False)
                      .head(args.top_n_contexts))

        ctx_html_parts = []
        for ev_idx, ev_row in enumerate(top_events.itertuples(), start=1):
            rank = int(ev_row.top_act_rank)
            key = f"{fid}_{rank}"
            J = influences.get(key)
            if J is None:
                ctx_html_parts.append(
                    f"<div class='ctx'><b>#{ev_idx}</b> "
                    f"<i>(no influence vector for {key})</i></div>")
                continue
            abs_peak = int(ev_row.abs_peak)
            ctx_start = abs_peak - args.ctx_len + 1
            if ctx_start < 0 or abs_peak >= corpus_tokens.numel():
                ctx_html_parts.append(
                    f"<div class='ctx'><b>#{ev_idx}</b> "
                    f"<i>(slice OOB)</i></div>")
                continue
            tok_ids_full = corpus_tokens[ctx_start:abs_peak + 1].tolist()
            # Display-only truncation to the last N tokens ending at peak.
            visible_n = max(1, min(args.max_visible_tokens,
                                   len(tok_ids_full)))
            tok_ids_visible = tok_ids_full[-visible_n:]
            J_visible = J[-visible_n:]
            peak_pos_in_slice = len(tok_ids_visible) - 1
            strip = render_token_strip(tok_ids_visible, J_visible,
                                       peak_pos_in_slice, tokenizer)
            # Colorbar legend reflects the visible window's J range so the
            # bar and the strip are consistent with each other.
            j_min = float(J_visible.min())
            j_max = float(J_visible.max())
            gradient_css = colorbar_css()
            legend_html = (
                f"<div class='cbar-row'>"
                f"<span class='cbar-label'>sqrt(J) scale: "
                f"J={j_min:.2g}</span>"
                f"<span class='cbar' style='background:{gradient_css}'></span>"
                f"<span class='cbar-label'>J={j_max:.3g}</span>"
                f"</div>"
            )
            ctx_html_parts.append(
                f"<div class='ctx'>"
                f"<div class='ctx-header'>#{ev_idx}  "
                f"&middot; act={ev_row.activation_value:.2f}  "
                f"&middot; H_pos={ev_row.H_pos_bits:.2f} bits  "
                f"&middot; corpus_idx={abs_peak}  "
                f"&middot; top_act_rank={rank}</div>"
                f"<div class='strip'>{strip}</div>"
                f"{legend_html}"
                f"</div>"
            )

        np_url = (f"https://www.neuronpedia.org/{preset_name}/{sae_id}/{fid}")
        desc_safe = (desc.replace("&", "&amp;").replace("<", "&lt;")
                       .replace(">", "&gt;"))
        cards_html.append(f"""<section class='card'>
  <h2>
    feature {fid}
    <span class='tier-badge tier-{tier}'>{tier}</span>
    <a class='np' href='{np_url}' target='_blank'>NP</a>
    <span class='hpos'>H_pos = {mean:.2f} &plusmn; {std:.2f} bits
      &middot; n={len(sub)} {recall_html}</span>
  </h2>
  <div class='desc'>{desc_safe}</div>
  <div class='hist'><img src='data:image/png;base64,{hist_b64}'/></div>
  <h3>Top {args.top_n_contexts} activating contexts (peak in red border, darker = higher gradient):</h3>
  {''.join(ctx_html_parts)}
</section>""")

        cards_meta.append({
            "feature_id": fid, "tier": tier, "description": desc,
            "mean_h_pos": mean, "std_h_pos": std, "n_events": int(len(sub)),
            "recall_alt": (float(recall) if recall is not None
                            and not pd.isna(recall) else None),
        })

    # Header CSS
    css = r"""
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1700px; margin: 1.0em auto; padding: 0 1em;
       color: #222; }
.meta { background: #f5f5f5; padding: 0.6em 1em; border-radius: 6px;
        font-size: 0.9em; margin-bottom: 1em; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1em 1.2em;
        margin: 1em 0; background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.card h2 { margin: 0 0 0.3em 0; font-size: 1.05em; }
.card h3 { margin: 0.6em 0 0.3em 0; font-size: 0.85em;
           color: #555; font-weight: 600; }
.tier-badge { display: inline-block; padding: 1px 8px; border-radius: 8px;
              color: white; font-weight: bold; font-size: 0.78em;
              margin-left: 0.4em; }
.tier-token    { background: #1f77b4; }
.tier-phrase   { background: #2ca02c; }
.tier-concept  { background: #ff7f0e; }
.tier-abstract { background: #d62728; }
.hpos { color: #555; font-weight: normal; font-size: 0.9em;
        margin-left: 0.6em; }
.desc { font-size: 0.9em; color: #444; margin-bottom: 0.4em; }
.hist img { display: block; }
.ctx { margin: 0.4em 0 0.6em 0; }
.ctx-header { font-size: 0.78em; color: #555; margin-bottom: 0.15em;
              font-variant-numeric: tabular-nums; }
.strip { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         font-size: 0.78em; line-height: 1.65em;
         word-break: break-word; }
.tok { display: inline-block; padding: 0px 1px; margin: 1px 0;
       border-radius: 3px; }
.cbar-row { margin-top: 0.25em; display: flex; align-items: center;
            gap: 6px; }
.cbar { display: inline-block; width: 220px; height: 9px;
        border: 1px solid #aaa; border-radius: 2px; }
.cbar-label { font-size: 0.7em; color: #666;
              font-variant-numeric: tabular-nums; }
a.np { color: #1f5fb4; text-decoration: none; font-size: 0.8em;
       margin-left: 0.4em; }
a.np:hover { text-decoration: underline; }
"""

    title = (f"Feature case studies @ CTX_LEN={args.ctx_len} "
             f"&mdash; {preset_name} layer {layer} ({sae_id})")
    html = f"""<!doctype html>
<html lang='en'><head>
<meta charset='utf-8'/><title>{title}</title>
<style>{css}</style>
</head><body>
<h1>{title}</h1>
<div class='meta'>
  <b>output dir:</b> {out_dir}<br/>
  <b>source experiment:</b> {src}<br/>
  <b>features:</b> {', '.join(str(f) for f in args.feature_ids)}<br/>
  <b>top contexts shown per feature:</b> {args.top_n_contexts}<br/>
  <b>visible tokens per strip:</b> last {args.max_visible_tokens} tokens
  ending at the activating position (display-only truncation; H_pos
  in the per-feature histogram was computed on the full
  {args.ctx_len}-token slice).<br/>
  Each token is coloured by the per-position squared gradient norm
  J(t') = ||&part;f/&part;x||&sup2;, using a per-context
  <b>sqrt(J) / sqrt(J_max)</b> normalisation in the <b>Greens</b>
  colormap (so colour is linear in the gradient norm itself; darker
  green = higher gradient). The activating token (last position of
  the slice) is outlined in red. Hover a token to see its raw J value.
  The colour-bar legend below each strip shows the J range mapped
  over. <code>corpus_idx</code> in each context header is the absolute
  token position of the activating token in the WikiText corpus.
</div>
{''.join(cards_html)}
</body></html>"""

    out_html = out_dir / "feature_case_studies.html"
    out_html.write_text(html)

    out_manifest = {
        "experiment": "feature_case_studies",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "preset": preset_name, "layer": layer, "sae_id": sae_id,
        "ctx_len": args.ctx_len, "top_n_contexts": args.top_n_contexts,
        "source_expt_dir": str(src),
        "features": cards_meta,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2))

    print(f"[INFO] wrote {out_html}")
    print(f"[INFO] wrote {out_dir / 'manifest.json'}")
    print(f"[INFO] features rendered: "
          f"{[m['feature_id'] for m in cards_meta]}")


if __name__ == "__main__":
    main()
