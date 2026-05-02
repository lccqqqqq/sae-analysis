"""Per-feature HTML report for a cherry-picked-entropy experiment.

Produces a single self-contained HTML file with one row per feature,
showing — at a glance — the per-feature distribution of H_pos values
across that feature's top-K activating events, alongside the
interpretation and the actual top-activating contexts that the
gradient was computed on.

Each row contains:

    rank | feature_id | tier | n_ok | mean ± std H_pos
         | inline H_pos histogram across the feature's events
         | description (from manifest)
         | top-3 activating contexts (peak token bolded)
         | link to the Neuronpedia dashboard for the feature

Rows are sorted ascending by mean H_pos so scrolling top-to-bottom
walks the localized → spread continuum directly.

A header section shows the experiment metadata and a population-wide
histogram of per-feature mean H_pos values.

Output: ``<expt-dir>/figures/feature_table_ctx<NNN>.html``
"""

from __future__ import annotations

import argparse
import base64
import io
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


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def per_feature_histogram(values: np.ndarray, ctx_len: int,
                           color: str) -> str:
    """Tiny histogram for one feature's H_pos values."""
    fig, ax = plt.subplots(1, 1, figsize=(2.4, 0.8))
    bins = np.linspace(0.0, float(np.log2(ctx_len)), 21)
    ax.hist(values, bins=bins, color=color, edgecolor="black",
            linewidth=0.4, alpha=0.85)
    ax.set_xlim(0, np.log2(ctx_len))
    ax.set_xticks([0, np.log2(ctx_len) / 2, np.log2(ctx_len)])
    ax.set_xticklabels([0, int(np.log2(ctx_len) / 2),
                        int(np.log2(ctx_len))], fontsize=6)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1)
    return fig_to_b64(fig)


def population_histogram(per_feat_means: pd.Series, ctx_len: int) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
    bins = np.linspace(0.0, float(np.log2(ctx_len)), 35)
    ax.hist(per_feat_means.values, bins=bins, color="#666666",
            edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.set_xlim(0, np.log2(ctx_len))
    ax.set_xlabel(f"mean H_pos per feature (bits)  —  range "
                  f"{per_feat_means.min():.2f} … {per_feat_means.max():.2f}",
                  fontsize=10)
    ax.set_ylabel("# features", fontsize=10)
    ax.set_title(f"Population: mean H_pos across {len(per_feat_means)} features "
                 f"at CTX_LEN={ctx_len}", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    return fig_to_b64(fig)


def render_context_html(tokens: list[str], peak_idx: int,
                         max_tokens_each_side: int = 8) -> str:
    """Render a single context with the peak token bolded and surrounding
    tokens visible (lightly truncated)."""
    if not tokens or not (0 <= peak_idx < len(tokens)):
        return ""
    lo = max(0, peak_idx - max_tokens_each_side)
    hi = min(len(tokens), peak_idx + max_tokens_each_side + 1)
    pieces: list[str] = []
    if lo > 0:
        pieces.append("<span class='ellipsis'>…</span>")
    for i in range(lo, hi):
        t = (tokens[i].replace("▁", "·").replace("Ġ", "·")
             .replace("\n", "↵"))
        # Escape angle brackets for HTML.
        t_html = (t.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;"))
        if i == peak_idx:
            pieces.append(f"<b class='peak'>[{t_html}]</b>")
        else:
            pieces.append(f"<span class='tok'>{t_html}</span>")
    if hi < len(tokens):
        pieces.append("<span class='ellipsis'>…</span>")
    return " ".join(pieces)


CSS = r"""
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1700px; margin: 1.0em auto; padding: 0 1em;
       color: #222; }
h1 { font-size: 1.4em; }
h2 { font-size: 1.05em; color: #555; }
.meta { background: #f5f5f5; padding: 0.7em 1em; border-radius: 6px;
        font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; font-size: 0.84em;
        margin-top: 1em; }
th, td { border-bottom: 1px solid #e0e0e0; padding: 6px 8px;
         vertical-align: top; }
th { background: #fafafa; text-align: left; position: sticky; top: 0;
     z-index: 2; }
.tier-badge { display: inline-block; padding: 1px 7px; border-radius: 8px;
              color: white; font-weight: bold; font-size: 0.85em; }
.tier-token    { background: #1f77b4; }
.tier-phrase   { background: #2ca02c; }
.tier-concept  { background: #ff7f0e; }
.tier-abstract { background: #d62728; }
td.center { text-align: center; }
td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.tok      { color: #444; }
.peak     { color: #b22222; }
.ellipsis { color: #999; }
.context-cell { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-size: 0.78em; line-height: 1.55em;
                max-width: 720px; }
.context-cell > div { padding: 1px 0;
                      border-bottom: 1px dotted #eee; }
.context-cell > div:last-child { border-bottom: none; }
.histo img { display: block; }
.desc { max-width: 280px; }
a.np { color: #1f5fb4; text-decoration: none; }
a.np:hover { text-decoration: underline; }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expt-dir", type=Path, required=True)
    ap.add_argument("--ctx-len", type=int, default=128)
    ap.add_argument("--top-contexts-shown", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--min-recall", type=float, default=None,
                    help="Drop features whose Neuronpedia recall_alt is "
                         "below this. Pulled from "
                         "<expt>/feature_quality_stats.csv if present.")
    ap.add_argument("--max-peak-diversity", type=float, default=None,
                    help="Drop features whose fraction of unique peak "
                         "tokens across top events exceeds this.")
    ap.add_argument("--exclude-bos-dominated", action="store_true",
                    help="Drop features whose >50%% top peaks are <bos>.")
    args = ap.parse_args()
    expt_dir = args.expt_dir.resolve()
    ctx_len = int(args.ctx_len)

    manifest = json.loads((expt_dir / "manifest.json").read_text())
    desc_by_id = {int(e["feature_id"]): e.get("description", "")
                  for e in manifest.get("features", [])}
    tier_by_id = {int(e["feature_id"]): e["tier"]
                  for e in manifest.get("features", [])}

    # CSVs
    frames = []
    for tier in TIER_ORDER:
        path = expt_dir / f"ctx{ctx_len:03d}_{tier}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no ctx{ctx_len:03d}_*.csv under {expt_dir}")
    df = pd.concat(frames, ignore_index=True)
    ok = df[df["status"] == "ok"].copy()

    # Top contexts JSON
    layer = manifest.get("layer", 12)
    top_ctx_path = expt_dir / f"top_contexts_layer{layer}.json"
    top_ctx = (json.loads(top_ctx_path.read_text())
               if top_ctx_path.exists() else {})

    # Optional quality-stats join for filtering.
    quality_csv = expt_dir / "feature_quality_stats.csv"
    quality = (pd.read_csv(quality_csv).set_index("feature_id")
               if quality_csv.exists() else None)
    n_filtered = 0
    filter_reasons: dict[int, str] = {}

    rows: list[dict] = []
    for fid, sub in ok.groupby("feature_id"):
        if quality is not None and int(fid) in quality.index:
            q = quality.loc[int(fid)]
            if (args.min_recall is not None
                    and (pd.isna(q.get("recall_alt"))
                         or float(q["recall_alt"]) < args.min_recall)):
                n_filtered += 1
                filter_reasons[int(fid)] = "recall_alt"
                continue
            if (args.max_peak_diversity is not None
                    and float(q["frac_unique"]) > args.max_peak_diversity):
                n_filtered += 1
                filter_reasons[int(fid)] = "peak_diversity"
                continue
            if args.exclude_bos_dominated and float(q["frac_bos"]) > 0.5:
                n_filtered += 1
                filter_reasons[int(fid)] = "bos_dominated"
                continue
        tier = tier_by_id.get(int(fid), str(sub["tier"].iloc[0]))
        vals = sub["H_pos_bits"].values
        rows.append({
            "feature_id": int(fid),
            "tier": tier,
            "n_ok": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
            "max_act": float(sub["activation_value"].max()),
            "values": vals,
            "description": desc_by_id.get(int(fid), ""),
            "examples": (top_ctx.get(str(fid)) or {}).get("examples", []),
        })
    if args.min_recall is None and args.max_peak_diversity is None \
            and not args.exclude_bos_dominated:
        # No filters set; the loop above didn't filter. Reset for clarity.
        n_filtered = 0
    rows.sort(key=lambda r: r["mean"])

    # Population histogram for header
    per_feat_means = pd.Series([r["mean"] for r in rows])
    pop_hist_b64 = population_histogram(per_feat_means, ctx_len)

    # Build HTML
    sae_id = manifest.get("sae_id", "")
    preset = manifest.get("preset", "")
    np_url = (f"https://www.neuronpedia.org/{preset}/{sae_id}/" +
              "{fid}") if (preset and sae_id) else None

    html_rows: list[str] = []
    for rank, r in enumerate(rows, start=1):
        hist_b64 = per_feature_histogram(
            r["values"], ctx_len, TIER_COLORS[r["tier"]])
        examples = r["examples"][:args.top_contexts_shown]
        ctx_html = "".join(
            f"<div>{render_context_html(ex['tokens'], int(ex['token_idx']))}"
            f" <span style='color:#888'>act={float(ex['activation']):.1f}</span></div>"
            for ex in examples
        ) or "<i>no contexts</i>"
        np_link = (f"<a class='np' href='{np_url.format(fid=r['feature_id'])}' "
                   "target='_blank'>NP</a>") if np_url else ""
        desc = r['description'].replace("<", "&lt;").replace(">", "&gt;")
        html_rows.append(f"""<tr>
  <td class='numeric'>{rank}</td>
  <td class='numeric'>{r['feature_id']}</td>
  <td><span class='tier-badge tier-{r['tier']}'>{r['tier']}</span></td>
  <td class='numeric'>{r['n_ok']}</td>
  <td class='numeric'>{r['mean']:.2f} ± {r['std']:.2f}</td>
  <td class='histo'><img src='data:image/png;base64,{hist_b64}'/></td>
  <td class='desc'>{desc}</td>
  <td class='context-cell'>{ctx_html}</td>
  <td class='center'>{np_link}</td>
</tr>""")

    n_total_features = len(rows)
    n_total_events = sum(r["n_ok"] for r in rows)
    title = (f"Per-feature H_pos report  —  {preset} layer {layer}  "
             f"({sae_id}, ctx_len={ctx_len}, top_k={manifest.get('top_k')}, "
             f"variant={manifest.get('variant', '?')})")

    html = f"""<!doctype html>
<html lang='en'><head>
<meta charset='utf-8'/>
<title>{title}</title>
<style>{CSS}</style>
</head><body>
<h1>{title}</h1>
<div class='meta'>
  <b>experiment dir:</b> {expt_dir}<br/>
  <b>features shown:</b> {n_total_features}  &middot;
  <b>filtered out:</b> {n_filtered}  &middot;
  <b>ok events:</b> {n_total_events}  &middot;
  <b>corpus:</b> {manifest.get('corpus', 'unknown')}<br/>
  <b>filters:</b> min_recall={args.min_recall}  max_peak_diversity={args.max_peak_diversity}  exclude_bos={args.exclude_bos_dominated}<br/>
  <b>sorted ascending by mean H_pos</b> — scroll top-to-bottom
  to walk the localized→spread continuum.
</div>

<h2>Population: mean H_pos across all {n_total_features} features</h2>
<img src='data:image/png;base64,{pop_hist_b64}'/>

<table>
  <thead><tr>
    <th>#</th>
    <th>feature_id</th>
    <th>tier</th>
    <th>n</th>
    <th>H_pos (bits)</th>
    <th>distribution</th>
    <th>interpretation</th>
    <th>top {args.top_contexts_shown} activating contexts (peak bolded)</th>
    <th>NP</th>
  </tr></thead>
  <tbody>
{chr(10).join(html_rows)}
  </tbody>
</table>
</body></html>"""

    suffix = ""
    if args.min_recall is not None:
        suffix += f"_recall{int(args.min_recall*100):02d}"
    if args.max_peak_diversity is not None:
        suffix += f"_div{int(args.max_peak_diversity*100):02d}"
    if args.exclude_bos_dominated:
        suffix += "_nobos"
    output = args.output or (expt_dir / "figures"
                             / f"feature_table_ctx{ctx_len:03d}{suffix}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    size_mb = output.stat().st_size / 1e6
    print(f"[INFO] wrote {output}  ({size_mb:.1f} MB, "
          f"{n_total_features} features)")


if __name__ == "__main__":
    main()
