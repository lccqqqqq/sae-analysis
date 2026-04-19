# Feature-quality ranking trial — per-batch diagnostic plots

## Summarized plan

- **Input:** `data/entropy_comparison_resid_out_layer5_20260414_053350.pt` (50 batches, τ = 0.2).
- **New script:** `scripts/plotting/regenerate_batch_plots_quality.py` — clone of the production plotter.
- **Cross-batch quality stats** computed from `batch_results[*]["feature_activations"]`:
  - `μ_on(a)` = mean activation of feature *a* across batches where it fired.
  - `σ_on(a)` = std of those on-values.
  - `margin(a) = μ_on / τ`, `cv(a) = σ_on / μ_on`.
- **Right panel redesign:** x = margin (log), y = entropy, top-5 ranked by margin, vertical cutoff at margin = 2, point size ∝ 1 / (1 + cv).
- **Left panel:** unchanged mechanics; colored curves track the new top-5.
- **Trial scope:** first 10 batches, layer 5; output under `plots/entropy_plots_quality_resid_out_layer5_20260414_053350/`.
- No edits to any existing script.

## Motivation

The production plotter ranks the per-batch "top-5" features by their raw activation at the last token position. The hypothesis under test: at late layers, the highest-activation features are disproportionately **polysemantic** — they encode broad, composed, or near-universal signals, and their high activations reflect that generality rather than specificity. Their gradient-based influence entropy then also runs high (because they mix many upstream causes), inflating the mean and confounding the "entropy grows with depth" result.

Two complementary quality scores from the feature-filter discussion:

- **Margin** `μ_on / τ` — how far above the activation cutoff the feature typically sits when it fires. Low margin → threshold-grazer, gradient dominated by ReLU-kink noise.
- **Dispersion** `σ_on / μ_on` — variability of on-state magnitudes. High dispersion → trigger-mixing.

Both are cheap: every active feature's last-position activation is already stored in `entropy_comparison_{site}_{ts}.pt`.

## Task

Re-rank the right panel of per-batch diagnostic plots at layer 5 using cross-batch margin instead of per-batch raw activation, for the first 10 batches, and judge whether the new top-5 (a) differs from the old top-5 and (b) sits cleaner against the `H_token` baseline.

## Methodology

1. Load `entropy_comparison_*.pt`, pull `tau = config.threshold` (= 0.2).
2. Iterate `batch_results` once, accumulating `on_vals[f] = [activation_in_batch_i, ...]` for every feature that ever appears in `feature_activations`.
3. Compute `mu_on`, `sigma_on` (population std, `ddof=0`; NaN if `n < 2`), `margin`, `cv` per feature.
4. For each batch, pick the top-5 features by `margin` (among features active in that batch). Colour them via `tab10`.
5. Right panel: log-x scatter `(margin, entropy)` for every active feature in the batch. Top-5 highlighted; grey dot size ∝ `1 / (1 + cv)`; vertical dotted line at `margin = 2` (the "solidly on" cutoff); red dashed horizontal line at the batch's `token_vector_entropy`.
6. Left panel: same structure as original — grey bulk, black-dashed token vector, coloured top-5 curves — but "top-5" refers to the margin ranking.
7. Token-heatmap rows below the plots: kept, with an extra annotation per row showing `μ/τ`, `CV`, `n_active`, `act_last` for the chosen feature.

## Codebase instructions

New file: `scripts/plotting/regenerate_batch_plots_quality.py`. Untouched: `regenerate_batch_plots.py`, everything under `scripts/analysis/` and `scripts/figures/`.

Rerun:

```bash
PYTHONPATH=scripts/plotting \
  python scripts/plotting/regenerate_batch_plots_quality.py \
    --pt-file data/entropy_comparison_resid_out_layer5_20260414_053350.pt \
    --num-batches 10
```

CLI options:

- `--pt-file PATH` (required) — an `entropy_comparison_*.pt`.
- `--num-batches N` (default 10) — render first N batches.
- `--rank-by {margin,cv}` (default `margin`).
- `--no-size-by-cv` — flat-sized points instead of cv-scaled.
- `--out-dir PATH` — default `plots/entropy_plots_quality_<site>_<ts>/`.

`PYTHONPATH=scripts/plotting` is needed because the new script imports helpers (`normalize_influence`, `draw_token_heatmap_rows`) from `regenerate_batch_plots.py`.

Outputs: `plots/entropy_plots_quality_resid_out_layer5_20260414_053350/batch_{000..009}.png`.

## Results

### Global quality distribution at layer 5

| Statistic | Value |
|---|---|
| Unique features ever active (50 batches) | 8,891 |
| `margin` min / median / max | 1.00 / 2.00 / 58.81 |
| Fraction with `margin < 2` (grazers) | **50.1%** |

Half of the ever-active features are grazers by the `μ/τ ≥ 2` rule. They contribute to the grey bulk of the right-panel scatter but are de-emphasised by size.

### Top-margin features are *already* the top-activation features

The top-5 by margin is **identical to the top-5 by raw activation** in all 10 trial batches:

`F23666, F12173, F16160, F7541, F7556`.

Their diagnostics across the 50 batches:

| Feature | `μ/τ` | `cv` | `n_active / 50` |
|---|---|---|---|
| F23666 | 58.8 | 0.16 | **50/50 (100%)** |
| F12173 | 53.1 | 0.16 | **50/50 (100%)** |
| F16160 | 46.2 | 0.15 | **50/50 (100%)** |
| F7541  | 40.4 | 0.16 | **50/50 (100%)** |
| F7556  | 39.7 | 0.16 | **50/50 (100%)** |
| F7880  | 24.1 | 0.22 | 50/50 (100%) |
| F22220 | 23.9 | 0.24 | 50/50 (100%) |
| F15249 | 23.9 | 0.23 | 50/50 (100%) |

Every one of the top-by-margin features at layer 5 fires in **every** batch. That is the polysemanticity signature: features that activate indiscriminately, regardless of the 64-token context.

### Per-batch entropy vs token-vector baseline

Count of top-5 features whose entropy sits **below** `H_token` in each of the 10 trial batches:

| Batch | `H_token` | top-5 `H` range | below / 5 |
|---|---|---|---|
| 0 | 4.07 | 3.89 – 4.02 | 5 |
| 1 | 4.26 | 4.09 – 4.61 | 1 |
| 2 | 3.56 | 3.17 – 3.39 | 5 |
| 3 | 3.85 | 3.52 – 3.71 | 5 |
| 4 | 3.28 | 3.41 – 3.48 | 0 |
| 5 | 4.72 | 4.82 – 4.88 | 0 |
| 6 | 3.90 | 4.06 – 4.24 | 0 |
| 7 | 4.25 | 4.21 – 4.28 | 3 |
| 8 | 3.24 | 3.23 – 3.26 | 2 |
| 9 | 3.42 | 2.62 – 2.66 | 5 |

Mean: 2.6 / 5 features sit below baseline. The top-margin features straddle `H_token` batch-by-batch — consistent with "low-entropy decomposition" being a **population-mean** claim, not a per-batch one, and specifically not a property of the features the production plotter draws attention to.

### Verdict on the trial

- **Margin ranking alone is not sufficient to isolate "good" features at layer 5.** The polysemantic suspects are *solidly on*, not grazers; a margin lower bound doesn't demote them.
- **The frequency *upper bound* is the missing filter.** The five dominant features have `n_active / 50 = 100%` — feature frequency ≈ 1.0, far above the `< 1e-2` criterion from the filter discussion. The cheapest way to demote them is filter 1 from that stack, not filter 3.
- **The trial is still useful downstream:** the grazer fraction (50%) and the cv/margin diagnostics are now readily available, and the quality-annotated right panel makes the polysemanticity pattern visually obvious.

### Representative PNGs

- Old: `plots/entropy_plots_resid_out_layer5_20260414_053350/batch_000.png`
  New: `plots/entropy_plots_quality_resid_out_layer5_20260414_053350/batch_000.png`
- Old: `plots/entropy_plots_resid_out_layer5_20260414_053350/batch_005.png`
  New: `plots/entropy_plots_quality_resid_out_layer5_20260414_053350/batch_005.png`
- Old: `plots/entropy_plots_resid_out_layer5_20260414_053350/batch_009.png`
  New: `plots/entropy_plots_quality_resid_out_layer5_20260414_053350/batch_009.png`

Compare the right panels: the new version shows the margin distribution on a log axis with the `margin = 2` grazer cutoff marked, and annotates each coloured point with `(μ/τ, CV, n_active)` in the legend, making the "fires in all 50 batches" property visible at a glance.

## Decisions / next steps

Resolved by the trial:

- **Margin re-ranking does not alter the visible story at layer 5.** Do not promote this to the canonical plotter as-is; keep it as a diagnostic variant.
- **Grazer population is real and substantial (50%).** Worth filtering *out* at the ranking stage, but the top-5 features at layer 5 are not in it.

Still open:

- **Implement a frequency *upper bound* filter** (`freq < 1e-2`) at `scripts/analysis/feature_token_influence.py:304–322` — this is what would actually exclude the polysemantic-always-on features. Requires `feature_sparsity_data_{site}.pt`, which currently isn't on disk at layer 5 (only at layer 3 via the tracked pipeline); regenerate via `scripts/analysis/feature_sparsity.py`.
- **Repeat on layers 0–4** to see how the grazer fraction and "always-on" count change with depth. Expectation: fewer always-on features at shallow layers.
- **Neuronpedia autointerp labels** for the five suspect features (F23666, F12173, F16160, F7541, F7556 at layer 5 / set `5-res-sm`) — if they come back with vague or empty descriptions, that's qualitative confirmation of polysemanticity.
- **Recompute the headline entropy-vs-depth curve** after excluding freq > 1e-2 features, and see whether the L0 → L5 gap shrinks. This is the actual test of the confound hypothesis.
