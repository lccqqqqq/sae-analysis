# Cleanup Changelog

Tracks deletions and deprecations from the trusted analysis tree under `scripts/`. Companion to `CHANGELOG.md` (which logs additions/modifications). Most recent first. Each entry preserves the original file's docstring and gives a reason code for the move.

Reason codes:
- **superseded** — replaced by a newer file that does the same job better.
- **unused** — no remaining caller, dead code.
- **incompatible** — breaks against the post-migration codebase and is not worth fixing in place.

---

## 2026-04-29 — Rename `BATCH_SIZE` → `CONTEXT_LEN` (and file/CLI/key equivalents)

**Reason:** misnomer fix (no deletion). Across the analysis tree, the constant
named `BATCH_SIZE` was always used as `chunk = tokens[i : i + BATCH_SIZE]`
fed to the model with batch-dim 1 — it is the **sequence/context length**,
not the batch dimension. Renaming everywhere it shows up to make the
semantics match the name.

### Changes

- Module-level constants:
  `BATCH_SIZE` → `CONTEXT_LEN` in `feature_sparsity.py`,
  `feature_location_analysis.py`, `feature_token_influence.py`,
  `compare_entropies.py`, `compare_entropies_multi_layer.py`,
  `token_vector_influence.py`. Local `BATCH_SIZE` in
  `plot_entropy_vs_activation_layer5.py` similarly renamed.
- Files:
  `scripts/analysis/entropy_vs_batch_size.py` →
  `scripts/analysis/entropy_vs_context_len.py`;
  `scripts/plot/plot_entropy_vs_batch_size_notebook.py` →
  `scripts/plot/plot_entropy_vs_context_len_notebook.py`.
- Constants/params/CLI inside `entropy_vs_context_len.py`:
  `MAX_BATCH_SIZE` → `MAX_CONTEXT_LEN`,
  `MIN_BATCH_SIZE` → `MIN_CONTEXT_LEN`,
  `BATCH_SIZE_STEP` → `CONTEXT_LEN_STEP`;
  function params `max_batch_size` / `min_batch_size` →
  `max_context_len` / `min_context_len`;
  CLI flags `--max-batch-size` / `--min-batch-size` →
  `--max-context-len` / `--min-context-len`.
- Saved-dict keys (in `.pt` outputs):
  `"batch_size"` → `"context_len"`;
  `"max_batch_size"` / `"min_batch_size"` → `"max_context_len"` /
  `"min_context_len"`;
  `"sub_batch_sizes"` → `"sub_context_lens"`;
  `"results_by_batch_size"` → `"results_by_context_len"`.
- Output filenames:
  `entropy_vs_batch_size_<site>_<timestamp>.pt|.png` →
  `entropy_vs_context_len_<site>_<timestamp>.pt|.png`,
  and the matching plot directory.
- Plot labels: `"Batch Size"` → `"Context length"` axis label and title.
- Docstrings, log messages, and README/`scripts/plot-logic-note.md`
  references updated correspondingly.

### Deliberately NOT changed

- `# We only have batch size 1 here` comments in `feature_sparsity.py`
  and `feature_location_analysis.py` — these refer to the genuine batch
  dimension of the unsqueezed tensor and are accurate.
- Variable / function names that contain bare `batch` (not `batch_size`):
  `large_batch`, `sub_batch`, `compute_entropy_for_sub_batch`,
  `process_batch_with_influence`. Out of scope for this rename.
- `--num-batches` CLI flag in `compare_entropies*.py` and the
  `submit_entropy_bench.sh` / `submit_all_presets.sh` scripts — that flag
  is the **count** of windows, not the size, so it is not a misnomer.
- `CHANGELOG.md` historical entries — not rewritten (history record).
- `deprecated/`, `scripts-v1/`, and `notebooks/*.ipynb` — out of the
  trusted-tree migration scope.

### Backward-compatibility break

Per the project's "no fallbacks on refactors" rule, the renamed loaders
read only `"context_len"` / `"results_by_context_len"`. Any pre-existing
`.pt` files saved with the old `"batch_size"` keys are no longer
readable — regenerate them.

### Safety snapshot

- Backup branch: `backup/pre-context-len-rename-20260429` at
  `63ed863` (pre-rename HEAD).
- Tarball:
  `/mnt/users/clin/backups/sae-analysis-pre-context-len-rename-20260429.tgz`
  (399 MB, excludes `.venv`, `__pycache__`, `notebooks/*.ipynb`,
  `data/`, `out/`).

---

## 2026-04-27 — Deprecate `scripts/plot/plot_entropy_vs_depth.py`

**Moved to:** `deprecated/plot_entropy_vs_depth.py`
**Reason:** superseded
**Replacement:** `scripts/plot/plot_entropy_vs_depth_violin.py` (ported from `scripts-v1/figures/fig05_entropy_vs_depth_preset.py` and renamed)

### Original docstring

```
Jupyter notebook code block to plot entropy of leading features vs depth (layer 0-5).

This code:
1. Loads entropy comparison data from all 6 layers
2. Identifies leading features (top features by activation)
3. Plots entropy vs depth with consistent colors per feature
```

### Why superseded

The deprecated script had several limitations that the violin replacement fixes:

1. **Pythia-70m only.** Glob pattern `entropy_comparison_resid_out_layer{layer}_*.pt` is hard-coded for Pythia's `resid_out_layerN` site naming. Other presets use `resid_post_layerN` (gpt2-small, qwen2-0.5b) or `resid_layerN` (gemma-2-2b, llama-3.2-1b, llama-3-8b) and would silently fail to find any files.
2. **Six-layer hard-code.** `LAYERS = [0, 1, 2, 3, 4, 5]` — would `FileNotFoundError` for any preset with a different layer count (gpt2-small has 12, gemma-2-2b has 26, etc.).
3. **Required underscore-suffix filename.** Glob `_*.pt` only matched the old single-layer-driver naming `entropy_comparison_<site>_<timestamp>.pt`. The new multi-layer driver writes `entropy_comparison_<site>.pt` (timestamp in parent dir), so the deprecated script could not consume any post-migration data without compatibility symlinks.
4. **Top-10 leading lines only — no view of the population.** Drew exactly 10 colour-coded lines and nothing else; the bulk distribution of feature entropies at each layer was invisible.
5. **Hybrid scoring formula** for "leading feature" mixes total activation magnitude and layer-presence count in a way that's hard to assign a clean meaning to (`feature_activation_sums / feature_appearance_count` where `feature_activation_sums` aggregates over (batch, layer) but `feature_appearance_count` counts only layers).
6. **No token-vector entropy reference.** The `H_token > <H_feat>` conjecture stated in `compare_entropies.py`'s docstring couldn't be eyeballed from the figure.
7. **Misleading line interpolation.** Leading features fired in fewer than all layers got partial lines that visually interpolated across missing layers without any break or marker.

### What the violin replacement does instead

`plot_entropy_vs_depth_violin.py`:

- Auto-discovers run directory under `data/<preset>/<timestamp>/` and parses the layer set from `entropy_comparison_*.pt` filenames via regex (works for any site naming and any layer count).
- Plots one **violin** per layer over the per-feature mean entropies — shows the *population*, not just the top-N.
- Overlays a **blue dashed line** for the mean entropy of the top-20 most-frequently-active features (count-based ranking, tie-broken by mean entropy).
- Overlays a **red dashed line** for the per-layer token-vector entropy.
- Marks the `log₂(seq_len)` reference line.
- Auto-iterates over all presets with `--all`, writing one PNG per preset to `figures/entropy_vs_depth__<preset>.png`.

### Callers

None. `plot_entropy_vs_depth.py` was an end-of-pipeline figure script with no upstream importers. `notebook_entropy_vs_depth.py` is a near-duplicate (~95 % same code) that remains in `scripts/plot/` for ad-hoc Jupyter use; it has the same limitations and would similarly be obsolete in a strict-deprecation review, but it has not been moved as part of this change.
