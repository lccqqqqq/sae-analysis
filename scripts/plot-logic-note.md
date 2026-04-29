# How `entropy_vs_depth`-style figures are produced from `scripts/`

This file traces the entropy-vs-depth figure family back to the minimal set of
scripts in `scripts/`. Unlike `scripts-v1/`, there is no preset registry, no
model/SAE adapter layer, and no `figures/` directory — everything is
Pythia-70m-only and hard-coded.

The pipeline has two stages:

1. **Stage 1 (data)** — `compare_entropies_multi_layer.py` produces one
   `entropy_comparison_resid_out_layer{N}_<timestamp>.pt` per layer, plus a
   directory of per-batch diagnostic PNGs.
2. **Stage 2 (figure)** — `plot/plot_entropy_vs_depth.py` (driven through
   `plot/run_plot.py`) consumes those `.pt` files and draws the depth plot.

---

## 1) Stage 1: producing the per-layer `.pt` data files

### Minimal file set (data-producing)

```
scripts/analysis/compare_entropies_multi_layer.py   (entry point — runs all layers in one pass)
scripts/analysis/compare_entropies.py               (compare_entropies_for_batch + plot_batch_comparison
                                                     + compute_feature_entropy + normalize_influence)
scripts/analysis/feature_token_influence.py         (process_batch_with_influence + load_sae
                                                     + get_sae_weights + get_feature_activations)
scripts/analysis/token_vector_influence.py          (process_batch_with_token_influence)
scripts/analysis/data_loader.py                     (load_wikitext_train_text — HF wikitext-2-raw-v1)
```

Note the absences vs. `scripts-v1/`:

- **No `model_adapters.py`.** `compare_entropies_multi_layer.py` calls
  `AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")`
  directly and accesses `model.gpt_neox.layers[layer_idx]` directly. Other
  architectures will not work.
- **No `sae_adapters.py`.** SAE loading goes through `load_sae` +
  `get_sae_weights` (defined in `feature_token_influence.py`), which
  `torch.load`s `ae.pt` and pulls out `decoder.weight` / `encoder.weight` /
  `encoder.bias` by name-search heuristics. There is no `SAEBundle` class.
- **No `presets.py`.** Model id, layer list `[0..5]`, threshold (`0.2`), and
  context length (`64`) are module-level constants in
  `compare_entropies_multi_layer.py`. No `--preset` flag.

### Logic (in execution order)

`compare_entropies_multi_layer.main(layers, num_batches, random_batches, random_seed)`:

1. Generate a timestamp `unique_id`. Hard-code
   `MODEL_NAME = "EleutherAI/pythia-70m-deduped"`. Pick device CUDA → MPS → CPU.
2. Load model + tokenizer once.
3. Load all SAEs once via `load_all_sae_weights(layers, d_model)`, which scans
   `dictionaries/pythia-70m-deduped/resid_out_layer{N}/<run>/ae.pt` for each
   requested layer.
4. Load the corpus once with `load_wikitext_train_text()` (HF
   `wikitext-2-raw-v1` train split — see `CHANGELOG.md` at repo root for the
   contamination context behind this swap).
5. Tokenize the whole corpus in one `tokenizer(text, return_tensors="pt")` call.
6. Pick `num_batches` start positions, each yielding a window of `CONTEXT_LEN = 64`. If
   `--random-batches` (default), `random.sample` over stride-64 starts; else
   evenly spaced.
7. For each batch, for each layer, call `compare_entropies_for_batch` (defined
   in `compare_entropies.py:303`).

`compare_entropies_for_batch` (one batch, one layer):

- Calls `process_batch_with_influence` (`feature_token_influence.py`) — returns
  `{feat_idx: J_a array of shape [seq_len]}` for every feature whose
  last-position activation exceeds `THRESHOLD = 0.2`. Internally swaps the
  embedding for a `DummyEmbed` exposing `input_embeds` as a leaf with
  `requires_grad=True`, hooks the chosen layer's residual output, and
  `feat.backward()` on each active feature to get
  `sum_mu (dx_grad[:,t',mu])^2`.
- Re-runs the model under `torch.no_grad()` and re-hooks the layer to record
  last-position activations (used for colouring / x-axis of the right panel of
  the per-batch plot).
- Calls `process_batch_with_token_influence` (`token_vector_influence.py`) —
  uses `torch.autograd.functional.jacobian` to compute the full Jacobian
  `dy_nu(t=L,n) / dx_mu(t')`, returning `R(t')` and the Shannon entropy
  `-sum P log2 P` of `R / sum R`.
- Per-feature entropy via `compute_feature_entropy` →
  `scipy.stats.entropy(P, base=2)` with `eps=1e-12`.
- Returns `feature_entropies`, `token_vector_entropy`, `num_active_features`,
  `feature_influences`, `feature_activations`, `token_vector_influence`.

### Outputs written by `compare_entropies_multi_layer`

- `entropy_plots_resid_out_layer{N}_<timestamp>/batch_NNN.png` — per-batch
  two-panel diagnostic, drawn by `plot_batch_comparison`. Left panel:
  per-token influence probability traces (token vector dashed black, top-5
  features coloured, others light grey). Right panel: scatter of (last-position
  activation, entropy) per active feature, with token-vector entropy as a red
  dashed line.
- `entropy_plots_resid_out_layer{N}_<timestamp>/batch_index.json` — manifest
  for the per-batch PNGs.
- **`entropy_comparison_resid_out_layer{N}_<timestamp>.pt`** — the file the
  depth plot reads. Top-level keys:
  - `batch_results` — list of dicts, one per batch. Each entry has
    `batch_idx`, `start_idx`, `feature_entropies` (dict feat→entropy),
    `token_vector_entropy`, `num_active_features`,
    `feature_influences` (dict feat→[seq_len] array),
    `feature_activations` (dict feat→last-position activation),
    `token_vector_influence` ([seq_len] array).
  - `summary` — mean/std stats across batches.
  - `config` — threshold, context_len, total_features, seed.
  - `plots_dir` — string path to the per-batch PNG directory.
  - `batch_start_indices` — list of corpus offsets sampled.

Output paths are bare (relative to cwd). Files land in whatever directory you
run from. There is no `data/<preset>/<timestamp>/` run dir like in
`scripts-v1/`.

> **Side note — per-batch field reference.** Each entry of `batch_results`
> contains:
>
> | key | type | shape / range | source |
> | --- | --- | --- | --- |
> | `batch_idx` | `int` | `0 .. num_batches-1` | loop counter |
> | `start_idx` | `int` | offset into the tokenised corpus | from `batch_start_indices` |
> | `num_active_features` | `int` | count of features with last-position activation > `THRESHOLD = 0.2` | `len(feature_entropies)` |
> | `feature_entropies` | `dict[int, float]` | `feat_idx → entropy` in bits | `scipy.stats.entropy(J_a / sum J_a, base=2)` per feature |
> | `feature_activations` | `dict[int, float]` | `feat_idx → last-position SAE activation` | second forward pass under `no_grad`, `feats[0, -1, feat_idx]` |
> | `feature_influences` | `dict[int, np.ndarray]` | values of shape `[seq_len] = [64]`, dtype `float32`, un-normalized | `J_a(t') = sum_mu (∂feat/∂x_emb[:,t',mu])^2` for each active feature |
> | `token_vector_entropy` | `float` | scalar in bits | entropy of `R / sum R` |
> | `token_vector_influence` | `np.ndarray` | `[seq_len] = [64]`, dtype `float32`, un-normalized | `R(t') = sum_{ν,μ} (∂y_ν(L) / ∂x_μ(t'))^2` via `autograd.functional.jacobian` |

Run command:

```
python scripts/analysis/compare_entropies_multi_layer.py \
    --layers 0 1 2 3 4 5 --num-batches 10
```

---

## 2) Stage 2: the entropy-vs-depth figure

### Minimal file set (plotting)

```
scripts/plot/plot_entropy_vs_depth.py        (the plotting logic, written notebook-cell-style)
scripts/plot/notebook_entropy_vs_depth.py    (near-duplicate of the above, intended for paste-into-notebook)
scripts/plot/run_plot.py                     (headless runner — chdirs into DATA_DIR,
                                              monkey-patches plt.show → savefig)
```

`plot_entropy_vs_depth.py` and `notebook_entropy_vs_depth.py` are ~95% the
same code. The "notebook" one omits the docstring and uses
`defaultdict(lambda: defaultdict(list))` instead of a manual two-step
`setdefault`-style accumulation; the resulting figure is identical.

### Inputs the depth plot expects

The plot does a flat `glob.glob("entropy_comparison_resid_out_layer{N}_*.pt")`
for `N in [0..5]` **in the current working directory**. There is no preset
filtering, no run-directory selection, no recursive search.

For each layer it picks the most recent matching file by mtime
(`max(files, key=lambda f: Path(f).stat().st_mtime)`).

### Aggregation logic (`plot_entropy_vs_depth.py`)

1. Load 6 `.pt` files (one per layer 0..5).
2. **Per-layer per-feature aggregation across batches.** For each
   `batch_result` in a layer's `batch_results` list:
   - Append `feature_entropies[feat_idx]` to a running list for
     `(layer, feat_idx)`.
   - Append `feature_activations[feat_idx]` to a running list, and accumulate
     into a global `feature_activation_sums[feat_idx]` (summed across all
     batches AND all layers).
3. **Mean over batches.**
   `feature_entropy_avg[layer][feat_idx] = np.mean(...)` — one entropy value
   per feature per layer.
4. **"Leading feature" selection.** A feature qualifies iff it appears in
   **≥ 2 layers** (`feature_appearance_count[feat_idx] >= 2`). Score =
   `feature_activation_sums[feat_idx] / feature_appearance_count[feat_idx]`
   (mean total-activation per layer it appeared in). Sort descending, take
   top **`NUM_LEADING_FEATURES = 10`**.
5. **Plot.** One line per leading feature, x = layer index, y = mean entropy
   at that layer, marker = `'o-'`. Colours come from `tab10` (or `tab20` if
   more than 10). Legend shows `Feature {idx}`. No violin, no token-vector
   reference line, no `log2(64)` reference line.

### How it gets executed

Because the script calls `plt.show()` and uses bare `glob` patterns, it isn't
run directly. Use `run_plot.py`:

```
DATA_DIR=<dir-containing-entropy_comparison_*.pt> \
    python scripts/plot/run_plot.py scripts/plot/plot_entropy_vs_depth.py
```

`run_plot.py`:

- `chdir`s into `DATA_DIR` (default `reproduction/`).
- Forces `matplotlib.use("Agg")`.
- Monkey-patches `plt.show` to `savefig <DATA_DIR>/figures/plot_entropy_vs_depth.png`
  at 150 dpi, then `close`.
- `exec`s the target script as `__main__`.

Output: `<DATA_DIR>/figures/plot_entropy_vs_depth.png` (and `_2`, `_3`, … if
the script calls `show()` more than once).

---

## What this depth plot represents

For each of the top-10 features (ranked by total activation, present in
≥ 2 layers), it plots how the **mean per-batch entropy of that feature's
influence distribution** evolves as you go deeper through the residual stream
layers 0 → 5. The hypothesis behind the figure is the one stated in
`compare_entropies.py`'s docstring: features should have lower entropy than
the token vector (more localized in token-space). The depth plot lets you
eyeball whether that localization tightens or loosens with depth, on a
per-feature basis.

---

## Differences vs. `scripts-v1/`'s `fig05_entropy_vs_depth*`

| Aspect | `scripts/plot/plot_entropy_vs_depth.py` | `scripts-v1/figures/fig05_entropy_vs_depth_preset.py` |
| --- | --- | --- |
| Aggregation primitive | mean-across-batches, then **line per top-10 feature** | mean-across-batches, then **violin over all features per layer** |
| Top-N criterion | sum of activations / appearance count, ≥ 2 layers | activation count (# batches active), ties by higher mean entropy |
| Token vector entropy | not shown | red dashed per-layer line |
| `log2(64)` reference | not drawn | drawn |
| Multi-preset support | none — Pythia-70m only | `--preset` and `--all` |
| File discovery | flat `glob` in cwd, latest by mtime | recursive glob under `data/<preset>/`, run dir with most files |
| Headless execution | via `run_plot.py` shim | argparse-driven, native |

---

## Caveat about the data this currently rests on

Every `entropy_comparison_*.pt` file produced **before commit `aa62f69`** was
generated using the contaminated `wikitext-2-train.txt` corpus — text
containing 14,334 `<unk>` tokens that were not present in Pythia's
pretraining and that the BPE tokenizer fragments into raw `<`, `un`, `k`, `>`
chars. Any depth plot drawn from that data inherits the contamination.

To regenerate cleanly, re-run `compare_entropies_multi_layer.py` after the
data-loader port (`aa62f69`); the new corpus is `wikitext-2-raw-v1` from
HuggingFace. The `@-@`, `@.@`, `@,@` markers remain in raw-v1 — those are
inherent to both WikiText-2 distributions on HF and are not fixable by a
config swap.

---

## Update — 2026-04-27 — preset/adapter migration changed Stage-1 output paths

After the preset migration (see `CHANGELOG.md` entry of the same date),
`compare_entropies_multi_layer.py` now writes to a structured run directory:

```
data/<preset>/<timestamp>/
    entropy_comparison_<site>.pt        # NOTE: no timestamp suffix in filename
    entropy_plots_<site>/batch_NNN.png
    run_config.json
    bench.json
data/<preset>/latest -> <timestamp>     # symlink
```

vs. the pre-migration layout `entropy_comparison_<site>_<timestamp>.pt` in cwd.
Single-layer `compare_entropies.py` outputs are unchanged (still timestamped
filename in cwd / `--output-dir`).

This **breaks** the depth-plot glob in `scripts/plot/plot_entropy_vs_depth.py`
and `scripts/plot/notebook_entropy_vs_depth.py`. Their pattern
`entropy_comparison_resid_out_layer{N}_*.pt` requires a literal underscore +
suffix that the new multi-layer outputs do not have.

To re-point the depth plot at a new run, either:

1. Run with `DATA_DIR=data/pythia-70m/latest` and change the glob pattern in
   the plot script from `_*.pt` to `*.pt` (matches both naming schemes), or
2. Run from a directory that contains symlinks to the new files with the
   old `_<timestamp>.pt` naming.

Fixing the plot scripts to handle both names is a separate follow-up; the
data files themselves are correct and complete.
