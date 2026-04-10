# sae-analysis — project context

This file is the canonical starting point for Claude sessions in this repo. It summarizes the research program, the data flow, and where to look for each experiment. Read `note.md` for the physics motivation and this file for the code map.

## Repository shape

Two layers stacked in one working tree:

- `dictionary_learning/` — vendored copy of Sam Marks / Adam Karvonen / Aaron Mueller's SAE training library (the `saprmarks/dictionary_learning` project). Provides `AutoEncoder`, `AutoEncoderTopK`, `GatedAutoEncoder`, `JumpReluAutoEncoder`, matryoshka / batch-top-k trainers, activation buffers, etc. Commits before `e7084f4` (2025-11-13) are upstream. Not actively developed here; used to load pretrained Pythia-70m SAEs downloaded by `./pretrained_dictionary_downloader.sh` into `dictionaries/pythia-70m-deduped/{site}/10_32768/ae.pt`.
- Top-level `*.py`, `*.ipynb`, `note.md`, `wikitext-2-{train,test}.txt` — the research layer, all owned by this project.

Primary model: `EleutherAI/pythia-70m-deduped`. Primary site: `resid_out_layer3`, with layer sweeps 0–5. Primary corpus: `wikitext-2-train.txt` (also bundled test split). Device preference in scripts: MPS if available, else CPU.

## Research motivation (see note.md)

Treat SAE features as the analog of **bulk operators in AdS/CFT** and input tokens as **boundary operators**.

- Sparsity of features ↔ semi-classical few-excitation regime where bulk spacetime is meaningful.
- Approximate independence of features on the natural text distribution ↔ **emergent bulk locality**. (Features are not independent on arbitrary inputs; the locality is distribution-dependent, just as bulk locality is emergent.)
- Number of input tokens that activate a feature ↔ bulk operator size / depth in the bulk.
- Feature–feature correlation matrix ↔ bulk geometry (the hope is to reconstruct a distance structure from it).
- Shannon / relative entropy of feature distributions ↔ geometric entropy analog; long-term question is whether an RT-like formula exists.
- HKLL-style smearing ↔ possible alternative feature-extraction map.

`note.md` references: Matryoshka SAEs (Bussmann+ 2025), Feature Absorption (Chanin+ 2024), Feature Hedging (Chanin+ 2025), Bricken+ 2023 Towards Monosemanticity, correlation-matrix scaling (arXiv 2410.19750), Zipf (2411.02124), error scaling (2509.02565), cross-layer routing (2503.08200).

## Definition of locality used in this repo

Two complementary notions, both implemented.

1. **Position locality along the token axis (quantitative).** For feature `f_a` at position `t`, layer `z`, compute `J_a(t, z | t') = ‖∂f_a(t,z) / ∂x_{:, t'}‖²` for `t' ≤ t` by automatic differentiation through the Pythia-70m forward pass. Normalize to `P(t' | t, z) = J / ΣJ` and take the Shannon entropy `H = −Σ P log₂ P`. Small `H` = local (small boundary footprint, few upstream tokens matter); large `H` = nonlocal. `e^H` is the "effective number of tokens" in the receptive field. This is the holographic analog of operator size.

2. **Feature independence (emergent-geometry).** Measured via the connected correlator `C_{ab} = ⟨A^a A^b⟩ − ⟨A^a⟩⟨A^b⟩` over token-conditioned feature activations. The hope is that this organizes into a geometric / distance-like structure — a concrete realization of "emergent bulk locality" in feature space.

## Commit timeline (research layer only)

| Date | Commit | Content |
|---|---|---|
| 2025-11-13 | `e7084f4` | Initial research commit: `sae_test.py`, `sae_test_with_prompt.py`, `test_lm_infer.py`. Prompt-level SAE feature injection demos (layer 3, feature 3339). |
| 2025-11-27 | `82c86b3` | `feature_sparsity.py`, `logit_lens.py`, `sae_visualizer.py`, `test_generation.py`, WikiText-2 corpus, sparsity plots. First dataset-scale pipeline. |
| 2025-11-28 | `07738c2` | `compute_correlations.py` + first long version of `note.md` laying out the AdS/CFT analogy. Produces `correlation_matrix.pt`. |
| 2026-04-06 | `7986bc7` + `84fe43a` + `2960c2a` | The big push: gradient-based influence and entropy machinery, location analysis, the large analysis notebooks, README pipeline docs. This is the current research frontier. |

**Most recent ideas (as of 2026-04):** the gradient-based nonlocality entropy program — measuring how spread-out a feature's dependence on upstream tokens is, comparing against the raw residual stream entropy, and sweeping over depth, batch size, and activation frequency to test whether SAE features constitute a "low-entropy decomposition" of the residual stream.

## Data pipeline (authoritative order)

All scripts are top-level and use `site = "resid_out_layer<N>"` as the key. Recommended order (mirrors `README.md`):

```
feature_sparsity.py                   →  feature_sparsity_data_<site>.pt + feature_sparsity_<site>.csv
  │
  ├─ compute_correlations.py          →  correlation_matrix_<site>.pt
  ├─ feature_location_analysis.py     →  feature_location_data.pt + feature_location.csv  (parallel, does its own pass)
  ├─ feature_token_influence.py       →  feature_token_influence_<site>.pt
  │    (selects leading features by frequency, computes gradient J(t,z|t'))
  │
  ├─ token_vector_influence.py        →  token_vector_influence_<site>.pt  (residual-stream baseline, independent)
  │
  ├─ compare_entropies.py             →  entropy_comparison_<site>_<timestamp>.pt
  ├─ compare_entropies_multi_layer.py →  one entropy_comparison_<site>_<timestamp>.pt per layer
  └─ entropy_vs_batch_size.py         →  entropy_vs_batch_size_<site>_<timestamp>.pt + plots
```

Downstream plot / analysis scripts read the saved `.pt` files and never regenerate data:

- `plot_feature_entropy_histogram.py`, `plot_all_features_entropy_histogram.py` ← `feature_token_influence_<site>.pt`
- `plot_entropy_vs_activation.py` ← `feature_token_influence_<site>.pt` + `feature_sparsity_data_<site>.pt`
- `plot_entropy_vs_depth.py`, `notebook_entropy_vs_depth.py` ← `entropy_comparison_<site>_<timestamp>.pt` across layers
- `plot_entropy_vs_batch_size_notebook.py` ← `entropy_vs_batch_size_<site>_<timestamp>.pt`
- `analyze_feature_token_influence*.py` ← `feature_token_influence_<site>.pt` (various notebook-style analyses)
- `analyze_feature_token_influence_with_batches.py` ← `entropy_comparison_<site>_<timestamp>.pt`
- `feature_analysis{,_cleaned,_backup,_v4}.ipynb` ← mostly `feature_sparsity_data_<site>.pt` plus the entropy files

## Main experiments in detail

### Experiment 1 — Feature activation frequency / sparsity
**Script:** `feature_sparsity.py`. Runs Pythia-70m on WikiText-2, hooks the chosen site, applies the SAE encoder with threshold `THRESHOLD = 1.0`, records activation counts, triggering tokens, and per-feature frequency. Writes `feature_sparsity_data_<site>.pt` (with `feature_counts`, `frequencies`, `total_tokens`) and the CSV. Feeds nearly everything else.

### Experiment 2 — Feature–feature correlations
**Script:** `compute_correlations.py`. Reads `feature_sparsity_data_<site>.pt`, builds `⟨A^a⟩`, `⟨A^a A^b⟩`, and `C_{ab}`. Uses the token-conditioned formulation `A^a(t) = N(a,t)/N(t)` so that averages become frequency-weighted sums — see the algebra in `note.md` §"Correlation between features". Output: `correlation_matrix.pt` (or `correlation_matrix_<site>.pt`). Intended use: study whether `C_{ab}` has a geometric / low-dimensional structure (the emergent-bulk-geometry question).

### Experiment 3 — Feature location / scale hierarchy
**Script:** `feature_location_analysis.py`. Computes `ℓ_a = Σ_t t · n_a(t) / Σ_t n_a(t)` — the mean batch position at which feature `a` fires — to look for a "small-scale (early) vs large-scale (late)" hierarchy, since later positions in a causal transformer can depend on more inputs. Output: `feature_location_data.pt` + `feature_location.csv`.

### Experiment 4 — Feature token-influence nonlocality (gradient entropy)
**Script:** `feature_token_influence.py`. This is the core of the April 2026 push and the one the user is most likely to iterate on.

- **Frequency filter (`feature_token_influence.py:304-322`):** loads `feature_sparsity_data_<site>.pt`, keeps features with `frequency > MIN_FREQ = 0.001` (≥ 0.1% of tokens), caps at `MAX_FEATURES = 500` top-by-frequency. This "leading features" set is the set of features with enough statistics to be worth entropy estimation.
- **Gradient pipeline (`feature_token_influence.py:96-189` `process_batch_with_influence`):** replaces `model.gpt_neox.embed_in` with a `DummyEmbed` wrapping the embedding output in a leaf tensor with `requires_grad=True`. Runs forward with a hook on `gpt_neox.layers[layer_idx]` to capture the residual stream, applies the SAE encoder, then for every leading feature whose activation at the **last position** exceeds `THRESHOLD = 1.0`, backpropagates `feat_activation.backward(retain_graph=True)` through the embedded inputs and stores `J(t') = Σ_μ (∂f_a / ∂x_{μ,t'})²` as a length-`BATCH_SIZE = 64` array.
- **Loop:** processes up to `MAX_BATCHES = 5000` contiguous 64-token windows from `wikitext-2-train.txt`, checkpointing every `CHECKPOINT_INTERVAL = 100` batches to `feature_token_influence_<site>_checkpoint.pt`, resumable.
- **Aggregation:** features with `len(influence_list) >= MIN_FEATURE_ACTIVATIONS = 10` are kept; output stores `mean_influence`, `std_influence`, raw `all_influences`, `num_samples` per feature.
- **Entropy computation (plot scripts):** `plot_feature_entropy_histogram.py:36-52` and siblings normalize each `J` vector with `ε = 1e-12`, compute `scipy.stats.entropy(P, base=2)` in bits. Max possible value is `log₂(64) = 6` bits (uniform), min is `0` (all mass on one token).

The three views built on this:
- `plot_feature_entropy_histogram.py` — histogram of `H` across batches for one chosen feature (default feature 100).
- `plot_all_features_entropy_histogram.py` — small-multiples histogram for the top-10 most-sampled features.
- `plot_entropy_vs_activation.py` — scatter of per-feature mean entropy vs activation probability. Direct test of "do rarer features have different nonlocality than common ones?"

### Experiment 5 — Feature vs residual-stream entropy comparison
**Scripts:** `compare_entropies.py` and `compare_entropies_multi_layer.py`, baseline by `token_vector_influence.py`. On the same batches, compute both the feature entropies `H_feat` (via Experiment 4's machinery) and the entropy `H_token` of the residual-stream direction's influence through the decoder map `D^a_ν I_{aμ}`. The headline conjecture from `note.md` §"Nonlocality of internal token vector versus features" is **`H_token > H_feat`** on average — SAE features are a lower-entropy (more localized) decomposition of the same residual-stream direction. Threshold here is `THRESHOLD = 0.2` (looser than Experiment 4's `1.0`). Output: `entropy_comparison_<site>_<timestamp>.pt` per batch or per layer. `plot_entropy_vs_depth.py` consumes these across layers 0–5.

### Experiment 6 — Entropy vs batch (window) size
**Script:** `entropy_vs_batch_size.py`. Sweeps the gradient window length to test whether the H_token − H_feat gap is a finite-window artifact or a real property that survives larger contexts. Output: `entropy_vs_batch_size_<site>_<timestamp>.pt` plus a plot directory. Read by `plot_entropy_vs_batch_size_notebook.py`.

## Conventions and gotchas

- **Site naming is load-bearing.** Every downstream script pattern-matches on `f"..._{site}.pt"`. Changing it mid-pipeline breaks the chain. Layer index is parsed from `site.rsplit("layer", 1)[-1]`.
- **Thresholds differ between experiments.** `feature_sparsity.py` and `feature_token_influence.py` use `THRESHOLD = 1.0`; `compare_entropies.py` uses `THRESHOLD = 0.2`. This is intentional but easy to trip over.
- **Embedding-layer swap is not thread-safe.** `process_batch_with_influence` monkey-patches `model.gpt_neox.embed_in` with a `DummyEmbed` inside a `try/finally`. Do not parallelize across that function with shared model state.
- **MPS / CPU only.** All scripts default to MPS if available, CPU otherwise. No CUDA-specific paths in the research layer.
- **WikiText-2 is committed to the repo.** `wikitext-2-train.txt` (~10 MB) and `wikitext-2-test.txt` are in-tree for reproducibility; don't delete them without checking.
- **Several large output files are also in-tree.** `correlation_matrix.pt` (~450 KB), `feature_sparsity_data.pt` (~3.4 MB), `feature_sparsity.csv` (~850 KB), `feature_analysis*.ipynb` (multi-MB notebooks). Be careful when committing — `strip_notebook_outputs.py` exists for this reason.
- **Notebook duplication.** `feature_analysis.ipynb`, `feature_analysis_backup.ipynb`, `feature_analysis_cleaned.ipynb`, `feature_analysis_v4.ipynb` overlap heavily. `_cleaned` is the small one; the others are ~5 MB each. `create_minimal_notebook.py` and `fix_notebook.py` are rescue utilities.
- **The `_simple` / `_notebook` / `_final` / `_with_batches` suffixes on `analyze_feature_token_influence*.py` are iterations, not a well-structured module tree.** They were added in one commit as notebook-cell dumps.

## Standalone demos (not part of the main pipeline)

- `sae_test.py` — inject a decoded SAE feature into a chosen layer, print how next-token logits change.
- `sae_test_with_prompt.py` — pick a feature from prompt-dependent activation, baseline vs patched logits, saves a comparison PNG.
- `sae_visualizer.py` — prompt-level visualizer for feature activations and vocabulary projections.
- `logit_lens.py` — prints per-layer token predictions from intermediate hidden states.
- `test_generation.py` / `test_lm_infer.py` — generation sanity checks.

## Where to look first

- **Physics motivation and definitions** → `note.md`.
- **End-to-end command order** → `README.md` "Local scripts" section.
- **Frequency-filtered entropy experiment** → `feature_token_influence.py` + `plot_entropy_vs_activation.py`.
- **Low-entropy-decomposition test** → `compare_entropies.py` / `compare_entropies_multi_layer.py`.
- **Correlation / emergent geometry** → `compute_correlations.py` + `correlation_matrix.pt`.
- **Layer-depth sweep** → `plot_entropy_vs_depth.py` reading multiple `entropy_comparison_<site>_<timestamp>.pt`.
