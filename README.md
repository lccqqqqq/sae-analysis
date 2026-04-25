# SAE Analysis

This repository analyses SAEs (Sparse Autoencoders) on neural-network
activations. The pipeline measures how many input tokens influence each
SAE feature in `EleutherAI/pythia-70m-deduped`, using a gradient-based
entropy.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio
uv pip install transformers accelerate einops datasets tqdm numpy matplotlib scipy
```

Pretrained SAE weights (Marks–Karvonen–Mueller, 32,768 features × 6 layers):

```bash
mkdir -p dictionaries
cd dictionaries
wget https://huggingface.co/saprmarks/pythia-70m-deduped-saes/resolve/main/dictionaries_pythia-70m-deduped_10.zip
unzip dictionaries_pythia-70m-deduped_10.zip
cd ..
```

This populates `dictionaries/pythia-70m-deduped/<site>/10_32768/ae.pt` for
every site (`resid_out_layer{0..5}`, `mlp_out_layer{0..5}`,
`attn_out_layer{0..5}`, `embed`).

## Local scripts

All paths below are relative to the repo root.

### Pipeline (`scripts/analysis/`)

Recommended order for the data-producing scripts:

1. **`scripts/analysis/feature_sparsity.py`** — first-stage batch analysis.
   Measures per-feature activation frequency on WikiText-2, records
   triggering tokens, saves `feature_sparsity_data_<site>.pt` plus
   `feature_sparsity_<site>.csv`.
2. From the sparsity output:
   - **`scripts/analysis/compute_correlations.py`** — reads
     `feature_sparsity_data_<site>.pt`, writes `correlation_matrix_<site>.pt`.
   - **`scripts/analysis/feature_location_analysis.py`** — runs its own pass
     over the text data, writes `feature_location_data.pt` plus
     `feature_location.csv`.
3. **`scripts/analysis/feature_token_influence.py`** — depends on
   `feature_sparsity_data_<site>.pt` to choose which features to track,
   writes `feature_token_influence_<site>.pt`.
4. **`scripts/analysis/token_vector_influence.py`** — non-SAE baseline.
   Independent of step 1; writes `token_vector_influence_<site>.pt`.
5. **`scripts/analysis/compare_entropies.py`** /
   **`compare_entropies_multi_layer.py`** — feature-vs-token entropy
   comparisons. Recompute both influence quantities internally; write
   `entropy_comparison_<site>_<timestamp>.pt`.
6. **`scripts/analysis/entropy_vs_batch_size.py`** — context-length sweep.
   Writes `entropy_vs_batch_size_<site>_<timestamp>.pt` plus an inline plot.

### Figures (`scripts/plot/`)

Each plot script is a notebook-cell-style file that loads one of the `.pt`
outputs above and renders a figure via `matplotlib.pyplot`. Use the runner
to render headlessly:

```bash
python scripts/plot/run_plot.py scripts/plot/plot_entropy_vs_depth.py
```

The runner `chdir`s into `$DATA_DIR` (default `reproduction/`) so bare-name
data lookups in the plot scripts resolve, and saves figures under
`$DATA_DIR/figures/`.

| Plot script | Reads |
| ----------- | ----- |
| `plot_entropy_vs_depth.py`, `notebook_entropy_vs_depth.py` | `entropy_comparison_<site>_<ts>.pt` for layers 0-5 |
| `plot_entropy_vs_batch_size_notebook.py` | `entropy_vs_batch_size_<site>_<ts>.pt` |
| `plot_entropy_vs_activation.py` | `feature_token_influence_<site>.pt` + `feature_sparsity_data_<site>.pt` |
| `plot_entropy_vs_activation_layer5.py` | same, layer-5 variant |
| `plot_feature_entropy_histogram.py`, `plot_all_features_entropy_histogram.py` | `feature_token_influence_<site>.pt` |

### Notebooks (`notebooks/`)

`feature_analysis_cleaned.ipynb` and three earlier variants — exploratory
analysis of saved outputs, primarily the files from `feature_sparsity.py`.

### Deprecated (`deprecated/`)

Scripts kept for reference but not part of the paper-figure pipeline:

- `analyze_feature_token_influence_*.py` (5 files): iterative dev variants
  that all produce the same per-batch token-influence bar plot.
- `sae_test.py`, `sae_test_with_prompt.py`, `sae_visualizer.py`: standalone
  SAE-feature-patching demos.
- `logit_lens.py`, `test_generation.py`, `test_lm_infer.py`: per-layer /
  generation sanity checks.
- `strip_notebook_outputs.py`, `fix_notebook.py`, `create_minimal_notebook.py`:
  notebook-maintenance utilities.

## Notes

The full theoretical motivation (the AdS/CFT analogy and the operator-size /
correlation / entropy quantities) is in [`note.md`](note.md).
