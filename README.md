# SAE Analysis

This repository is for analyzing SAEs (Sparse Autoencoders) on neural network activations. It uses the SAE developed by Samuel Marks, Adam Karvonen, and Aaron Mueller. The method also applies to other SAE. In `deprecated/sae_test.py`, we obtain the features encoded for a given input prompt, and then decode a feature to see what tokens it corresponds to.

Install the requirements:
```bash
uv venv .venv && source .venv/bin/activate
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio

# Core libs
uv pip install transformers accelerate einops datasets
# (optional) convenience libs
uv pip install tqdm numpy matplotlib scipy
```

Download the pretrained SAE dictionaries:
```bash
mkdir -p dictionaries
cd dictionaries
wget https://huggingface.co/saprmarks/pythia-70m-deduped-saes/resolve/main/dictionaries_pythia-70m-deduped_10.zip
unzip dictionaries_pythia-70m-deduped_10.zip
cd ..
```

## Local scripts

Recommended order for the data-producing scripts:

1. Run `scripts/analysis/feature_sparsity.py` first for a site/layer such as `resid_out_layer3`.
   It produces `feature_sparsity_data_<site>.pt` and `feature_sparsity_<site>.csv`, which are used by later analysis scripts.
2. Optional next steps from the sparsity output:
   `scripts/analysis/compute_correlations.py` reads `feature_sparsity_data_<site>.pt` and writes `correlation_matrix_<site>.pt`.
   `scripts/analysis/feature_location_analysis.py` runs its own pass over the text data and writes `feature_location_data.pt` and `feature_location.csv`.
3. Run `scripts/analysis/feature_token_influence.py` after `scripts/analysis/feature_sparsity.py`.
   It depends on `feature_sparsity_data_<site>.pt` to decide which features to track, and writes `feature_token_influence_<site>.pt`.
4. Run `scripts/analysis/token_vector_influence.py` when you want the non-SAE baseline influence for the same site.
   It does not depend on `scripts/analysis/feature_sparsity.py`, and writes `token_vector_influence_<site>.pt`.
5. Run `scripts/analysis/compare_entropies.py` or `scripts/analysis/compare_entropies_multi_layer.py` when you want feature-vs-token entropy comparisons.
   These scripts recompute the needed feature and token-vector influences internally and write `entropy_comparison_<site>_<timestamp>.pt`.
6. Run `scripts/analysis/entropy_vs_context_len.py` after that if you want context-length sensitivity plots for a site.
   It writes `entropy_vs_context_len_<site>_<timestamp>.pt` plus a plot directory.
7. Open the notebook helpers and plotting snippets after the corresponding data files exist.
   Most of them read the saved `.pt` outputs above rather than generating data from scratch.

- **Model / SAE inspection**
  - `deprecated/sae_test.py`: standalone demo that injects a decoded SAE feature into a chosen layer and reports how next-token logits change.
  - `deprecated/sae_test_with_prompt.py`: standalone demo that picks a feature based on prompt-dependent activation, compares baseline vs patched logits, and saves a comparison plot.
  - `deprecated/sae_visualizer.py`: standalone prompt-level visualizer for feature activations and vocabulary projections.
  - `deprecated/logit_lens.py`: standalone utility that prints per-layer token predictions from intermediate hidden states.
  - `deprecated/test_generation.py`: standalone language-model generation sanity check.

- **Dataset-level feature statistics**
  - `scripts/analysis/feature_sparsity.py`: first-stage batch analysis. Measures per-feature activation frequency on text data, records triggering tokens, and saves `feature_sparsity_data_<site>.pt` plus `feature_sparsity_<site>.csv`.
  - `scripts/analysis/compute_correlations.py`: downstream of `scripts/analysis/feature_sparsity.py`. Reads `feature_sparsity_data_<site>.pt` and writes `correlation_matrix_<site>.pt`.
  - `scripts/analysis/feature_location_analysis.py`: parallel first-stage analysis. Runs directly on the dataset and writes `feature_location_data.pt` plus `feature_location.csv`.

- **Influence and entropy analysis**
  - `scripts/analysis/feature_token_influence.py`: downstream of `scripts/analysis/feature_sparsity.py`. Reads `feature_sparsity_data_<site>.pt`, computes token-to-feature influence distributions for selected features, and writes `feature_token_influence_<site>.pt`.
  - `scripts/analysis/token_vector_influence.py`: independent baseline analysis. Computes influence norms for the raw residual/token vector and writes `token_vector_influence_<site>.pt`.
  - `scripts/analysis/compare_entropies.py`: combined analysis for one layer. Recomputes both feature and token-vector influence quantities internally and writes `entropy_comparison_<site>_<timestamp>.pt`.
  - `scripts/analysis/compare_entropies_multi_layer.py`: multi-layer version of `scripts/analysis/compare_entropies.py`; writes one `entropy_comparison_<site>_<timestamp>.pt` file per layer.
  - `scripts/analysis/entropy_vs_context_len.py`: studies how feature entropy changes as the context length (window size) changes and writes `entropy_vs_context_len_<site>_<timestamp>.pt` plus plots.

- **Notebook helpers and plotting snippets**
  - `notebooks/feature_analysis.ipynb`, `notebooks/feature_analysis_backup.ipynb`, `notebooks/feature_analysis_cleaned.ipynb`, `notebooks/feature_analysis_v4.ipynb`: notebooks for exploratory analysis of saved outputs, primarily the files from `scripts/analysis/feature_sparsity.py`.
  - `scripts/plot/plot_entropy_vs_depth.py`, `scripts/plot/notebook_entropy_vs_depth.py`: read saved `entropy_comparison_<site>_<timestamp>.pt` files across layers.
  - `scripts/plot/plot_entropy_vs_context_len_notebook.py`: reads `entropy_vs_context_len_<site>_<timestamp>.pt`.
  - `scripts/plot/plot_entropy_vs_activation.py`: reads both `feature_token_influence_<site>.pt` and `feature_sparsity_data_<site>.pt`.
  - `scripts/plot/plot_feature_entropy_histogram.py`, `scripts/plot/plot_all_features_entropy_histogram.py`: read `feature_token_influence_<site>.pt`.
  - `deprecated/analyze_feature_token_influence.py`, `deprecated/analyze_feature_token_influence_simple.py`, `deprecated/analyze_feature_token_influence_notebook.py`, `deprecated/analyze_feature_token_influence_final.py`: read `feature_token_influence_<site>.pt`.
  - `deprecated/analyze_feature_token_influence_with_batches.py`: reads an `entropy_comparison_<site>_<timestamp>.pt` file.

- **Notebook maintenance utilities**
  - `deprecated/strip_notebook_outputs.py`: removes notebook outputs to reduce file size.
  - `deprecated/fix_notebook.py`: attempts to repair corrupted notebook JSON.
  - `deprecated/create_minimal_notebook.py`: creates a minimal valid notebook shell from an existing notebook.
