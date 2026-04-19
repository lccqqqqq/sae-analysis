# Figure generation scripts

One script per paper figure. Each script loads pre-computed `.pt` data files and saves
directly to `paper/figures/`. Run from the repo root.

## Quick reference

| Script | Paper figure | Output file | Required data |
|--------|-------------|-------------|---------------|
| `fig01_unique_tokens_histogram.py` | Fig. 1 | `unique_tokens_histogram.png` | `feature_sparsity_data_resid_out_layer0.pt` |
| `fig02_correlation_heatmap.py` | Fig. 2 | `correlation_heatmap.png` | `correlation_matrix_resid_out_layer3.pt` |
| `fig03_influence_heatmap.py` | Fig. 3 | `influence_heatmap_0.png` | `feature_token_influence_resid_out_layer3.pt` |
| `fig04_entropy_distribution_batches.py` | Fig. 4 | `entropy_distribution_batches.png` | `feature_token_influence_resid_out_layer3.pt` |
| `fig05_entropy_vs_depth.py` | Fig. 5 | `entropy_vs_depth.png` | `entropy_comparison_resid_out_layer{0..5}_*.pt` |
| `fig06_entropy_multilayer_histograms.py` | Fig. 6 | `entropy_vs_activation_multilayer.png` | `entropy_comparison_resid_out_layer{0..4}_*.pt` |
| `fig07_entropy_vs_batchsize.py` | Fig. 7 | `entropy_vs_batchsize.png` | `entropy_vs_batch_size_resid_out_layer2_*.pt` |
| `fig08_entropy_vs_activation.py` | Fig. 8 | `entropy_vs_activation.png` | `feature_token_influence_resid_out_layer5.pt` + `feature_sparsity_data_resid_out_layer5.pt` |

## How to regenerate all figures

**Step 1 — produce the required data files** (see `scripts/analysis/` for details):

```bash
# Figs 1, 2, 3, 4, 8: sparsity + influence data
python scripts/analysis/feature_sparsity.py          # edit SITE in script for each layer
python scripts/analysis/compute_correlations.py       # → correlation_matrix_<site>.pt
python scripts/analysis/feature_token_influence.py    # → feature_token_influence_<site>.pt

# Figs 5, 6: entropy comparison across all layers
python scripts/analysis/compare_entropies_multi_layer.py --layers 0 1 2 3 4 5 --num-batches 50

# Fig 7: context-length sweep
python scripts/analysis/entropy_vs_batch_size.py --site resid_out_layer2
```

**Step 2 — generate each figure:**

```bash
python scripts/figures/fig01_unique_tokens_histogram.py
python scripts/figures/fig02_correlation_heatmap.py
python scripts/figures/fig03_influence_heatmap.py
python scripts/figures/fig04_entropy_distribution_batches.py
python scripts/figures/fig05_entropy_vs_depth.py
python scripts/figures/fig06_entropy_multilayer_histograms.py
python scripts/figures/fig07_entropy_vs_batchsize.py
python scripts/figures/fig08_entropy_vs_activation.py
```

All outputs land in `paper/figures/` for direct inclusion in the LaTeX source.
