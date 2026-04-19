# Notebooks

| Notebook | Description |
|---|---|
| `feature_analysis.ipynb` | Full exploratory analysis of feature sparsity, location, and correlation (outputs embedded) |
| `feature_analysis_cleaned.ipynb` | Same notebook with outputs stripped — use this as the clean starting point |

## Prerequisites

Run the data-producing scripts first (see the repo README for the recommended order):

1. `scripts/analysis/feature_sparsity.py` — produces `feature_sparsity_data_<site>.pt` and `feature_sparsity_<site>.csv`
2. `scripts/analysis/compute_correlations.py` — produces `correlation_matrix_<site>.pt`
3. `scripts/analysis/feature_location_analysis.py` — produces `feature_location_data.pt`

## Running

```bash
jupyter notebook notebooks/feature_analysis_cleaned.ipynb
```
