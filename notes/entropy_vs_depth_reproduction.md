# Reproducing the "Entropy of Leading Features vs Layer Depth" plot

This plot shows the **mean entropy (bits) of each leading SAE feature** as a line across layers 0–5, visualizing how a feature's nonlocality changes with depth. The conjecture is that deeper layers produce features with broader boundary footprints (higher entropy).

## Scripts that produce this plot

Two scripts produce this plot — they're essentially the same code:

| Script | Use case |
|---|---|
| `plot_entropy_vs_depth.py` | Standalone `.py` (tries to open a window) |
| `notebook_entropy_vs_depth.py` | Copy-paste into a Jupyter cell (`plt.show()` renders inline) |

Both auto-discover the most recent `entropy_comparison_resid_out_layer<N>_*.pt` per layer via glob.

## How to reproduce

### Step 1 — Generate the data

No `.pt` files exist yet. Run:

```bash
python compare_entropies_multi_layer.py --layers 0 1 2 3 4 5 --num-batches 10
```

This runs all 6 layers in one pass, producing 6 files:
- `entropy_comparison_resid_out_layer0_<timestamp>.pt`
- `entropy_comparison_resid_out_layer1_<timestamp>.pt`
- … through layer 5

Each file contains per-batch `feature_entropies` (dict of feature_idx → H in bits) and `token_vector_entropy` (the residual-stream baseline H), plus per-batch comparison PNGs in `entropy_plots_resid_out_layer<N>_<timestamp>/`.

### Step 2 — Plot entropy vs depth

```bash
python plot_entropy_vs_depth.py
```

Or paste `notebook_entropy_vs_depth.py` into a Jupyter cell for inline rendering.

The plot shows the top-10 features (ranked by activation, appearing in ≥2 layers) with entropy on the y-axis and layer on the x-axis.

## What the plot shows vs what the per-batch PNGs show

The **depth plot** focuses on feature-level entropy trajectories across layers.

The comparison with `token_vector_entropy` (the residual-stream baseline) is in the **per-batch PNGs** from Step 1 (`entropy_plots_resid_out_layer<N>_<timestamp>/`), not in this depth plot.

## Notes

- Step 1 takes a while (loads Pythia-70m, runs 10 batches × 6 layers with gradient backprop per active feature). On MPS it's on the order of minutes.
- The data generation script (`compare_entropies_multi_layer.py`) uses `THRESHOLD = 0.2` for active-feature selection, which is looser than `feature_token_influence.py`'s `THRESHOLD = 1.0`.
- Entropy is capped at `log₂(64) = 6` bits for the default `BATCH_SIZE = 64`.
- Leading features are ranked by total activation across layers, filtered to those appearing in ≥2 layers.
