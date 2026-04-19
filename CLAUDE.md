# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is the code and paper for "Analyzing the Nonlocality of Sparse Autoencoder Features" (Xiao-Liang Qi, Stanford). It measures how many input tokens influence each SAE feature in Pythia-70m-deduped, using a gradient-based entropy measure. The conceptual motivation is an analogy with holographic duality (AdS/CFT), where SAE features play the role of bulk fields and input tokens are boundary operators.

For the full paper agent persona, figure reproduction guide, and extension ideas, see `AGENTS.md`.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install . torch torchvision torchaudio
uv pip install transformers accelerate einops datasets tqdm numpy matplotlib scipy
```

External data (all auto-downloaded on first use, cached under `~/.cache/huggingface/`):

- WikiText-2 corpus — loaded via `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")` inside `scripts/analysis/data_loader.load_wikitext_train_text()`; every script that needs the corpus calls that helper.
- Pretrained SAEs — `saprmarks/pythia-70m-deduped-saes` (Pythia) and the respective HF releases for the other presets, via `scripts/analysis/sae_adapters.load_sae()`.

All analysis scripts must be run from the **repo root**. Scripts in `scripts/analysis/` import each other (`compare_entropies.py` imports from `feature_token_influence.py`), so the working directory matters.

## Core methodology: how entropy is defined

The central quantity is the **influence entropy** of an SAE feature. The chain is:

1. **Forward pass with gradient tracking.** The embedding layer is monkey-patched with a `DummyEmbed` that wraps the embeddings as a leaf tensor with `requires_grad=True`. This makes `x_{mu,t'}` a differentiable input.

2. **Influence matrix.** For feature `f_a` at the last token position `t=L` and layer `n`:

   ```text
   I_{a,mu}(t,n | t') = df_a(t,n) / dx_{mu,t'}
   ```

   Computed via `torch.autograd.grad` with `retain_graph=True` for each active feature.

3. **Influence strength.** Squared L2 norm over the embedding dimension (chosen for additivity):

   ```text
   J_a(t,n | t') = sum_mu [I_{a,mu}]^2       shape: [seq_len]
   ```

4. **Normalize to probability.** `P(t') = J(t') / sum J(t')` with eps=1e-12.

5. **Shannon entropy.** `S = -sum P log2 P` in bits. The quantity `2^S` is the effective number of input tokens that matter. Range: 0 (single-token dependence) to `log2(L)` (uniform).

**Why the last token position?** The causal attention mask gives position `t=L` access to all preceding tokens. Any earlier position would artificially truncate the influence distribution.

**Why average over active batches only?** Inactive features have numerically zero gradient (ReLU below threshold), which would bias entropy toward 0.

### Token vector entropy (baseline comparison)

Same idea applied to the full residual stream vector `y_nu(t,n)` at the last position, using `torch.autograd.functional.jacobian` for the full Jacobian of shape `[d_model, L, d_model]`. The influence is `R(t') = sum_{nu,mu} (dy_nu/dx_{mu,t'})^2`. The conjecture (confirmed): **H_token > H_feat** on average — SAE features are a lower-entropy decomposition than the residual stream direction they reconstruct.

## How features are filtered

Three different filtering stages operate with different thresholds:

| Script | What is filtered | Threshold | Selection |
| ------ | ---------------- | --------- | --------- |
| `feature_sparsity.py` | Activation counting | THRESHOLD=1.0 | All 32,768 features; records counts and frequencies |
| `feature_token_influence.py` | Gradient computation (single-layer) | THRESHOLD=1.0 | "Leading features": freq > 0.1%, capped at top 500 by frequency |
| `compare_entropies.py` | Entropy comparison (multi-layer) | THRESHOLD=0.2 | All 32,768 are candidates; per-batch filtering by activation at last position |

The lower threshold (0.2) in `compare_entropies.py` captures more lightly-active features. This is intentional but easy to trip over when comparing outputs across scripts.

"Leading features" (`feature_token_influence.py:307-322`): load frequencies from `feature_sparsity_data_{site}.pt`, keep features with frequency > MIN_FREQ=0.001 (>= 0.1% of tokens), cap at MAX_FEATURES=500 by top frequency.

## Reproducing the entropy-vs-depth result (Fig 5 and Fig 6)

This is the headline finding: mean feature entropy increases monotonically from ~0.9 bits (layer 0) to ~4.0 bits (layer 5).

**Step 1 — Generate entropy comparison data for all layers:**

```bash
cd /path/to/repo
python scripts/analysis/compare_entropies_multi_layer.py --layers 0 1 2 3 4 5 --num-batches 50
```

This produces `entropy_comparison_resid_out_layer{0..5}_{timestamp}.pt` (one per layer). Takes ~2-4 hours on MPS. Start with `--num-batches 10` for a quick test.

Note: this script does NOT require `feature_sparsity_data_*.pt` — it checks all 32,768 features per batch and filters by activation threshold (0.2) on the fly.

**Step 2 — Generate the figure:**

```bash
python scripts/figures/fig05_entropy_vs_depth.py    # Fig 5: scatter + lines
python scripts/figures/fig06_entropy_multilayer_histograms.py  # Fig 6: histograms
```

These scripts find the most recent `entropy_comparison_*.pt` files by modification time.

**What the figure shows:**

- Gray dots: each dot is one feature's mean entropy (averaged over batches where it was active)
- Colored lines: 10 most-activated features tracked across layers
- Blue dashed: average entropy of top-20 features per layer
- Red dashed: token vector entropy per layer

**Published values (mean +/- std):** L0: 0.898 +/- 0.618, L1: 1.859 +/- 0.880, L2: 2.484 +/- 0.922, L3: 3.163 +/- 0.842, L4: 3.705 +/- 0.809, L5: 4.054 bits.

## Data pipeline (dependency order)

```text
feature_sparsity.py          → feature_sparsity_data_{site}.pt  (stage 1, needed by stages 2a, 3, fig08)
compute_correlations.py      → correlation_matrix_{site}.pt     (stage 2a, reads stage 1)
feature_token_influence.py   → feature_token_influence_{site}.pt (stage 3, reads stage 1, for figs 3,4,8)
compare_entropies_multi_layer.py → entropy_comparison_{site}_{ts}.pt  (stage 4, independent of stages 1-3)
entropy_vs_batch_size.py     → entropy_vs_batch_size_{site}_{ts}.pt   (stage 5, independent)
```

Stage 4 (`compare_entropies*`) is self-contained — it loads the model, SAE, and data directly. Stages 3 and 8 (entropy vs activation) need stage 1 output.

## Architecture: how the gradient flows

The gradient computation path (in `feature_token_influence.py:96-189`):

```text
input_ids [1, 64]
  → embed_layer → input_embeds [1, 64, 512]  (detached, requires_grad=True)
  → DummyEmbed replaces model.gpt_neox.embed_in
  → model forward pass (causal transformer)
  → hook captures residual stream at gpt_neox.layers[layer_idx]  →  resid [1, 64, 512]
  → SAE encoder: ReLU(W_enc @ resid + b_enc)  →  feats [1, 64, 32768]
  → for each feature active at last position (feats[0, -1, feat_idx] > threshold):
      feat_activation.backward(retain_graph=True)
      grads = input_embeds.grad  →  [1, 64, 512]
      J(t') = (grads ** 2).sum(dim=-1)  →  [64]   (influence per token position)
```

The `DummyEmbed` swap is wrapped in `try/finally` to restore the original. Not thread-safe.

For token vector influence (`token_vector_influence.py`), the same DummyEmbed trick is used, but instead of per-feature `backward()`, it calls `torch.autograd.functional.jacobian` which computes the full `[512, 64, 512]` Jacobian in one vectorized call. Memory: ~16M floats per batch.

## Gotchas

- **Site naming is load-bearing.** Every `.pt` file uses `f"..._{site}.pt"`. Layer index is parsed from `site.rsplit("layer", 1)[-1]`.
- **Scripts in `scripts/analysis/` import from each other.** `compare_entropies.py` imports from `feature_token_influence.py` and `token_vector_influence.py`. Run from repo root with `scripts/analysis/` in the Python path, or `cd scripts/analysis/` first.
- **MPS / CPU only.** No CUDA paths. Device selection: `"mps" if torch.backends.mps.is_available() else "cpu"`.
- **Batch size = context length.** "Batch size 64" means a 64-token contiguous window, not 64 independent sequences. This is the `L` in the entropy formula.
- **The eps=1e-12 regularization** in entropy computation adds a tiny floor to all probabilities before normalizing. This prevents log(0) but slightly biases entropy upward when the distribution is very sparse.
- **`compare_entropies.py` uses random batch selection by default.** Pass `--random-seed N` for reproducibility.
- **Checkpoint files** (`*_checkpoint.pt`) are automatically deleted on successful completion. If a run is interrupted, it resumes from the checkpoint.

## Physics intuition

The holographic analogy (see `note.md` for full development):

- **Input tokens ↔ boundary CFT operators.** The token sequence is the "boundary" of the system.
- **SAE features ↔ bulk fields.** They are nonlocal in the input, approximately independent under the data distribution, and sparse.
- **Layer depth ↔ radial/bulk depth.** Deeper layers have access to more tokens and encode more abstract concepts, analogous to fields deeper in the AdS bulk being reconstructed from more boundary operators.
- **Influence entropy ↔ operator size.** `2^S` counts the effective number of boundary operators needed to reconstruct the bulk field.

The key prediction: entropy should grow with layer depth. This is confirmed empirically and is the main result of the paper.

The "low-entropy decomposition" finding: individual features have lower entropy than the token vector `y = sum_a D_a f_a`. This is analogous to saying that individual bulk excitations are simpler (fewer boundary operators needed) than the full field configuration.

## File naming conventions for outputs

- `feature_sparsity_data_{site}.pt` — activation counts, frequencies, token associations
- `feature_sparsity_{site}.csv` — human-readable version of above
- `correlation_matrix_{site}.pt` — feature-feature correlation matrix
- `feature_token_influence_{site}.pt` — per-feature gradient influence distributions
- `entropy_comparison_{site}_{timestamp}.pt` — per-batch feature + token vector entropies
- `entropy_vs_batch_size_{site}_{timestamp}.pt` — entropy as function of context length

Timestamp format: `YYYYMMDD_HHMMSS`. Figure scripts find the most recent file by modification time using `glob.glob` + `max(key=st_mtime)`.
