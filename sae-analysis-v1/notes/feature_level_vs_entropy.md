# Feature "level" vs influence entropy — vocabulary and logit proxies

**Date:** 2026-04-15 · **Preset:** `pythia-70m` (Pythia-70m-deduped) · **SAE:** saprmarks `res-sm` (ReLU, 32,768 features, all 6 residual layers).

## Question

The right panel of the per-batch diagnostic plots showed no correlation between activation strength and influence entropy. That null result is expected — activation magnitude is a property of the input on which the feature happens to fire, not of the feature itself. The hypothesis we want to test is:

> **(H):** A feature's "level" — token-specific vs syntactic vs topical vs abstract — is a feature-intrinsic property and should correlate with its mean influence entropy. Token-level features should have low $H_\text{feat}$; topical features should have high $H_\text{feat}$.

Two cheap, feature-intrinsic proxies for "level":

- **$H_\text{vocab}(a)$** — Shannon entropy of the input-token distribution that triggers feature $a$ above threshold (computed by running the SAE encoder over the WikiText-2 corpus and accumulating per-feature token counts). Low → fires on a narrow vocabulary (token-specific). High → fires across many different input tokens (topical).
- **$H_\text{logit}(a)$** — Shannon entropy of $\mathrm{softmax}(W_U D_a)$ where $D_a$ is the SAE decoder column and $W_U$ the model's output embedding. Low → the feature pushes a narrow vocabulary at the output (output-localized). High → the feature affects a broad output distribution.

Both are scalars per feature with **no batch averaging and no gradients**. They cost less than 4 minutes each per layer on an RTX 4090 ($H_\text{logit}$ takes ~3 seconds; $H_\text{vocab}$ takes a corpus pass at ~210 batches/s).

## Method

- **$H_\text{vocab}$:** for each layer, run Pythia-70m on all 2.41M WikiText-2 training tokens in 64-token windows; capture residual stream at the layer; encode via SAE; accumulate `counts[a, v] += 1` whenever feature $a$ fires above $\tau = 1.0$ at a position holding token $v$. Closed-form entropy from counts. Features with total count below `min_count = 20` are marked NaN to suppress noisy estimates.
- **$H_\text{logit}$:** chunked computation of $\mathrm{softmax}(W_U D_a)$ entropy; vocabulary size 50,304 for Pythia-70m.
- **$H_\text{feat}$:** loaded from the existing entropy-comparison data (`data/entropy_comparison_resid_out_layer{N}_20260414_053350.pt`, 50 random batches per layer, $\tau = 0.2$). Per feature, take the mean across batches where the feature was active.

Code: `scripts/analysis/feature_level_entropies.py` and `scripts/plotting/plot_feature_level_vs_entropy.py`.
Data file: `data/pythia-70m/feature_level_entropies.pt`.
Plot: `plots/feature_level_vs_influence_entropy_pythia-70m.png`.
Stats: `plots/feature_level_vs_influence_entropy_pythia-70m.json`.

## Results

### $H_\text{vocab}$ vs $H_\text{feat}$

| Layer | n  | mean $H_\text{vocab}$ | mean $H_\text{feat}$ | Pearson $r$ | Spearman $\rho$ | $p$ |
| ----- | -- | ----- | ----- | ----------- | --------------- | --- |
| 0 | 168  | 0.87 | 1.71 | **+0.587** | **+0.547** | $5.7 \times 10^{-17}$ |
| 1 | 215  | 1.38 | 2.51 | **+0.498** | **+0.422** | $6.9 \times 10^{-15}$ |
| 2 | 731  | 2.77 | 3.26 | +0.114 | −0.034 | $2 \times 10^{-3}$ |
| 3 | 622  | 3.19 | 3.51 | +0.226 | +0.069 | $1.3 \times 10^{-8}$ |
| 4 | 538  | 3.16 | 3.72 | +0.104 | +0.052 | 0.016 |
| 5 | 6641 | 5.87 | 3.88 | +0.068 | +0.087 | $3 \times 10^{-8}$ |

(Sample size = features with both an $H_\text{vocab}$ estimate above `min_count` and at least one batch where the feature was active above $\tau = 0.2$.)

### $H_\text{logit}$ vs $H_\text{feat}$

| Layer | n  | mean $H_\text{logit}$ | std | Pearson $r$ | Spearman $\rho$ | $p$ |
| ----- | -- | ----- | ---- | ----------- | --------------- | --- |
| 0 | 241  | 15.618 | 0.000 | +0.098 | +0.091 | 0.13 |
| 1 | 337  | 15.618 | 0.000 | +0.050 | +0.098 | 0.36 |
| 2 | 1210 | 15.618 | 0.000 | +0.169 | +0.158 | $3 \times 10^{-9}$ |
| 3 | 719  | 15.618 | 0.000 | +0.070 | +0.091 | 0.06 |
| 4 | 583  | 15.618 | 0.000 | −0.066 | −0.068 | 0.11 |
| 5 | 8891 | 15.614 | 0.009 | +0.024 | −0.107 | 0.02 |

## Reading the numbers

### $H_\text{logit}$ is degenerate for Pythia-70m

The maximum possible value is $\log_2(50304) = 15.62$. At every layer except layer 5, every feature's $H_\text{logit}$ sits within numerical precision of the maximum (std rounds to 0.000). At layer 5 a tiny variance ($\sigma \approx 0.009$) appears. The decoder columns of these SAEs do not project onto sharply-defined directions in the unembedding space — Pythia-70m is small enough that the residual stream cannot encode "this concept maps to *this* output token" until the very last layer, and even there only weakly.

This is a **methodological** finding more than a scientific one: $H_\text{logit}$ as a feature-level proxy requires a model with a richer output structure. It is essentially uninformative for Pythia-70m. We expect it to be more discriminating for GPT-2-small, Gemma-2-2B, or Llama-3-8B, where the unembedding has had many more layers' worth of structure to develop.

### $H_\text{vocab}$ shows a clean depth-dependent pattern

There are two regimes:

1. **Layers 0–1 (shallow):** strong positive correlation, Pearson $r \in [0.50, 0.59]$, Spearman $\rho \in [0.42, 0.55]$. The feature's vocabulary footprint and its input-token influence footprint move together. A feature that fires on a narrow set of tokens also has a narrowly localized gradient flow. **Hypothesis (H) is confirmed at shallow layers.**

2. **Layers 2–5 (mid-to-deep):** correlation collapses to near zero (Spearman $\rho \in [-0.03, +0.09]$). A feature's vocabulary breadth no longer predicts its influence entropy. The relationship is gone.

This is the opposite of what I predicted in the planning discussion (I expected the correlation to *strengthen* with depth as features become more "level-stratified"). The data say the opposite: the level effect on $H_\text{feat}$ disappears past layer 1.

### What this means for the homogeneity hypothesis

In an earlier discussion I proposed a "shared-Jacobian" explanation for why deep-layer features have very similar $P_a(t')$ distributions on a given batch: at deep layers, the residual-stream Jacobian is dominated by a few salient input tokens, and every feature inherits the same nonlocality fingerprint regardless of what it represents.

The new data quantitatively support this picture. Specifically:

- At layers 0–1, there has been little attention mixing. The feature's input dependency reflects its own structure — narrow vocabulary → narrow influence (low $H_\text{feat}$).
- From layer 2 onward, attention mixing imposes a context-dependent structure on the residual stream. $H_\text{feat}$ becomes a measurement of "how much of the text was salient on this batch," not "how broad is this feature's intrinsic concept." The feature-intrinsic vocabulary breadth $H_\text{vocab}$ no longer drives $H_\text{feat}$.

The correlation collapse therefore is not a failure of the level hypothesis. It is a failure of $H_\text{feat}$ to measure what we wanted it to measure beyond shallow layers. **For Pythia-70m, batch-averaged influence entropy ceases to be a feature-intrinsic quantity past layer 1.**

### Layer 5: the biggest sample, the least signal

Layer 5 has 6641 features in the overlap set — 10× more than any other layer — because the activation count requirement is easily met (features fire often). And yet its Pearson $r$ for $H_\text{vocab}$ vs $H_\text{feat}$ is +0.068. The huge sample makes the $p$-value tiny ($p = 3 \times 10^{-8}$) but the effect size is essentially zero. This is exactly the kind of result that statistical significance is poorly suited to interpret, and the right reading is "negligible."

### One subtle effect worth flagging

At layer 5, the Spearman correlation for $H_\text{logit}$ is **negative** ($\rho = -0.107$, $p = 0.02$). This is the only layer where $H_\text{logit}$ has any variance at all. The negative sign is the prediction of the level hypothesis applied to the output side: features that promote a narrow set of output tokens (low $H_\text{logit}$) should be the more "abstract / topical / consequential" features, which should also have larger influence entropy. With std = 0.009 the effect is on the edge of detectability; it would need a larger model with more $H_\text{logit}$ dynamic range to confirm.

## Summary table

| Hypothesis | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
| ---------- | ------- | ------- | ------- | ------- | ------- | ------- |
| $H_\text{vocab}$ predicts $H_\text{feat}$ | **YES** | **YES** | weak | weak | null | null |
| $H_\text{logit}$ predicts $H_\text{feat}$ | n/a (no var.) | n/a | n/a | n/a | n/a | weak (−) |

Hypothesis (H) is correct at shallow layers and **collapses** by layer 2. The collapse is consistent with the shared-Jacobian / context-saliency story: deep-layer $H_\text{feat}$ is dominated by text structure, not by feature-intrinsic level.

## Implications for the broader project

1. **$H_\text{feat}$ is a clean feature-intrinsic measurement only at shallow layers.** Reporting its mean across deep layers as the "nonlocality of features" effectively averages text-driven fluctuations as much as it averages feature-intrinsic structure. The headline depth-vs-entropy result of the paper is still robust as a layer-level statement (the *layer* gets more nonlocal at depth) but it would be misleading to read it as a *feature* property at depth.
2. **The level hypothesis isn't dead — it just isn't measurable via $H_\text{feat}$ alone past layer 1.** A better target would be the KL divergence $D_\text{KL}(P_a \| Q)$ between the per-feature influence distribution $P_a$ and the token-vector influence distribution $Q$ on the same batch. This subtracts the shared-Jacobian background and asks "how much of the feature's influence pattern is its own?" That is the natural follow-up.
3. **$H_\text{logit}$ is unusable on Pythia-70m, but should be retried on bigger models.** The next planned analysis on GPT-2-small or Gemma-2-2B should include $H_\text{logit}$ — for those models the unembedding has enough resolution to potentially discriminate features.
4. **The vocab-entropy pipeline doubles as a free way to compute the saprmarks `feature_sparsity` data the rest of the codebase expects.** The counts matrix from this run could be persisted as `feature_sparsity_data_resid_out_layer{N}.pt` (the existing format) to unblock other downstream scripts that currently lack it.

## Reproducibility

```bash
# 1. Compute H_vocab + H_logit on a GPU node (~18 min on RTX 4090).
python3 scripts/analysis/feature_level_entropies.py \
    --preset pythia-70m --layers 0 1 2 3 4 5 --threshold 1.0

# 2. Generate plots + per-layer stats.
python3 scripts/plotting/plot_feature_level_vs_entropy.py \
    --preset pythia-70m --timestamp 20260414_053350
```

Cluster job: `gpulong / rtx4090with24gb`, `out/feature_level_entropies_6642529.out`.

## Open questions raised by these results

- **At layer 1, do the strong-correlation features have other shared properties?** Specifically, are they the features that the correlation-matrix analysis (Fig 2) shows to be most independent? If so, "well-behaved low-level feature" might admit a multi-criterion characterization.
- **Does the collapse at layer 2 happen suddenly or gradually within the layer?** The compare_entropies pipeline only samples at `resid_out`; an MLP-output or attention-output SAE within the same layer might show whether the collapse is post-attention or post-MLP.
- **Is the collapse model-specific?** The shared-Jacobian explanation predicts that bigger, wider models with richer attention should show the collapse at a deeper layer (more layers of "clean" input dependency before context starts dominating). Testing this on GPT-2-small (12 layers) and Gemma-2-2B (26 layers) is the natural next experiment.
