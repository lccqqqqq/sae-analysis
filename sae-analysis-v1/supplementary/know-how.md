# Know-How: Methodology Decisions and Tacit Knowledge

This document captures reasoning behind key design choices in the paper that are not fully explained in the paper itself. It is secondary to `paper/paper.tex` — if anything conflicts, the paper takes precedence.

---

## Why squared L2 norm for influence strength?

The influence strength $J_a(t,n \mid t') = \sum_\mu [\partial f_a / \partial x_{\mu,t'}]^2$ uses the squared Frobenius norm rather than the L1 norm or an unsquared L2 norm. The reason is additivity: $\sum_{t' \le t} J_a(t,n \mid t') = J_{a,\text{tot}}(t,n)$ holds exactly for the squared norm, which is necessary for the normalized $p(t')$ to be a well-defined probability distribution. With the L1 norm, this additive decomposition breaks down.

---

## Why evaluate at the last token position?

The influence entropy is computed at the last token position $t = L$ of each batch. This is because the causal attention mask gives position $t=L$ access to all preceding positions — any other choice would artificially exclude some tokens from the influence distribution. Computing at an earlier position would undercount nonlocality.

---

## Why average over active batches only?

A feature's entropy is averaged over batches where it is active ($f_a > 0.2$), not over all batches. In batches where the feature is inactive, the gradient $\partial f_a / \partial x_{\mu,t'}$ is numerically zero everywhere (the ReLU is below threshold), which would bias the entropy toward 0. Conditioning on active batches is the right thing to do for the same reason that one conditions on firing events when studying a neuron.

---

## Why threshold 0.2?

The activation threshold of 0.2 is a soft empirical choice. The SAE uses a ReLU, so strictly $f_a = 0$ for inactive features. In practice there are very small positive activations from numerical noise. A threshold of 0.2 was found to cleanly separate genuinely active features from near-zero activations in early experiments, while being small enough not to exclude lightly active features.

---

## The holographic analogy: what it does and doesn't claim

The AdS/CFT analogy is conceptual motivation, not a mathematical derivation. The paper does not claim that transformers implement holographic duality — only that the structure of SAE features (nonlocal, approximately independent, sparse) shares three formal properties with bulk fields in AdS/CFT. The analogy motivates the question and the choice of entropy as a measure, but the results stand independently of whether the analogy is deep or superficial.

The mapping is: input tokens ↔ boundary operators; SAE features ↔ bulk fields; feature layer depth ↔ radial coordinate; entropy ↔ nonlocality / operator size. The Ryu–Takayanagi formula would suggest an entanglement-entropy connection, but that direction was not pursued here.

---

## Why token vector comparison uses the decoder?

The token vector comparison (Section 3.3) uses the reconstructed token vector $\tilde{y} = D f$ rather than the raw residual stream $y$. The reason is that $\tilde{y}$ is directly expressible as a linear combination of the decoder columns weighted by feature activations, making the Jacobian decomposable into per-feature contributions. Using $y$ directly would conflate the SAE's influence with other components (the reconstruction error $y - \tilde{y}$). The comparison is therefore specifically between SAE features and the SAE's reconstruction, not between features and the full residual stream.

---

## Leading features: top 500 by activation frequency

"Leading features" are defined as those with activation frequency > 0.1% (at least 1 in 1000 tokens). At each layer, the top 500 by frequency are selected for detailed analysis. This cutoff is practical: features that fire very rarely do not accumulate enough statistics for a reliable entropy estimate across batches. The frequency threshold matters more than the exact cutoff of 500.

---

## Compute and memory notes

- The full Jacobian for token vector influence (`torch.autograd.functional.jacobian`) allocates a tensor of shape $[d_\text{model}, L, d_\text{model}] = [512, 64, 512]$ ≈ 16M floats per batch. On CPU this is slow; MPS handles it in ~2s/batch.
- Feature influence (`torch.autograd.grad`) is computed sequentially for each active feature, retaining the computation graph. For 500 features per batch this is the dominant cost (~0.5s/batch).
- Running all 6 layers in `compare_entropies_multi_layer.py` with `--num-batches 50` takes 2–4 hours on M-series hardware. Start with 1–2 layers and a small batch count (10–20) to validate the setup before running the full sweep.

---

## What didn't work / dead ends

- **Entropy as a training regularizer**: One natural follow-up is to train SAEs with entropy minimization as an additional regularizer. Brief experiments showed this was numerically unstable with the standard architecture — the entropy gradient interacts poorly with the L1 sparsity penalty. Not explored further.
- **Rényi entropy**: The paper uses Shannon entropy. Rényi entropy $H_\alpha$ with $\alpha > 1$ would downweight the tails of the influence distribution, giving a "core" nonlocality measure. The qualitative trends should be similar but the quantitative values would differ.
- **Layer 5 in entropy-vs-depth plots**: Layer 5 is partially excluded from Fig. 6 because the entropy comparison data (from `compare_entropies_multi_layer.py`) for layer 5 had fewer leading features than expected in some runs. Fig. 5 does include layer 5 via a separate data source (`feature_token_influence_resid_out_layer5.pt`).
