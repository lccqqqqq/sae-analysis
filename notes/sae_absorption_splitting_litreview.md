# SAE Feature Absorption & Splitting — Literature Overview

**Caveat:** arXiv IDs drawn from training memory (cutoff May 2025); spot-check before citing. Web tools were denied during the search pass.

## 1. Feature absorption — Chanin et al. (2024)

**"A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"**, Chanin, Wilken-Smith, Dulka, Bhatnagar, Bloom. arXiv:2409.14507.

**Phenomenon.** When an SAE has both a *general* feature (e.g. "starts with S") and a *specific* feature ("snake"), the L1/L0 sparsity penalty pushes the SAE to silence the general feature on inputs where the specific one fires and fold the "starts with S" direction into the specific feature's decoder vector. The general feature ends up with mysterious holes — it fires on most S-words but not on ones with dedicated features.

**Measurement.** On Gemma Scope SAEs over Gemma-2-2B, they train a linear probe for first-letter classification, locate the putative "letter X" feature, and for tokens where the probe is confident but that feature is silent, check whether the active features' decoder vectors have a large component along the probe direction. Reported as an **absorption rate** per letter / per SAE.

**Headline result.** Absorption is pervasive and *grows with SAE width* — more splitting produces more absorbers. TopK and JumpReLU variants both exhibit it. Absorption is a direct consequence of the sparsity penalty rewarding merging co-occurring directions.

## 2. Splitting vs absorption vs hedging

- **Feature splitting (Bricken et al. 2023, "Towards Monosemanticity", Anthropic).** As dictionary width grows, a coarse feature subdivides into many fine ones. Splitting itself is not pathological — it's cluster refinement.
- **Absorption.** Failure mode *caused by* splitting + L1: the coarse feature is eliminated on examples where fine features fire, its direction silently folded into them.
- **Feature hedging — Chanin et al. (2025), arXiv:2505.10809 (ID uncertain).** Dual mechanism operating even before splitting: when two features correlate, a narrow SAE mixes one decoder partially toward the other and the encoder under-activates to compensate. Explains why narrow SAEs *look* cleaner than they are.

## 3. Mitigation efforts

| Method | Paper | Helps absorption? |
|---|---|---|
| **Matryoshka SAE** | Bussmann, Leask, Nanda 2025, arXiv:2503.17547 | **Yes, directly** |
| Gated SAE | Rajamanoharan et al. 2024, arXiv:2404.16014 | Fixes shrinkage, not absorption |
| JumpReLU | Rajamanoharan et al. 2024, arXiv:2407.14435 | Mild — absorption persists |
| TopK / BatchTopK | Gao et al. 2024, arXiv:2406.04093 | Partial — similar pressure via k |
| End-to-end SAE | Braun et al. 2024, arXiv:2405.12241 | Different failure modes; not targeted |
| Transcoders / cross-layer | Dunefsky+ 2024 arXiv:2406.11944; Lindsey+ 2025 arXiv:2503.08200 | Sidesteps single-site reconstruction |

## 4. Transcoders (brief)

A **transcoder** is a sparse autoencoder whose input is an MLP's *input* and whose output reconstructs the MLP's *output*:

$$\hat{y} = W_{dec}\,\sigma(W_{enc}x + b_{enc}) + b_{dec},\quad\text{trained on pairs }(x_{\text{MLP-in}}, y_{\text{MLP-out}}).$$

It's a sparse, interpretable surrogate for the MLP sublayer, not a decomposition of a single activation site. Advantage: input-side and output-side features are jointly defined, enabling circuit tracing *through* the MLP. **Skip transcoders** add a learned linear shortcut for fidelity; **cross-layer transcoders** generalize to many MLPs jointly (Anthropic 2025 circuit tracing of Claude, arXiv:2503.08200).

## 5. Matryoshka SAEs (brief)

Partition the dictionary into nested prefixes $d_1 < d_2 < \dots < d_K$ and train with reconstruction loss summed over every prefix:

$$\mathcal{L} = \sum_k \bigl\|x - W_{dec}^{(1:d_k)} f^{(1:d_k)}(x)\bigr\|^2 + \lambda \sum_k \|f^{(1:d_k)}\|_1.$$

Every prefix must independently reconstruct, so the coarsest shell *must* encode general features — it has nowhere else to hide reconstruction mass. On the Chanin absorption benchmark the "first letter X" feature survives in the coarse shell and fires on all X-tokens.

**Relevance to the holographic picture:** the nested shell structure is a natural candidate for a bulk **radial coordinate** — coarse shells = deeper / more universal, fine shells = near-boundary / token-specific. Worth asking whether influence-entropy $S$ separates cleanly by shell.

---

## References (verify before citing)

- Chanin, Wilken-Smith, Dulka, Bhatnagar, Bloom, "A is for Absorption", arXiv:2409.14507 (2024)
- Bricken et al., "Towards Monosemanticity", transformer-circuits.pub (Anthropic, 2023)
- Chanin et al., "Feature Hedging" (2025) — arXiv ID uncertain (2505.10809 or 2505.11756)
- Bussmann, Leask, Nanda, "Matryoshka SAEs", arXiv:2503.17547 (2025)
- Rajamanoharan et al., "Improving Dictionary Learning with Gated SAEs", arXiv:2404.16014 (2024)
- Rajamanoharan et al., "JumpReLU SAEs", arXiv:2407.14435 (2024)
- Gao et al. (OpenAI), "Scaling and Evaluating Sparse Autoencoders" (TopK), arXiv:2406.04093 (2024)
- Braun et al. (Apollo), "Identifying Functionally Important Features with End-to-End SAEs", arXiv:2405.12241 (2024)
- Dunefsky, Chlenski, Nanda, "Transcoders Find Interpretable LLM Feature Circuits", arXiv:2406.11944 (2024)
- Lindsey, Ameisen et al. (Anthropic), "Circuit Tracing / Cross-Layer Transcoders", arXiv:2503.08200 (2025)
