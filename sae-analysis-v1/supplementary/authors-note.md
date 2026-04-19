# Author's Note

*From Xiao-Liang Qi — what I want readers to know beyond the paper.*

---

## Background and motivation

I am a theoretical physicist, not an ML researcher. I came to this work knowing holographic duality (AdS/CFT) well and sparse autoencoders not at all. The paper is a first attempt to connect two very different fields — gravitational holography and mechanistic interpretability — and should be read with that in mind.

If you are an interpretability researcher reading this paper: the holographic analogy is not a claim that transformers implement AdS/CFT. It is a physicist's instinct that the structural properties of SAE features (nonlocal, approximately independent, sparse) look like the structural properties of bulk fields in a holographic theory, and that this resemblance might be worth making quantitative. The entropy measure is the natural tool for doing that from a physics perspective.

If you are a physicist reading this paper: SAEs are a decomposition of internal neural network activations into a sparse set of interpretable features, trained with an L1 sparsity penalty. The "features" are directions in activation space that correspond to recognizable concepts (names, syntax, semantics). The key prior work is Bricken et al. (2023) and Templeton et al. (2024); read those first if the SAE setup is unfamiliar.

---

## Physical intuition behind the entropy measure

Shannon entropy measures uncertainty — or equivalently, how spread-out a probability distribution is. When we normalize the influence strengths $J_a(t,n \mid t')$ to a probability distribution over token positions, the entropy tells us how many tokens the distribution is effectively spread over. A feature that depends strongly on one token has low entropy (the distribution is concentrated). A feature that draws equally from many tokens has high entropy (the distribution is spread out).

The quantity $2^{S_a}$ is the effective number of tokens — an exponential of the entropy — which is the physicist's natural way to count the "support" of a distribution. This is sometimes called the "number of microstates" in statistical mechanics. So when the paper says a feature has entropy 3 bits, it means it effectively depends on $2^3 = 8$ tokens.

This is why entropy, rather than, say, the maximum influence or the number of tokens above a threshold, is the right measure: entropy is the unique measure of spread that satisfies natural information-theoretic axioms (it is additive for independent systems, maximal for uniform distributions, and zero for point distributions).

---

## What this paper is not

This paper does not:
- Train new SAEs or modify the SAE architecture
- Claim that the holographic analogy is mathematically rigorous
- Provide a theory of *why* deeper layers have more nonlocal features (the result is empirical)
- Establish that low-entropy decomposition is a criterion for "good" SAEs (that remains an open question)

The paper establishes a measurement tool and applies it to one model. The interesting questions — whether this generalizes across models, whether it can guide SAE training, whether the holographic connection is deep — are open.
