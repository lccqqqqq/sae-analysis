# On the "low-entropy = good feature" hypothesis

**Question:** Is a good SAE feature one whose influence distribution is sharply localized in input-token space — i.e., a feature whose entropy $S_a$ is low, so that only one or two tokens trigger it?

**Short answer:** Partially, but not as a global principle. The holographic analogy actually argues against making low entropy the defining property of a good feature. Low entropy picks out *surface-level* features; the semantically rich features are expected to be more nonlocal, not less. A better criterion is **feature-specific structure** rather than sheer locality.

The rest of this note walks through why.

---

## 1. The hypothesis, precisely

Let $f_a(t, n)$ be the activation of SAE feature $a$ at token position $t$, layer $n$. Its input-dependency distribution is

$$P_a(t' \mid t, n) = \frac{\|\partial f_a / \partial x_{\cdot, t'}\|_2^2}{\sum_{t'' \le t} \|\partial f_a / \partial x_{\cdot, t''}\|_2^2}, \qquad S_a = -\sum_{t'} P_a(t') \log_2 P_a(t').$$

The hypothesis under consideration is:

> **(H):** A "good" SAE feature has low $S_a$. Ideally $S_a \lesssim 1$ bit, meaning the feature fires based on one or two specific input tokens.

This is the natural extrapolation from monosemanticity intuitions: "Golden Gate" feature fires when "Golden Gate" appears; "hot days" feature fires on heat-related words; etc. The feature is interpretable precisely because its trigger is narrow.

---

## 2. Translating to the holographic picture

In AdS/CFT:

- Boundary CFT operators $\mathcal{O}(x)$ ↔ input tokens $x_{t'}$.
- Bulk fields $\phi(X, z)$ at radial depth $z$ ↔ SAE features $f_a$ at layer $n$.
- HKLL smearing: $\phi(X, z) = \int dy \, K(X, z; y) \, \mathcal{O}(y)$, with the **smearing kernel** $K$ encoding how many boundary operators are needed to reconstruct the bulk field.

The support of $K$ on the boundary is the bulk field's *operator size*. In the SAE language, $P_a(t')$ is the magnitude-squared of the effective smearing kernel (we're seeing $|K|^2$, not $K$ itself, because we took a squared-gradient norm).

Entropy $S_a$ is a good proxy for the log of the effective support of $K$. So:

> low $S_a$ $\Leftrightarrow$ narrow smearing kernel $\Leftrightarrow$ bulk field reconstructible from a small boundary region.

---

## 3. Where the hypothesis succeeds

Holography does give us one strong, correct statement: **near-boundary bulk fields have narrow smearing kernels**. A field at radial depth $z \to 0$ is essentially a local boundary operator, reconstructible from a single point on the boundary. In our analogy, a feature at layer 0 or layer 1 should indeed be approximately local in input-token space — its receptive field on the input is small because not much attention-mixing has happened yet.

This matches the paper's empirical finding: at layer 0, mean $S_a \approx 0.9$ bits — features fire based on essentially a single input token. At layer 0, **(H) holds**. Low-entropy features at shallow layers are "good" in the sense of being cleanly identifiable with specific tokens.

So the hypothesis is **locally true** for the surface region of the network.

---

## 4. Where the hypothesis breaks down

Now push deeper. In AdS/CFT, as you move into the bulk:

- The smearing kernel $K$ gets **wider** (larger causal wedge / entanglement wedge).
- Operator size grows.
- Bulk fields become **more** nonlocal in the boundary, not less.

And this is exactly what makes them interesting. A deep bulk field is a *genuinely emergent* object — you cannot identify it with any single boundary operator. It encodes correlations across many boundary degrees of freedom. It's the place where the dual picture has *content* beyond trivial restatement of the boundary theory.

Translating back: **deep-layer features are expected to be nonlocal, and this is not a defect.** A feature corresponding to, say, "this paragraph is about physics" or "the subject of the sentence" *must* pool information from many input tokens. If such a feature had $S_a \approx 0$, it would be a near-surface feature in disguise — not actually encoding an abstract concept.

The paper's empirical finding confirms this: at layer 5, $S_a \approx 4$ bits, $2^{S_a} \approx 16$ effective tokens per feature. Those features are not local, and they *shouldn't* be. They are the deep-bulk analogues.

So **(H) fails as a universal criterion**. Applied to deep layers, it would reject exactly the features that carry the most interpretive value. In holographic language: insisting on low entropy is insisting on staying near the boundary, where the physics is trivially the boundary physics.

---

## 5. The homogeneity trap

There is a *further* reason to be cautious about low entropy — the shared-Jacobian effect visible in the new heatmap plots.

At deep layers, many active features' $P_a(t')$ distributions look nearly identical across features. This is because

$$\frac{\partial f_a}{\partial x_{\mu, t'}} = W^\text{enc}_a \cdot \frac{\partial y(t, n)}{\partial x_{\mu, t'}}$$

and the second factor — the residual-stream Jacobian — is **shared across all features at a given layer**. If the text has a few salient tokens that dominate this shared Jacobian, then *every* feature's $P_a(t')$ inherits the same peaks, regardless of what the feature represents.

In holographic terms, this is a statement about the **state**, not the **operator**: if the boundary state has most of its "information" concentrated on certain operators $\mathcal{O}(y_i)$, then every bulk field sampled in that state will show apparent localization near those $y_i$. This doesn't mean the bulk fields are themselves local.

Consequence: **low $S_a$ on a specific batch may reflect the input's structure, not the feature's structure**. A feature that is "sharp" only because the text happens to be sharp is not distinguishing itself from the crowd.

---

## 6. What a better criterion looks like

The holographic picture suggests the right notion of "good feature" is not low entropy but **distinctive structure**. Specifically:

1. **Feature-specific nonlocality pattern.** A good feature has a $P_a(t')$ that differs from the token-vector distribution $Q(t')$ — the feature is picking out its own set of input tokens, not riding the shared bottleneck. Measured by $D_\text{KL}(P_a \,\|\, Q)$; large = distinctive.

2. **Characteristic nonlocality scale.** Across batches the feature should have a stable $S_a$ — the specific value matters less than its consistency. A feature with wildly varying $S_a$ is being dragged around by context rather than encoding a robust concept.

3. **Monosemantic in vocabulary.** The feature fires on a narrow vocabulary of triggering tokens (low entropy in token-count space, from `feature_sparsity_data_*.pt`). This is the classical Bricken/Anthropic monosemanticity criterion and is independent of the gradient analysis.

4. **Sharp decoder projection.** Projecting $D_a$ through the unembedding matrix and examining the induced vocabulary distribution. A good feature's decoder picks out a specific output semantic direction.

**A composite filter** — low $S_a$ AND high $D_\text{KL}(P_a \| Q)$ AND low vocab entropy — selects features that are sharp in multiple senses simultaneously. This is much more discriminating than any single criterion.

In holographic language, what we want is an operator that is both well-localized on the boundary *relative to the state it is evaluated in* (high KL against the state-induced background) and has a consistent action across states (stable across batches).

---

## 7. Bottom line

| Criterion | Holographic meaning | Good feature? |
| --------- | ------------------- | ------------- |
| Low $S_a$ | Narrow smearing kernel | Only for surface-depth features |
| Low $S_a$ at deep layer | Near-boundary in disguise, or text-saliency artifact | Often a red flag, not a good sign |
| High $D_\text{KL}(P_a \,\|\, Q)$ | Feature-specific boundary footprint | Strong positive signal at any depth |
| Low vocab entropy | Narrow state-support on boundary | Classical monosemanticity, usually good |
| Stable $S_a$ across batches | Intrinsic operator size | Good: "this has a real scale" |

So the hypothesis (H) captures one important axis but isn't the whole story. It correctly identifies surface-level features and correctly predicts the shallow-layer finding. It **fails as a general definition** because it rejects exactly the features that realize the nontrivial bulk — the deep, emergent, semantically rich ones. And it **fails as a single-batch filter** because apparent sharpness on one text is often a state effect, not a feature effect.

The recommended path: use low $S_a$ as a *shallow-layer* interpretability filter, but at intermediate and deep layers combine it with distinctiveness ($D_\text{KL}$) and monosemanticity (vocab entropy) criteria. "Good" is multi-dimensional; no single projection captures it.
