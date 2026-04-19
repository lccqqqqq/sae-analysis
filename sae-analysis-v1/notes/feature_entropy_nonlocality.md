# Feature entropy and nonlocality

This note pins down how "entropy of a feature" is defined in this repo and why it counts as a measure of nonlocality. It expands on `note.md` §"Influence of input tokens to a given feature" with the concrete normalization, the code pointers, and the interpretation of $2^H$ as an effective receptive-field size.

## What $f_a(t, z)$ is

$f_a(t, z)$ is the $a$-th SAE feature activation at token position $t$, read off from the residual stream at layer $z$. The repo loads a pretrained one-layer ReLU SAE from `dictionaries/pythia-70m-deduped/<site>/10_32768/ae.pt`, with encoder weight $E \in \mathbb{R}^{n_\text{latent}\times d_\text{model}}$ and bias $b_E \in \mathbb{R}^{n_\text{latent}}$, and defines

$$
f_a(t, z) \;=\; \mathrm{ReLU}\!\bigl(E\,y(t, z) + b_E\bigr)_a \;=\; \max\!\Bigl(0,\; \sum_{\mu=1}^{d_\text{model}} E_{a\mu}\, y_\mu(t, z) + (b_E)_a\Bigr),
$$

where $y(t, z)$ is the residual-stream vector at position $t$ produced by layer $z$ of Pythia-70m-deduped. This is literally `feature_token_influence.py:54-58` (`get_feature_activations`): `F.relu(torch.matmul(x, enc_w) + enc_b)`. The input $y$ is captured by a forward hook on `model.gpt_neox.layers[layer_idx]` in `process_batch_with_influence` (`feature_token_influence.py:130-139`), and the encoder matrix is loaded from the `encoder.weight` / `decoder.weight` entries of `ae.pt` via `get_sae_weights` (`feature_token_influence.py:26-52`).

Index ranges:

- $a \in \{0, \ldots, n_\text{latent} - 1\}$, with $n_\text{latent} = 32768$ for the `10_32768` pretrained dictionaries — 32k candidate features per site. The "leading features" filter in `feature_token_influence.py:304-322` cuts this down to the top $\le 500$ by frequency before any gradient work.
- $t \in \{0, \ldots, L-1\}$ where $L = 64$ is the batch window. In practice the entropy is only computed at $t = L-1$ (the last position), since in a causal transformer only the last position can depend on every upstream token.
- $z$ is a hook site, one of `resid_out_layer0`, …, `resid_out_layer5` for Pythia-70m-deduped. The integer layer index is parsed from the site string via `int(site.rsplit("layer", 1)[-1])` in `feature_token_influence.py:340`. Each layer has its own $E$, $b_E$, and $n_\text{latent}$, loaded from its own `ae.pt`.

Two features of this definition matter for the gradient/entropy step:

- **The chain that gets differentiated.** When `feat_activation.backward()` fires in `compute_influence_for_feature`, the derivative flows through `input_embeds → transformer layers 0..z → residual-stream output y(t, z) → E y + b_E → ReLU → f_a`. So $\partial f_a(t,z)/\partial x_{\mu, t'}$ in Step 1 below is a composition of the Pythia attention blocks of layers $0$ through $z$ with the linear encoder row $E_a$.
- **ReLU gating.** If $f_a(t, z) = 0$ on this batch, the gradient is identically zero and no entropy is defined for that $(a, t, z)$. That is why the pipeline filters with `last_pos_feats[feat_idx] > THRESHOLD` at `feature_token_influence.py:174` before calling `compute_influence_for_feature` — an inactive feature has a flat gradient and no meaningful $P(t'\mid t,z)$.

This is the standard Marks/Karvonen/Mueller ReLU SAE — not TopK or gated. The pretrained `10_32768` checkpoints are ReLU SAEs, which is why `get_feature_activations` hardcodes the ReLU form. Swapping in a TopK SAE would require applying the top-$k$ mask inside `get_feature_activations`, and the gradient structure in Step 1 would change correspondingly.

## Definition

The entropy is the Shannon entropy of a single probability distribution over upstream token positions, built from the gradient of the feature activation with respect to the input embeddings. Three steps.

### Step 1 — gradient-norm influence per upstream position

Fix a feature index $a$, a layer $z$ (the hook site such as `resid_out_layer3`), and a read-out position $t$ inside a batch (in practice always the last position of a 64-token window, since only the last position can depend on all the others in a causal transformer). Form the Jacobian of the scalar $f_a(t,z)$ with respect to every input-embedding vector at position $t'\le t$:

$$
I_{a\mu}(t, z \mid t') \;=\; \frac{\partial f_a(t, z)}{\partial x_{\mu, t'}},
$$

where $\mu$ indexes the $d_\text{model}$ components of the embedding at position $t'$. Collapse the embedding axis with a squared L2 norm to get one scalar per upstream position:

$$
J_a(t, z \mid t') \;=\; \bigl\|\,I_{a\mu}(t, z \mid t')\,\bigr\|_2^{2} \;=\; \sum_{\mu}\left(\frac{\partial f_a(t,z)}{\partial x_{\mu, t'}}\right)^{2}.
$$

In code this is `feature_token_influence.py:63-93` (`compute_influence_for_feature`). The implementation backpropagates a single scalar `feat_activation` through a `DummyEmbed` leaf tensor, reads `input_embeds.grad` of shape `[1, seq_len, d_model]`, and returns `(grads ** 2).sum(dim=-1)` — a vector of length `seq_len`, one scalar per upstream token.

The squared-L2 choice is deliberate. It makes the total

$$
J_{a,\text{tot}}(t, z) \;=\; \sum_{t'\le t} J_a(t, z \mid t')
$$

a genuine non-negative additive quantity, so the pieces behave like a mass distribution and Step 2 yields a proper probability.

### Step 2 — normalize to a probability distribution over $t'$

$$
P(t' \mid t, z) \;=\; \frac{J_a(t, z \mid t')}{J_{a,\text{tot}}(t, z)}.
$$

$P(t'\mid t,z)$ is the fraction of the feature's sensitivity that lives on upstream token $t'$. It is non-negative and sums to $1$ over $t'\le t$.

### Step 3 — Shannon entropy of that distribution

$$
H_a(t, z) \;=\; -\sum_{t'\le t} P(t' \mid t, z)\,\log_2 P(t' \mid t, z).
$$

In code this is literally `scipy.stats.entropy(P, base=2)` — see `plot_feature_entropy_histogram.py:36-52`, `plot_all_features_entropy_histogram.py:46-62`, `plot_entropy_vs_activation.py:27-43`, and `compare_entropies.py:57-73` (`compute_feature_entropy`). All of them add `eps = 1e-12` before normalizing to avoid $\log 0$. The unit is bits. Because $t$ is pinned to the last position of the batch, $H_a(t,z)$ is a single scalar per (feature, layer, batch).

## Why this number measures nonlocality

Read $P(t'\mid t,z)$ as "where does this feature's sensitivity live along the token axis". The entropy of that distribution is small when the sensitivity concentrates on a few positions, and large when it smears across many. Concretely, in a window of length $L$ (the pipeline uses $L=64$):

- **Fully local feature.** All the gradient mass sits on one upstream token, say $t' = t-3$. Then $P$ is a delta, $H_a = 0$ bits. The feature reads exactly one input token and nothing else.
- **Fully delocalized feature.** $P$ is uniform over all $L$ upstream positions. Then $H_a = \log_2 L = 6$ bits for $L=64$ — the maximum possible in the window. The feature draws equally from every upstream token.
- **Intermediate.** If $P$ is uniform over $k$ positions and zero on the rest, then $H_a = \log_2 k$, so

$$
\boxed{\; 2^{H_a} \;=\; k \;}
$$

is exactly the size of the feature's receptive field. For a non-uniform $P$, $2^{H_a}$ is the *effective* number of tokens — the perplexity of $P$, i.e. the size of a uniform distribution with the same entropy. That is the one-line interpretation: **$2^{H_a}$ is the effective size of the feature's boundary footprint along the token axis**. `note.md` writes the same statement with natural log ($e^H$); the scripts happen to use $\log_2$, so in bits the effective receptive-field size is $2^{H}$.

So the pipeline turns a vague question — "how many input tokens does this feature actually depend on?" — into a rigorous scalar: compute the gradient at the feature, square it, normalize over upstream positions, take entropy.

## Connection to holography

Under the dictionary in `note.md`:

- input tokens ↔ boundary CFT operators,
- SAE features ↔ bulk operators,
- the gradient influence $J_a(t,z\mid t')$ ↔ the boundary-smearing kernel of a bulk operator.

Features with $H_a$ near $0$ are like near-boundary bulk operators, reconstructed from a tiny HKLL kernel localized on a handful of boundary operators. Features with $H_a$ near $\log_2 L$ are like deep-bulk operators that need smearing over the whole boundary region. The sequence of questions in `note.md` §"Operator Size Distribution" — is there a hierarchy of operator sizes among features, and does it match a bulk depth? — becomes the empirical question of how $H_a$ is distributed across the feature index, and how it varies with layer $z$, activation frequency, and batch length.

## The feature-vs-token-vector conjecture

`compare_entropies.py` and `compare_entropies_multi_layer.py` run the same three-step construction twice on the same batches and layers:

1. For every active leading SAE feature, compute $H^\text{feat}_a$ as above.
2. For the raw residual-stream vector at the same $(t,z)$, compute an analogous quantity $H^\text{tok}$ using the decoder-composed influence

$$
F_{\nu\mu}(t, z \mid t') \;=\; D^{a}_{\nu}\,I_{a\mu}(t, z \mid t'), \qquad S_\nu(t, z \mid t') \;=\; \|F_{\nu\mu}(t, z \mid t')\|_2^{2},
$$

which is implemented in `token_vector_influence.py` (the `process_batch_with_token_influence` helper used by both comparison scripts).

The conjecture is $H^\text{tok} > \langle H^\text{feat}_a\rangle_a$ on average — i.e. the raw residual-stream direction has a broader boundary footprint, and the SAE provides a *lower-entropy decomposition* of that same direction. The array of downstream plots (`plot_entropy_vs_depth.py` across layers, `entropy_vs_batch_size.py` across window lengths, `plot_entropy_vs_activation.py` across feature frequency) test whether this gap is robust and how it evolves.

## Code map

| Step | File and lines |
|---|---|
| Backprop to get $J_a(t,z\mid t')$ | `feature_token_influence.py:63-93` (`compute_influence_for_feature`), called from `feature_token_influence.py:96-189` (`process_batch_with_influence`) |
| Same on the raw residual stream | `token_vector_influence.py` (`process_batch_with_token_influence`) |
| Normalize and take $H$ | `compare_entropies.py:57-73` (`compute_feature_entropy`, `normalize_influence`); identical helper duplicated in every `plot_*_entropy_*.py` script |
| Aggregate over batches and save | `feature_token_influence.py:415-452` writes `feature_token_influence_<site>.pt` with `mean_influence`, `std_influence`, raw `all_influences`, `num_samples` per feature |
| Feature-vs-token comparison per batch | `compare_entropies.py:299-380` (`compare_entropies_for_batch`), called from `main()` |

## Gotchas

- The read-out position $t$ is always the **last** position of the batch. If you want $H$ at a different position you have to rewrite the "active feature" selection in `process_batch_with_influence` (it currently filters on `feats[0, -1, :]`).
- $H$ is capped at $\log_2(\text{batch\_size})$. With the default `BATCH_SIZE = 64` in `feature_token_influence.py` and `compare_entropies.py`, that's 6 bits. Comparisons across different batch sizes must be rescaled — this is exactly what `entropy_vs_batch_size.py` studies, and why `plot_entropy_vs_batch_size_notebook.py` draws the $\log_2(n)$ reference line.
- `feature_token_influence.py` uses `THRESHOLD = 1.0` for "active at last position" but `compare_entropies.py` uses `THRESHOLD = 0.2`. Different thresholds select different feature subsets; the entropies themselves are threshold-independent conditional on a feature being included.
- The gradient is taken through a `DummyEmbed` that monkey-patches `model.gpt_neox.embed_in`. It is restored in a `finally` block, but this is not safe to parallelize across threads sharing the same model object.
