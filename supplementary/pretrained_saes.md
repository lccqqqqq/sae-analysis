# Pretrained SAE Catalog

Publicly available Sparse Autoencoder weights paired with open-source language models. Curated for extending the nonlocality entropy analysis to models beyond Pythia-70m.

Last updated: 2026-04-14.

**Key considerations for this project:**
- The analysis requires SAEs covering **all or most layers** (for the entropy-vs-depth sweep). Single-layer SAEs are noted but less useful.
- SAE architecture matters: the current codebase uses a vanilla ReLU SAE (`f = ReLU(Wx + b)`). TopK and JumpReLU SAEs have different activation functions and may need code changes.
- The loading format must be adapted: this repo uses `dictionary_learning`'s `ae.pt` format; SAELens and EleutherAI `sparsify` use different formats.

---

## Recommended starting points

These have the best combination of layer coverage, documentation, and accessibility:

| Model | SAE source | Layers | Dict size | Architecture | Loading |
| ----- | ---------- | ------ | --------- | ------------ | ------- |
| GPT-2 Small (124M) | OpenAI via jbloom | all 12 | 32k, 128k | ReLU | SAELens |
| Gemma 2 2B | Google Gemma Scope | all 26 | 16k--1M | JumpReLU | SAELens |
| Gemma 2 9B | Google Gemma Scope | all 42 | 16k--1M | JumpReLU | SAELens |
| Qwen2-0.5B | chanind | all 24 | unknown | SAELens | SAELens |
| Llama-3.2-1B | EleutherAI | unknown | 131k | TopK | sparsify |
| Llama-3.1-8B | EleutherAI | multiple | 131k, 262k | TopK/MultiTopK | sparsify |
| Pythia-70m | saprmarks | all 6 | 32k | ReLU | dictionary_learning |
| Pythia-160m | EleutherAI | unknown | 768--131k | TopK | sparsify |

---

## GPT-2 family

### GPT-2 Small (124M) -- best covered

**OpenAI v5 SAEs (via jbloom/SAELens)** -- gold standard

| HuggingFace ID | Site | Dict size |
| -------------- | ---- | --------- |
| `jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs` | residual stream post | 32,768 |
| `jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs` | residual stream post | 131,072 |
| `jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs` | residual stream mid | 32,768 |
| `jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs` | residual stream mid | 131,072 |
| `jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs` | MLP output | 32,768 |
| `jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs` | MLP output | 131,072 |
| `jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs` | attention output | 32,768 |
| `jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs` | attention output | 131,072 |

- **Base model:** `openai-community/gpt2` (12 layers, d_model=768)
- **Architecture:** ReLU SAE
- **Layers:** All 12 (blocks.0 through blocks.11)
- **Validated:** OpenAI paper "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2024)
- **Loading:** `from sae_lens import SAE; sae, _, _ = SAE.from_pretrained("gpt2-small-resid-post-v5-128k", "blocks.11.hook_resid_post")`
- **Model internals:** embedding = `model.transformer.wte`, layers = `model.transformer.h[i]`

**Other GPT-2 Small SAEs:**

- `jbloom/GPT2-Small-SAEs` -- jbloom's earlier training, 32x expansion (~24,576 features), residual stream, all 12 layers. ReLU.
- `apollo-research/e2e-saes-gpt2` -- end-to-end trained SAEs (loss includes downstream model performance). Different paradigm.
- `jacobcd52/gpt2-small-sparse-autoencoders` -- all 12 layers, mlp_out and resid_pre sites, ~4096 dict size. Trained on 1B tokens from OpenWebText.

### GPT-2 Medium (355M)

**No public SAE weights found.**

### GPT-2 Large (774M) / GPT-2 XL (1.5B)

Only fragmentary, undocumented releases exist (single layers, no validation). Not recommended.

---

## Llama family

### Llama-3.1-8B -- best Llama coverage

**EleutherAI SAEs** (via `sparsify` library)

| HuggingFace ID | Architecture | Sites | Dict size |
| -------------- | ------------ | ----- | --------- |
| `EleutherAI/sae-llama-3.1-8b-32x` | TopK | residual stream + MLP | 131,072 |
| `EleutherAI/sae-llama-3.1-8b-64x` | MultiTopK | MLP | 262,144 |

- **Base model:** `meta-llama/Meta-Llama-3.1-8B` (32 layers, d_model=4096)
- **Training data:** RedPajama v2 (~8.5B tokens)
- **Loading:** `from sparsify import Sae; sae = Sae.load_from_hub("EleutherAI/sae-llama-3.1-8b-32x", hookpoint="layers.23.mlp")`
- **Model internals:** embedding = `model.model.embed_tokens`, layers = `model.model.layers[i]`
- **Memory note:** Jacobian for d_model=4096 is 64x larger than Pythia-70m. Reduce batch size or use H200.

### Llama-3-8B

| HuggingFace ID | Architecture | Dict size | Notes |
| -------------- | ------------ | --------- | ----- |
| `EleutherAI/sae-llama-3-8b-32x` | TopK | 131,072 | Multiple layers |
| `EleutherAI/sae-llama-3-8b-32x-v2` | TopK | 131,072 | Layer 24 complete; others partial |

- **Base model:** `meta-llama/Meta-Llama-3-8B` (32 layers, d_model=4096)

### Llama-3.2-1B

| HuggingFace ID | Architecture | Dict size | Layers | Notes |
| -------------- | ------------ | --------- | ------ | ----- |
| `EleutherAI/sae-Llama-3.2-1B-131k` | TopK | 131,072 | not documented | EleutherAI program |
| `chanind/sae-llama-3.2-1b-res` | SAELens | not documented | L0--L8 (9 layers) | SAELens format, easiest integration |

- **Base model:** `meta-llama/Llama-3.2-1B` (16 layers, d_model=2048)
- **Good candidate for scaling test:** smaller than 8B but same architecture family as Llama-3.1-8B.

### Llama-3.1-8B-Instruct

| HuggingFace ID | Layers | Notes |
| -------------- | ------ | ----- |
| `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` | Layer 19 only | L0=91, trained on LMSYS-Chat-1M |

### Llama-2-7B

| HuggingFace ID | Notes |
| -------------- | ----- |
| `yuzhaouoe/Llama2-7b-SAE` | Associated with NAACL 2025 paper. Sparse documentation. |

---

## Qwen family

### Qwen2-0.5B -- best Qwen coverage

| HuggingFace ID | Architecture | Sites | Layers |
| -------------- | ------------ | ----- | ------ |
| `chanind/sae-qwen2-0.5b-res` | SAELens (likely JumpReLU) | residual stream post | All 24 layers |

- **Base model:** `Qwen/Qwen2-0.5B` (24 layers, d_model=896)
- **Loading:** SAELens `SAE.from_pretrained`
- **Model internals:** embedding = `model.model.embed_tokens`, layers = `model.model.layers[i]` (same as Llama)
- **Best starting point for Qwen:** full-layer coverage, SAELens format.

### Qwen2.5-1.5B

| HuggingFace ID | Dict size | Training data |
| -------------- | --------- | ------------- |
| `Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k` | 65,536 | unknown |
| `huypn16/sae-qwen-2.5-1.5B-OWM-16x` | 16x expansion | OpenWebMath |
| `huypn16/sae-qwen-2.5-1.5B-OMS-16x` | 16x expansion | OpenMathStack (?) |

- **Base model:** `Qwen/Qwen2.5-1.5B` (28 layers, d_model=1536)
- **Note:** Minimal documentation on all. Layer coverage unknown.

### Qwen2.5-7B

| HuggingFace ID | Architecture | Sites | Layers | Dict size |
| -------------- | ------------ | ----- | ------ | --------- |
| `pozoviy/sae_Qwen_Qwen2.5-7B_resid_post_{5,15,25}_*` | BatchTopK | resid post | L5, L15, L25 | 16,384 |
| `nikoryagin/sae_Qwen_Qwen2.5-7B_*` (hundreds) | BatchTopK | resid pre+post | L24, L25 | 16,384 |

- **Base model:** `Qwen/Qwen2.5-7B` (28 layers, d_model=3584)
- **Note:** Massive experiment sweeps by nikoryagin/pozoviy. Layers 5, 15, 24, 25 have best coverage. Not all layers are covered.

### DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 architecture)

| HuggingFace ID | Architecture | Sites | Dict size |
| -------------- | ------------ | ----- | --------- |
| `EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k` | TopK | MLP outputs | 65,536 |

- **Base model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (Qwen2-1.5B architecture, distilled from DeepSeek-R1)
- **Interesting for comparison:** math-reasoning model on Qwen architecture.

### Qwen2.5-Math / Qwen2.5-Coder

**No general-purpose SAEs found for Qwen2.5-Math.**

One SAE for Qwen2.5-Coder: `huypn16/sae-Qwen2.5-Coder-7B-Instruct-codeforces-8x` (trained on Codeforces data, 8x expansion).

---

## Gemma family (bonus -- best SAE ecosystem)

### Google Gemma Scope -- most comprehensive public SAE release

| HuggingFace ID | Base model | Site | Layers | Dict widths |
| -------------- | ---------- | ---- | ------ | ----------- |
| `google/gemma-scope-2b-pt-res` | Gemma 2 2B | residual stream | all 26 | 16k--1M |
| `google/gemma-scope-2b-pt-mlp` | Gemma 2 2B | MLP output | all 26 | 16k--1M |
| `google/gemma-scope-2b-pt-att` | Gemma 2 2B | attention output | all 26 | 16k--1M |
| `google/gemma-scope-9b-pt-res` | Gemma 2 9B | residual stream | all 42 | 16k--1M |
| `google/gemma-scope-9b-pt-mlp` | Gemma 2 9B | MLP output | all 42 | 16k+ |
| `google/gemma-scope-9b-pt-att` | Gemma 2 9B | attention output | all 42 | 16k+ |
| `google/gemma-scope-9b-it-res` | Gemma 2 9B IT | residual stream | all 42 | various |
| `google/gemma-scope-27b-pt-res` | Gemma 2 27B | residual stream | selected | 131k |

- **Architecture:** JumpReLU
- **Paper:** arXiv:2408.05147 ("Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2")
- **License:** CC-BY-4.0
- **Loading:** `from sae_lens import SAE; sae, _, _ = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_0/width_16k/canonical")`
- **Model internals:** embedding = `model.model.embed_tokens`, layers = `model.model.layers[i]`
- **Note:** By far the most thorough SAE release. If you want to test the entropy-vs-depth analysis on a well-characterized larger model, Gemma 2 2B or 9B with Gemma Scope is the strongest option.

---

## Pythia family (for reference)

| HuggingFace ID | Base model | Architecture | Dict size | Layers |
| -------------- | ---------- | ------------ | --------- | ------ |
| `saprmarks/pythia-70m-deduped-saes` | Pythia-70m-deduped | ReLU | 32,768 | all 6 |
| `EleutherAI/Pythia-160m-SAE-k{K}-{D}` | Pythia-160m | TopK | 768--131k | unknown |
| `tim-lawson/sae-pythia-{size}-deduped-x64-k32-*` | 70m, 160m, 410m | TopK (MLSAE) | 64x | various |

- Pythia-160m EleutherAI SAEs come in many (k, dict_size) configurations. No model cards published.
- tim-lawson MLSAE bundles transformer + SAE weights in PyTorch Lightning format. See arXiv:2409.04185.
- **No well-documented SAEs found for Pythia-1.4B.**

---

## Loading interfaces

The three main SAE loading libraries and their formats:

| Library | Install | Loading | Used by |
| ------- | ------- | ------- | ------- |
| `dictionary_learning` | vendored in this repo | `torch.load("ae.pt")` | saprmarks Pythia SAEs |
| SAELens | `pip install sae-lens` | `SAE.from_pretrained(release, hook)` | OpenAI GPT-2, Gemma Scope, chanind, jbloom |
| sparsify (EleutherAI) | `pip install sparsify` | `Sae.load_from_hub(repo, hookpoint)` | EleutherAI Llama, Pythia-160m |

To use SAELens or sparsify SAEs with this repo's analysis pipeline, you need to:
1. Load the SAE weights using the appropriate library
2. Extract encoder weights (`W_enc`, `b_enc`) and decoder weights (`W_dec`) into the format expected by `get_sae_weights()`
3. Adapt the activation function if not ReLU (TopK selects top-k activations instead of thresholding; JumpReLU uses a learnable threshold)

---

## Architecture paths by model family

Required code changes in `feature_token_influence.py` and siblings:

| Model family | Embedding layer | Transformer layers | Num layers | d_model |
| ------------ | --------------- | ------------------ | ---------- | ------- |
| Pythia (GPT-NeoX) | `model.gpt_neox.embed_in` | `model.gpt_neox.layers[i]` | 6 | 512 |
| GPT-2 | `model.transformer.wte` | `model.transformer.h[i]` | 12 | 768 |
| Llama / Qwen / Gemma | `model.model.embed_tokens` | `model.model.layers[i]` | varies | varies |
