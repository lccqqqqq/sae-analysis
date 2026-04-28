# Changelog

Tracked changes to this repository. Scope is the trusted analysis tree under `scripts/` (changes to `scripts-v1/` are not logged here — that tree is reference-only). Most recent first. Each entry states what changed, why, and any impact on previously generated artefacts.

---

## 2026-04-26 — Centralised corpus loading; switch from preprocessed to raw WikiText-2

### What changed

- **Added** `scripts/analysis/data_loader.py` — single entry point for natural-text corpora.
  - `load_wikitext_train_text()` — drop-in default. Returns the WikiText-2 *raw* train split (HuggingFace config `wikitext-2-raw-v1`) as a single concatenated string. Process-cached.
  - `load_hf_text(dataset, config, split)` — general-purpose loader for any HuggingFace text dataset whose rows expose a `"text"` field. Same cache.
- **Added** `scripts/analysis/test_data_loader.py` — six unit tests covering: non-empty load, no `<unk>` contamination, case preserved, in-process cache identity, generic-loader cache sharing, distinct-split keying. All pass against the cached HuggingFace dataset.
- **Modified** the eight analysis scripts under `scripts/analysis/` that previously opened `wikitext-2-train.txt` directly. They now `from data_loader import load_wikitext_train_text` and call `text = load_wikitext_train_text()` in place of the previous five-line `Path("wikitext-2-train.txt") / open(...) / read()` block.
- **Removed** the silent dummy-text fallback from `feature_sparsity.py` and `feature_location_analysis.py`. If the corpus cannot be loaded the scripts now raise rather than running on 50 copies of "Quantum mechanics is a fundamental theory…".
- The local `wikitext-2-train.txt` file (10.3 MB) is **no longer read** by any analysis script. It is kept on disk for now to avoid disturbing collaborator workflows; it can be deleted in a follow-up commit once everyone has rebased.

### Files modified (8)

- `scripts/analysis/feature_sparsity.py`
- `scripts/analysis/compute_correlations.py`
- `scripts/analysis/feature_location_analysis.py`
- `scripts/analysis/feature_token_influence.py`
- `scripts/analysis/token_vector_influence.py`
- `scripts/analysis/compare_entropies.py`
- `scripts/analysis/compare_entropies_multi_layer.py`
- `scripts/analysis/entropy_vs_batch_size.py`

### Files added (3)

- `scripts/analysis/data_loader.py`
- `scripts/analysis/test_data_loader.py`
- `CHANGELOG.md` (this file, at repo root)

### Why — contaminated corpus in previous runs

The local `wikitext-2-train.txt` shipped in this repo is the **preprocessed** variant of WikiText-2 (Merity et al. 2016, the original word-level distribution). It contains two known-suboptimal artefacts when used as inference text for a modern subword-tokenized causal LM such as Pythia-70m:

1. **`<unk>` substitution.** 14,334 occurrences (~0.7% of all tokens) of the literal four-character string `<unk>`, used in the original distribution as the unknown-word placeholder for a closed vocabulary. Pythia's BPE tokenizer does **not** treat `<unk>` as a special unknown-token marker. The string is BPE-segmented into raw characters (`<`, `un`, `k`, `>`) which never co-occur in this pattern in Pythia's pretraining corpus (the Pile). Activations and gradients computed at these positions are off-distribution.
2. **`@-@` / `@.@` / `@,@` markers and space-separated punctuation.** Hyphens, decimal points, and commas inside numbers are encoded as `@-@`, `@.@`, `@,@` respectively, and most punctuation is separated by spaces from neighbouring words. Less harmful than (1) — the `@` glyph does occur in natural text — but still distorts both tokenization and natural-text statistics.

Switching to the HuggingFace `wikitext-2-raw-v1` config fixes (1) entirely and also restores the original case of proper nouns (the preprocessed variant is fully lowercased). Verified empirically on the HuggingFace cached dataset:

- `wikitext-2-raw-v1` train: 10.89 M characters, 0 occurrences of `<unk>`, 16,906 of `@-@`, mixed-case proper nouns preserved (e.g. "Valkyria", "Sega", "PlayStation").

It does **not** fix (2). The `@-@` and space-separated punctuation are inherent to both WikiText-2 distributions on HuggingFace. Removing them would require porting to a different corpus (`wikipedia`, `the_pile`, or scraping Wikipedia directly), which is out of scope for this change.

### Impact on existing `.pt` outputs

Every `.pt` artefact produced by the analysis pipeline before this change was generated against the contaminated corpus. They are mildly biased — not catastrophically so (the affected fraction of tokens is ~0.7% from `<unk>` and ~0.8% from the `@-@`/`@.@`/`@,@` markers, all of which are still present post-change for the second category) — but the bias is **concentrated** at positions where rare words used to live (proper nouns, foreign words, technical terminology). Specifically:

- `feature_sparsity_data_<site>.pt` — per-feature activation counts on positions whose token id is one of the BPE pieces of `<unk>` are spuriously inflated. Some "leading features" reported in earlier outputs may turn out to be detectors firing on `<`/`k`/`>` rather than on a meaningful semantic concept.
- `feature_token_influence_<site>.pt`, `token_vector_influence_<site>.pt`, `entropy_comparison_<site>_<timestamp>.pt`, `entropy_vs_batch_size_<site>_<timestamp>.pt` — the gradient and entropy quantities computed in batches that include `<unk>` positions are mildly biased; the residual stream at those positions is genuinely off-distribution for a model that never saw `<unk>` during pretraining.
- `correlation_matrix_<site>.pt` — `<unk>` is a single token id from the model's perspective only after BPE, but in practice the BPE pieces (`<`, `k`, `>`) appear so frequently in the contaminated corpus that they may show up in the per-feature correlation matrix as anomalously large entries.

**Recommendation.** Rerun the affected analyses on the raw corpus before drawing conclusions from any pre/post comparison. The order of dependencies is: `feature_sparsity.py` first, then everything else (per the README's dependency order).

### New runtime dependency

The analysis scripts now require the HuggingFace `datasets` package at import time of any module under `scripts/analysis/` (transitively, via `data_loader.py`). The package is already installed in the project's environment (verified `datasets==4.4.0`). On first call, `load_wikitext_train_text()` downloads the corpus to the standard HuggingFace cache (`~/.cache/huggingface/datasets/`); subsequent calls in any process hit the on-disk cache. All later calls within the same Python process hit an in-memory cache and return the same string object (verified by the `test_in_process_cache_returns_same_object` test).

### Test verification

All six tests in `scripts/analysis/test_data_loader.py` pass:

```
$ python scripts/analysis/test_data_loader.py
Running data_loader tests...
[INFO] Loading wikitext/wikitext-2-raw-v1[train] from HuggingFace...
  ok  test_loads_nonempty_string
  ok  test_no_unk_contamination
  ok  test_case_preserved
  ok  test_in_process_cache_returns_same_object
  ok  test_generic_loader_shares_cache
[INFO] Loading wikitext/wikitext-2-raw-v1[test] from HuggingFace...
  ok  test_different_split_is_distinct
All tests passed.
```

All eight modified scripts import cleanly and pass `python -m py_compile`.
