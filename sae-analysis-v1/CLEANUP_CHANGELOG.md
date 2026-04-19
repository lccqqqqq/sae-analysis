# sae-analysis cleanup changelog — 2026-04-19

## Summary

This document records the file-by-file justification for the one-shot
repository simplification performed on the `cleanup` branch. The cleanup
collapses a two-tree structure (flat legacy top-level + `sae-analysis-v1/`
subdir with reorganized `scripts/` and the paper draft) into a single
clean root, removes a vendored SAE-training library that the analysis
pipeline does not import, and parks exploratory-but-maybe-useful code
in `deprecated/` rather than deleting it outright.

- **Pre-cleanup snapshot branch:** `origin/pre-cleanup-snapshot` at `a49db4d` (full working-tree state before any deletion).
- **Local tarball backup:** `/mnt/users/clin/sae-analysis-backup-20260416.tar.gz` (includes SAE weights and experiment `.pt` files that are gitignored).
- **Files deleted:** 81 (≈ 72.7 MB).
- **Files moved to `deprecated/`:** 15 (≈ 609.7 KB).

## Deletions

### `sae-analysis-v1/dictionary_learning`
- **Size:** 201.5 KB
- **Original functionality** (inferred): (path is a directory or missing)
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `VENDORED` — Vendored third-party library not imported by the pipeline.
- **Replacement:** `use torch.load on the pretrained ae.pt files instead`

### `dictionary_learning`
- **Size:** 201.5 KB
- **Original functionality** (inferred): (path is a directory or missing)
- **Provenance:** 0ff8888 2025-02-09 feat: pypi packaging and auto-release with semantic release
- **Reason:** `VENDORED` — Vendored third-party library not imported by the pipeline.
- **Replacement:** `use torch.load on the pretrained ae.pt files instead`

### `sae-analysis-v1/tests/test_end_to_end.py`
- **Size:** 8.0 KB / 262 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** fe54b00 2024-12-17 Add a simple end to end test
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `sae-analysis-v1/tests/test_pytorch_end_to_end.py`
- **Size:** 7.9 KB / 261 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** fe54b00 2024-12-17 Add a simple end to end test
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `sae-analysis-v1/tests/unit/test_dictionary.py`
- **Size:** 4.1 KB / 136 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 0ff8888 2025-02-09 feat: pypi packaging and auto-release with semantic release
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `sae-analysis-v1/.github/workflows/build.yml`
- **Size:** 3.0 KB
- **Original functionality** (leading comment): name: build
- **Provenance:** 0ff8888 2025-02-09 feat: pypi packaging and auto-release with semantic release
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `tests/test_end_to_end.py`
- **Size:** 8.0 KB / 262 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** fe54b00 2024-12-17 Add a simple end to end test
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `tests/test_pytorch_end_to_end.py`
- **Size:** 7.9 KB / 261 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** fe54b00 2024-12-17 Add a simple end to end test
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `tests/unit/test_dictionary.py`
- **Size:** 4.1 KB / 136 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 0ff8888 2025-02-09 feat: pypi packaging and auto-release with semantic release
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `.github/workflows/build.yml`
- **Size:** 3.0 KB
- **Original functionality** (leading comment): name: build
- **Provenance:** 0ff8888 2025-02-09 feat: pypi packaging and auto-release with semantic release
- **Reason:** `DEPENDS-ON-VENDORED` — Test / CI file whose only purpose is to exercise the vendored library; dies with it.
- **Replacement:** no replacement (truly obsolete)

### `sae-analysis-v1/entropy_comparison_resid_out_layer0_20260414_053350.pt`
- **Size:** 484.7 KB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `sae-analysis-v1/data/pythia-70m/20260414_053350/entropy_comparison_resid_out_layer0.pt`

### `sae-analysis-v1/entropy_comparison_resid_out_layer1_20260414_053350.pt`
- **Size:** 415.0 KB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `sae-analysis-v1/data/pythia-70m/20260414_053350/entropy_comparison_resid_out_layer1.pt`

### `sae-analysis-v1/entropy_comparison_resid_out_layer2_20260414_053350.pt`
- **Size:** 1.8 MB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `sae-analysis-v1/data/pythia-70m/20260414_053350/entropy_comparison_resid_out_layer2.pt`

### `sae-analysis-v1/entropy_comparison_resid_out_layer3_20260414_053350.pt`
- **Size:** 1.5 MB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `sae-analysis-v1/data/pythia-70m/20260414_053350/entropy_comparison_resid_out_layer3.pt`

### `sae-analysis-v1/entropy_comparison_resid_out_layer4_20260414_053350.pt`
- **Size:** 982.7 KB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `sae-analysis-v1/data/pythia-70m/20260414_053350/entropy_comparison_resid_out_layer4.pt`

### `sae-analysis-v1/entropy_comparison_resid_out_layer5_20260414_053350.pt`
- **Size:** 26.4 MB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `sae-analysis-v1/data/pythia-70m/20260414_053350/entropy_comparison_resid_out_layer5.pt`

### `sae-analysis-v1/notebooks/feature_analysis.ipynb`
- **Size:** 6.4 MB
- **Original functionality** (inferred): Binary / notebook artefact (.ipynb)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `DUPLICATE` — Near-identical copy of another file still kept.
- **Replacement:** `sae-analysis-v1/notebooks/feature_analysis_cleaned.ipynb`

### `sae-analysis-v1/run_llama3_8b.sh.sh`
- **Size:** 1.4 KB
- **Original functionality** (leading comment): #!/bin/bash -l
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `TYPO-ARTIFACT` — Filename or contents indicate a one-off that was never cleaned up.
- **Replacement:** no replacement (truly obsolete)

### `sae-analysis-v1/scripts/plotting/analyze_feature_token_influence.py`
- **Size:** 5.8 KB / 135 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig03_influence_heatmap.py`

### `sae-analysis-v1/scripts/plotting/analyze_feature_token_influence_simple.py`
- **Size:** 7.2 KB / 159 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Uses the stored influence data directly (no recomputation needed) Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig03_influence_heatmap.py`

### `sae-analysis-v1/scripts/plotting/analyze_feature_token_influence_notebook.py`
- **Size:** 7.1 KB / 162 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig03_influence_heatmap.py`

### `sae-analysis-v1/scripts/plotting/analyze_feature_token_influence_final.py`
- **Size:** 13.6 KB / 360 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig04_entropy_distribution_batches.py`

### `sae-analysis-v1/scripts/plotting/analyze_feature_token_influence_with_batches.py`
- **Size:** 11.5 KB / 260 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Uses entropy comparison file as input, picks a leading feature, then shows tokens Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig04_entropy_distribution_batches.py`

### `sae-analysis-v1/scripts/plotting/plot_entropy_vs_depth.py`
- **Size:** 5.7 KB / 139 LOC
- **Original functionality** (docstring): Jupyter notebook code block to plot entropy of leading features vs depth (layer 0-5).
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig05_entropy_vs_depth.py`

### `sae-analysis-v1/scripts/plotting/plot_entropy_vs_depth_preset.py`
- **Size:** 7.4 KB / 178 LOC
- **Original functionality** (docstring): Entropy-vs-depth figure for any preset (generalization of the paper's Fig 5).
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig05_entropy_vs_depth_preset.py`

### `sae-analysis-v1/scripts/plotting/plot_entropy_vs_depth_comparison.py`
- **Size:** 8.1 KB / 202 LOC
- **Original functionality** (docstring): Cross-model entropy-vs-depth comparison — one panel per model.
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig05_entropy_vs_depth.py`

### `sae-analysis-v1/scripts/plotting/notebook_entropy_vs_depth.py`
- **Size:** 4.6 KB / 118 LOC
- **Original functionality** (leading comment): Jupyter Notebook Code Block: Plot Entropy of Leading Features vs Depth Copy this entire block into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig05_entropy_vs_depth.py`

### `sae-analysis-v1/scripts/plotting/plot_entropy_vs_batch_size_notebook.py`
- **Size:** 4.3 KB / 115 LOC
- **Original functionality** (leading comment): Jupyter Notebook Code Block for Plotting Entropy vs Batch Size Copy this entire block into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig07_entropy_vs_batchsize.py`

### `sae-analysis-v1/scripts/plotting/plot_entropy_vs_activation.py`
- **Size:** 3.8 KB / 101 LOC
- **Original functionality** (leading comment): Plot averaged entropy vs activation count Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig08_entropy_vs_activation.py`

### `sae-analysis-v1/scripts/plotting/plot_feature_entropy_histogram.py`
- **Size:** 3.1 KB / 85 LOC
- **Original functionality** (leading comment): Plot histogram of entropy distribution for a given feature across batches Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig06_entropy_multilayer_histograms.py`

### `sae-analysis-v1/scripts/plotting/plot_all_features_entropy_histogram.py`
- **Size:** 4.3 KB / 122 LOC
- **Original functionality** (leading comment): Plot histogram of entropy distribution for all available features (or top 10) Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `SUPERSEDED` — A canonical replacement exists and produces the same or better output.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig04_entropy_distribution_batches.py`

### `feature_sparsity.py`
- **Size:** 10.2 KB / 295 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/feature_sparsity.py`

### `feature_token_influence.py`
- **Size:** 19.0 KB / 521 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/feature_token_influence.py`

### `token_vector_influence.py`
- **Size:** 18.5 KB / 519 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/token_vector_influence.py`

### `compare_entropies.py`
- **Size:** 30.1 KB / 707 LOC
- **Original functionality** (docstring): Compare entropy between feature influences and token vector influences.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/compare_entropies.py`

### `compare_entropies_multi_layer.py`
- **Size:** 13.9 KB / 375 LOC
- **Original functionality** (docstring): Compare entropy between feature influences and token vector influences for multiple layers.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/compare_entropies_multi_layer.py`

### `compute_correlations.py`
- **Size:** 11.0 KB / 288 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 07738c2 2025-11-28 updated note and correlation computation
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/compute_correlations.py`

### `entropy_vs_batch_size.py`
- **Size:** 17.7 KB / 501 LOC
- **Original functionality** (docstring): Study entropy as a function of batch size.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/entropy_vs_batch_size.py`

### `feature_location_analysis.py`
- **Size:** 11.9 KB / 295 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/analysis/feature_location_analysis.py (moved to deprecated/)`

### `sae_test.py`
- **Size:** 5.3 KB / 143 LOC
- **Original functionality** (leading comment): sae_test_decode_only.py
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/inspect/sae_test.py (moved to deprecated/)`

### `sae_test_with_prompt.py`
- **Size:** 9.3 KB / 249 LOC
- **Original functionality** (leading comment): sae_test_decode_only.py
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/inspect/sae_test_with_prompt.py (moved to deprecated/)`

### `sae_visualizer.py`
- **Size:** 9.9 KB / 280 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/inspect/sae_visualizer.py (moved to deprecated/)`

### `logit_lens.py`
- **Size:** 2.6 KB / 73 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/inspect/logit_lens.py (moved to deprecated/)`

### `test_generation.py`
- **Size:** 1.1 KB / 35 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/inspect/test_generation.py (moved to deprecated/)`

### `test_lm_infer.py`
- **Size:** 681 B / 25 LOC
- **Original functionality** (leading comment): lm_infer.py
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/inspect/test_lm_infer.py (moved to deprecated/)`

### `create_minimal_notebook.py`
- **Size:** 1.9 KB / 68 LOC
- **Original functionality** (docstring): Create a minimal valid notebook from the existing notebook structure.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/utils/create_minimal_notebook.py (moved to deprecated/)`

### `fix_notebook.py`
- **Size:** 3.0 KB / 96 LOC
- **Original functionality** (docstring): Script to fix corrupted Jupyter notebook JSON files.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/utils/fix_notebook.py (moved to deprecated/)`

### `strip_notebook_outputs.py`
- **Size:** 2.2 KB / 68 LOC
- **Original functionality** (docstring): Script to remove outputs from a Jupyter notebook to reduce file size.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/utils/strip_notebook_outputs.py`

### `analyze_feature_token_influence.py`
- **Size:** 5.8 KB / 135 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig03_influence_heatmap.py`

### `analyze_feature_token_influence_simple.py`
- **Size:** 7.2 KB / 159 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Uses the stored influence data directly (no recomputation needed) Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig03_influence_heatmap.py`

### `analyze_feature_token_influence_notebook.py`
- **Size:** 7.1 KB / 162 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig03_influence_heatmap.py`

### `analyze_feature_token_influence_final.py`
- **Size:** 13.6 KB / 360 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig04_entropy_distribution_batches.py`

### `analyze_feature_token_influence_with_batches.py`
- **Size:** 11.5 KB / 260 LOC
- **Original functionality** (leading comment): Analyze which tokens have strong influence on a given feature Uses entropy comparison file as input, picks a leading feature, then shows tokens Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig04_entropy_distribution_batches.py`

### `plot_entropy_vs_depth.py`
- **Size:** 5.7 KB / 139 LOC
- **Original functionality** (docstring): Jupyter notebook code block to plot entropy of leading features vs depth (layer 0-5).
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig05_entropy_vs_depth.py`

### `notebook_entropy_vs_depth.py`
- **Size:** 4.6 KB / 118 LOC
- **Original functionality** (leading comment): Jupyter Notebook Code Block: Plot Entropy of Leading Features vs Depth Copy this entire block into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig05_entropy_vs_depth.py`

### `plot_entropy_vs_batch_size_notebook.py`
- **Size:** 4.3 KB / 115 LOC
- **Original functionality** (leading comment): Jupyter Notebook Code Block for Plotting Entropy vs Batch Size Copy this entire block into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig07_entropy_vs_batchsize.py`

### `plot_entropy_vs_activation.py`
- **Size:** 3.8 KB / 101 LOC
- **Original functionality** (leading comment): Plot averaged entropy vs activation count Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig08_entropy_vs_activation.py`

### `plot_feature_entropy_histogram.py`
- **Size:** 3.1 KB / 85 LOC
- **Original functionality** (leading comment): Plot histogram of entropy distribution for a given feature across batches Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig06_entropy_multilayer_histograms.py`

### `plot_all_features_entropy_histogram.py`
- **Size:** 4.3 KB / 122 LOC
- **Original functionality** (leading comment): Plot histogram of entropy distribution for all available features (or top 10) Copy this into a Jupyter notebook cell
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/scripts/figures/fig04_entropy_distribution_batches.py`

### `feature_analysis.ipynb`
- **Size:** 6.4 MB
- **Original functionality** (inferred): Binary / notebook artefact (.ipynb)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `DUPLICATE` — Near-identical copy of another file still kept.
- **Replacement:** `sae-analysis-v1/notebooks/feature_analysis_cleaned.ipynb`

### `feature_analysis_backup.ipynb`
- **Size:** 5.3 MB
- **Original functionality** (inferred): Binary / notebook artefact (.ipynb)
- **Provenance:** 84fe43a 2026-04-06 Add analysis notebooks
- **Reason:** `DUPLICATE` — Near-identical copy of another file still kept.
- **Replacement:** `sae-analysis-v1/notebooks/feature_analysis_cleaned.ipynb`

### `feature_analysis_v4.ipynb`
- **Size:** 5.3 MB
- **Original functionality** (inferred): Binary / notebook artefact (.ipynb)
- **Provenance:** 84fe43a 2026-04-06 Add analysis notebooks
- **Reason:** `DUPLICATE` — Near-identical copy of another file still kept.
- **Replacement:** `sae-analysis-v1/notebooks/feature_analysis_cleaned.ipynb`

### `feature_analysis_cleaned.ipynb`
- **Size:** 41.2 KB
- **Original functionality** (inferred): Binary / notebook artefact (.ipynb)
- **Provenance:** 84fe43a 2026-04-06 Add analysis notebooks
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/notebooks/feature_analysis_cleaned.ipynb`

### `correlation_matrix.pt`
- **Size:** 438.4 KB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** 07738c2 2025-11-28 updated note and correlation computation
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `regenerable via scripts/analysis/compute_correlations.py`

### `feature_sparsity_data.pt`
- **Size:** 3.3 MB
- **Original functionality** (inferred): Binary / notebook artefact (.pt)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `regenerable via scripts/analysis/feature_sparsity.py`

### `feature_sparsity.csv`
- **Size:** 827.9 KB
- **Original functionality** (leading comment): feature_idx,count,frequency,top_tokens
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `regenerable via scripts/analysis/feature_sparsity.py`

### `logits_comparison_layer3_feature3339.png`
- **Size:** 698.4 KB
- **Original functionality** (inferred): Binary / notebook artefact (.png)
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `regenerable via sae_test_with_prompt.py (deprecated)`

### `sae_heatmap.png`
- **Size:** 141.1 KB
- **Original functionality** (inferred): Binary / notebook artefact (.png)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `regenerable via sae_visualizer.py (deprecated)`

### `sparsity_histogram.png`
- **Size:** 29.0 KB
- **Original functionality** (inferred): Binary / notebook artefact (.png)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `STALE-OUTPUT` — Precomputed data/figure file superseded by a newer timestamped version.
- **Replacement:** `regenerable via scripts/figures/fig01_unique_tokens_histogram.py`

### `dictionaries`
- **Size:** 0 B
- **Original functionality** (inferred): (path is a directory or missing)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/dictionaries (2.4 GB of pretrained SAEs, gitignored, regenerable via pretrained_dictionary_downloader.sh)`

### `README.md`
- **Size:** 22.1 KB
- **Original functionality** (leading comment): # SAE Analysis
- **Provenance:** 741f4d6 2023-10-24 Initial commit
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/README.md (publication landing page)`

### `CLAUDE.md`
- **Size:** 14.0 KB
- **Original functionality** (leading comment): # sae-analysis — project context
- **Provenance:** 1ebe660 2026-04-10 Add CLAUDE.md project context document
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/CLAUDE.md (describes the reorganized layout)`

### `CHANGELOG.md`
- **Size:** 48.0 KB
- **Original functionality** (leading comment): # CHANGELOG
- **Provenance:** 07975f7 2025-02-12 0.1.0
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/CHANGELOG.md`

### `LICENSE`
- **Size:** 1.0 KB
- **Original functionality** (inferred): (unknown file type)
- **Provenance:** 32fec9c 2024-10-22 Create LICENSE
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/LICENSE`

### `.gitignore`
- **Size:** 3.4 KB
- **Original functionality** (inferred): (unknown file type)
- **Provenance:** 741f4d6 2023-10-24 Initial commit
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/.gitignore (more comprehensive)`

### `pyproject.toml`
- **Size:** 1.4 KB
- **Original functionality** (leading comment): [tool.poetry]
- **Provenance:** 0ff8888 2025-02-09 feat: pypi packaging and auto-release with semantic release
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/pyproject.toml (will be updated in Phase 5 to add missing runtime deps from top-level)`

### `note.md`
- **Size:** 8.5 KB
- **Original functionality** (leading comment): # Note on the analysis of sparse auto-encoder
- **Provenance:** 07738c2 2025-11-28 updated note and correlation computation
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/note.md (byte-identical)`

### `pretrained_dictionary_downloader.sh`
- **Size:** 171 B
- **Original functionality** (leading comment): #!/bin/bash
- **Provenance:** 6771aff 2024-03-28 added pretrained_dictionary_downloader.sh
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/pretrained_dictionary_downloader.sh`

### `wikitext-2-train.txt`
- **Size:** 10.3 MB
- **Original functionality** (leading comment): = Valkyria Chronicles III =
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/wikitext-2-train.txt`

### `wikitext-2-test.txt`
- **Size:** 1.2 MB
- **Original functionality** (leading comment): = Robert <unk> =
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/wikitext-2-test.txt`

### `.claude`
- **Size:** 851 B
- **Original functionality** (inferred): (path is a directory or missing)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `REDUNDANT-COPY` — The same file now exists in the v1 subtree and will take its place after promotion.
- **Replacement:** `sae-analysis-v1/.claude (per-project Claude config)`

## Deprecations (moved to `deprecated/`)

### `sae-analysis-v1/scripts/analysis/feature_location_analysis.py`
- **Size:** 11.9 KB / 295 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/analysis/fetch_neuronpedia_interp.py`
- **Size:** 8.6 KB / 228 LOC
- **Original functionality** (docstring): Fetch Neuronpedia autointerp explanations for the top-5-colored features
in each batch of the entropy_comparison analysis.
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/inspect/logit_lens.py`
- **Size:** 2.6 KB / 73 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `DEMO` — Standalone demo / sandbox; parked in deprecated/ for future debugging use.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/inspect/sae_test.py`
- **Size:** 5.3 KB / 143 LOC
- **Original functionality** (leading comment): sae_test_decode_only.py
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `DEMO` — Standalone demo / sandbox; parked in deprecated/ for future debugging use.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/inspect/sae_test_with_prompt.py`
- **Size:** 9.3 KB / 249 LOC
- **Original functionality** (leading comment): sae_test_decode_only.py
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `DEMO` — Standalone demo / sandbox; parked in deprecated/ for future debugging use.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/inspect/sae_visualizer.py`
- **Size:** 9.9 KB / 280 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `DEMO` — Standalone demo / sandbox; parked in deprecated/ for future debugging use.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/inspect/test_generation.py`
- **Size:** 1.1 KB / 35 LOC
- **Original functionality** (inferred): (no docstring or leading comment; see file for details)
- **Provenance:** 82c86b3 2025-11-27 updated spasity caculation for wikitext
- **Reason:** `DEMO` — Standalone demo / sandbox; parked in deprecated/ for future debugging use.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/inspect/test_lm_infer.py`
- **Size:** 681 B / 25 LOC
- **Original functionality** (leading comment): lm_infer.py
- **Provenance:** e7084f4 2025-11-13 initial commit
- **Reason:** `DEMO` — Standalone demo / sandbox; parked in deprecated/ for future debugging use.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/plotting/plot_feature_level_vs_entropy.py`
- **Size:** 7.4 KB / 186 LOC
- **Original functionality** (docstring): Scatter plots: per-feature mean influence entropy H_feat vs feature-intrinsic
"level" proxies (H_vocab, H_logit), one panel per layer.
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/plotting/plot_level_correlation_summary.py`
- **Size:** 8.1 KB / 187 LOC
- **Original functionality** (docstring): Summary figure for the H_vocab/H_logit vs H_feat trial.
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/plotting/regenerate_batch_plots.py`
- **Size:** 14.1 KB / 352 LOC
- **Original functionality** (docstring): Regenerate per-batch diagnostic plots with z-order fix + decoded-text caption.
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/plotting/regenerate_batch_plots_quality.py`
- **Size:** 12.8 KB / 321 LOC
- **Original functionality** (docstring): Trial plotter: rank features by cross-batch activation quality.
- **Provenance:** a49db4d 2026-04-19 Snapshot: pre-cleanup state with sae-analysis-v1/ imported as subdir
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/utils/create_minimal_notebook.py`
- **Size:** 1.9 KB / 68 LOC
- **Original functionality** (docstring): Create a minimal valid notebook from the existing notebook structure.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `RESCUE-UTIL` — Rescue utility whose target file is also being deleted; kept in deprecated/ in case it is useful again.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/scripts/utils/fix_notebook.py`
- **Size:** 3.0 KB / 96 LOC
- **Original functionality** (docstring): Script to fix corrupted Jupyter notebook JSON files.
- **Provenance:** 7986bc7 2026-04-06 Add analysis scripts and update notes
- **Reason:** `RESCUE-UTIL` — Rescue utility whose target file is also being deleted; kept in deprecated/ in case it is useful again.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

### `sae-analysis-v1/out`
- **Size:** 513.0 KB
- **Original functionality** (inferred): (path is a directory or missing)
- **Provenance:** (not tracked in outer repo prior to snapshot; originated in v1 subtree)
- **Reason:** `OFF-PIPELINE` — Exploratory code not referenced by the paper; parked in deprecated/ for possible revival.
- **Replacement:** no replacement (parked in `deprecated/` for future revival)

## Structural changes

- Legacy flat top-level tree (28 `*.py`, 4 `feature_analysis*.ipynb`,
  the vendored `dictionary_learning/`, stale `.pt` outputs, duplicate
  `README.md`/`CLAUDE.md`/`note.md`/`pyproject.toml`/`.gitignore`/
  `LICENSE`/`CHANGELOG.md`) was deleted; see *Deletions* above.
- `sae-analysis-v1/` contents promoted to repo root; the
  `sae-analysis-v1/` directory itself removed.
- `sae-analysis-v1/.git/` was a separate nested repository pointing at
  `github.com/lccqqqqq/sae-analysis-v1.git` — its committed history
  remains available on that remote, but the local clone was discarded
  so that v1's tree could be imported into the outer fork's git history.
- `tests/` and `.github/workflows/` removed as collateral of removing
  `dictionary_learning/` (their only purpose was to exercise it).

## Rollback

- Branch `pre-cleanup-snapshot` on `origin` (commit `a49db4d`) contains the full pre-cleanup state and can be checked out directly.
- Local tarball `/mnt/users/clin/sae-analysis-backup-20260416.tar.gz` contains everything including the gitignored SAE weights and experiment `.pt` files — extract with `tar -xzf <path> -C /tmp/`.
