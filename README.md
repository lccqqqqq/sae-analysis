# Analyzing the Nonlocality of Sparse Autoencoder Features

**Xiao-Liang Qi** — Stanford Institute for Theoretical Physics, Stanford University

We introduce a gradient-based entropy measure that quantifies how nonlocal each SAE feature is in the input token space of a transformer. Motivated by an analogy with holographic duality (AdS/CFT), we apply this measure to Pythia-70m-deduped with pretrained SAEs across all six layers, finding that features exhibit a broad hierarchy of nonlocality, deeper layers produce more nonlocal features, and individual features have lower entropy than the reconstructed token vector.

## Talk to this paper

This paper is published with an AI agent ([Agentic Publication Protocol](https://github.com/LionSR/AgenticPublicationProtocol)). Clone this repo and open it in an AI coding agent to ask questions, reproduce figures, and explore the work.

**Claude Code:**
```bash
git clone https://github.com/XiaoliangQi/sae-analysis-v1
cd sae-analysis-v1
claude
```
Or use the load skill from any Claude Code session:
```
/paper-protocol:load-paper-agent https://github.com/XiaoliangQi/sae-analysis-v1
```

**Other agents (Codex, Cursor, etc.):**
Clone the repo and open it — any agent that reads `AGENTS.md` or `README.md` will pick up the paper context automatically.

## Figures

Each figure has a dedicated generation script in `scripts/figures/` that saves directly to `paper/figures/`.

| Figure | Description | Script | Required data |
|--------|-------------|--------|---------------|
| Fig. 1 | Unique-token distribution per feature (layer 0) | `scripts/figures/fig01_unique_tokens_histogram.py` | `feature_sparsity_data_resid_out_layer0.pt` |
| Fig. 2 | Feature correlation matrix (layer 3) | `scripts/figures/fig02_correlation_heatmap.py` | `correlation_matrix_resid_out_layer3.pt` |
| Fig. 3 | Influence heatmap for Feature 531 (layer 3) | `scripts/figures/fig03_influence_heatmap.py` | `feature_token_influence_resid_out_layer3.pt` |
| Fig. 4 | Per-feature entropy distributions (layer 3) | `scripts/figures/fig04_entropy_distribution_batches.py` | `feature_token_influence_resid_out_layer3.pt` |
| Fig. 5 | Feature entropy vs. layer depth | `scripts/figures/fig05_entropy_vs_depth.py` | `entropy_comparison_resid_out_layer{0..5}_*.pt` |
| Fig. 6 | Entropy histograms across layers 0–4 | `scripts/figures/fig06_entropy_multilayer_histograms.py` | `entropy_comparison_resid_out_layer{0..4}_*.pt` |
| Fig. 7 | Feature entropy vs. context length (layer 2) | `scripts/figures/fig07_entropy_vs_batchsize.py` | `entropy_vs_batch_size_resid_out_layer2_*.pt` |
| Fig. 8 | Entropy vs. activation probability (layer 5) | `scripts/figures/fig08_entropy_vs_activation.py` | `feature_token_influence_resid_out_layer5.pt` + `feature_sparsity_data_resid_out_layer5.pt` |

## Reproducing results

### Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install . torch torchvision torchaudio
uv pip install transformers accelerate einops datasets tqdm numpy matplotlib scipy
./pretrained_dictionary_downloader.sh   # downloads pretrained SAEs (~2.5 GB)
python scripts/utils/download_data.py   # downloads WikiText-2
```

### Generate figures from pre-computed data

If you have the `.pt` data files, run the figure scripts directly:

```bash
python scripts/figures/fig01_unique_tokens_histogram.py
python scripts/figures/fig02_correlation_heatmap.py
# ... through fig08
```

### Run the full pipeline from scratch

```bash
# Stage 1: sparsity statistics (needed for Figs 1, 2, 3, 4, 8)
python scripts/analysis/feature_sparsity.py          # ~5–15 min per layer, MPS

# Stage 2: correlation matrix (Fig 2)
python scripts/analysis/compute_correlations.py

# Stage 3: feature influence (Figs 3, 4, 8)
python scripts/analysis/feature_token_influence.py   # ~20–60 min per layer, MPS

# Stage 4: entropy comparison across all layers (Figs 5, 6)
python scripts/analysis/compare_entropies_multi_layer.py --layers 0 1 2 3 4 5 --num-batches 50

# Stage 5: context-length sweep (Fig 7)
python scripts/analysis/entropy_vs_batch_size.py --site resid_out_layer2
```

See `AGENTS.md` for full details including per-task timing estimates.

## Repository structure

```
paper/                  ← LaTeX source and figures (ground truth)
scripts/
├── figures/            ← one script per paper figure (fig01–fig08)
├── analysis/           ← data-producing scripts (run these first)
├── plotting/           ← exploratory plotting utilities
├── inspect/            ← standalone model/SAE demos
└── utils/              ← data download and notebook utilities
notebooks/              ← Jupyter notebooks for exploratory analysis
dictionary_learning/    ← SAE library (Marks, Karvonen & Mueller)
```

## Citation

```bibtex
@article{qi2026nonlocality,
  title={Analyzing the Nonlocality of Sparse Autoencoder Features},
  author={Qi, Xiao-Liang},
  year={2026},
  note={Stanford Institute for Theoretical Physics, Stanford University}
}
```
