# Publication Checklist

This is a computational paper (analysis scripts + data pipeline + figures).

---

## Paper
- [x] Paper source is included and designated as ground truth (`paper/paper.tex`)
- [ ] Paper compiles without errors (`pdflatex paper/paper.tex`)
- [x] All figures match the submitted version (`paper/figures/`)
- [x] BibTeX entry is complete and correct (see AGENTS.md Citation section)
- [x] License file is present (`LICENSE`)

## Code
- [x] All scripts run with the publication repo directory structure
- [x] Import paths are updated for the publication repo layout
- [ ] No hardcoded absolute paths — verify by grepping scripts for `/Users/`
- [ ] Dependencies are pinned (currently documented in README setup commands; consider freezing to `requirements.txt`)
- [x] Figure generation scripts (`scripts/figures/fig01–fig08`) produce output matching `paper/figures/`

## Data
- [x] Local data files excluded (large .pt and .csv files not committed)
- [x] External data links documented in AGENTS.md Repository Map:
  - [x] WikiText-2 — downloaded via `python scripts/utils/download_data.py` (HuggingFace datasets)
  - [x] Pythia-70m-deduped — auto-downloaded by HuggingFace `transformers`
  - [x] Pretrained SAEs — `./pretrained_dictionary_downloader.sh` (~2.5 GB from baulab.us)
- [ ] Verify pretrained SAE download URL is still live: `curl -sIL https://baulab.us/u/smarks/autoencoders/`
- [x] Download instructions documented and tested (see AGENTS.md and README)

## Environment
- [x] Setup instructions in README and AGENTS.md
- [x] Platform documented: macOS Apple M-series, MPS backend, Python 3.10+
- [x] Heavy computation requirements flagged in AGENTS.md Computational Requirements

## Supplementary Materials
- [x] `know-how.md` captures key methodology decisions and tacit knowledge
- [x] `authors-note.md` reflects what the author wants readers to know
- [ ] Conversation history — not included (researcher preference)
- [ ] Slides (`slides/sae_analysis.pdf`) — decide whether to include in `supplementary/materials/`
- [x] All supplementary files pass confidentiality screening (no credentials or private paths)

## AGENTS.md
- [x] All file paths in Repository Map exist in the repo
- [x] All commands in figure generation table match `scripts/figures/fig01–fig08`
- [x] Paper Summary captures what makes the work distinctive
- [x] Ground truth hierarchy is clear (paper is authoritative)
- [x] Computational requirements are accurate (verified against paper appendix)
- [x] Citation entry is correct
- [ ] Supplementary Materials section added — update AGENTS.md to reference `supplementary/`

## Review
- [x] Structure review passed
- [x] AGENTS.md review passed (v1.0.1)
- [x] README review passed (v1.0.1)
- [ ] Full review passed after supplementary materials added

## Final
- [x] README is complete and accurate (v1.0.1)
- [x] `.gitignore` covers build artifacts and generated files
- [x] No sensitive information in any committed file
- [x] Researcher has reviewed and approved supplementary materials
