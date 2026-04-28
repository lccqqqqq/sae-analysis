#!/usr/bin/env bash
# Submit compare_entropies_multi_layer.py for all six SAE presets.
#
# Usage:
#   scripts/experiments/submit_all_presets.sh [--smoke] [--num-batches N]
#       [--queue Q] [--gputype G] [--mem GB]
#
# Pass-through args are forwarded verbatim to submit_entropy_bench.sh per
# preset. Per-preset queue/memory/layer defaults live there; override per-call
# with --queue / --gputype / --mem if you want to deviate (e.g. force every
# preset onto sondhigpu).
#
# Smoke vs. real:
#   --smoke      3 batches, sparse layer subset per preset (~5 min each on H200,
#                ~30 min on RTX 4090)
#   (no flag)    50 batches, all preset-default layers (real run; hours per
#                preset, especially gemma-2-2b and llama-3-8b)
#
# Watch with:
#   q
#   tail -f out/entropy_<preset>_<jobid>.out

set -euo pipefail

cd "$(dirname "$0")"
SUBMIT="./submit_entropy_bench.sh"

# Order: smallest first so any failure shows up quickly during smoke runs.
PRESETS=(
    pythia-70m
    qwen2-0.5b
    gpt2-small
    llama-3.2-1b
    gemma-2-2b
    llama-3-8b
)

declare -a JOB_LINES

for preset in "${PRESETS[@]}"; do
    echo
    echo "================================================================"
    echo "Submitting: $preset"
    echo "================================================================"
    if ! "$SUBMIT" --preset "$preset" "$@"; then
        echo "[WARN] submission failed for $preset (continuing)" >&2
    fi
done

echo
echo "================================================================"
echo "All ${#PRESETS[@]} preset submissions issued."
echo "================================================================"
echo "Inspect the queue with:   q"
echo "Live tail:                 tail -f out/entropy_<preset>_<jobid>.out"
echo "GPU availability:          showgpus"
