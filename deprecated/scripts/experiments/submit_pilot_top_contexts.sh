#!/usr/bin/env bash
# Stage 1 of the cherry-pick pilot: scan the corpus for top-k activating
# context windows for a list of candidate feature indices, so the user can
# read them and assign tier labels (token / phrase / concept / abstract)
# in the YAML at scripts/experiments/configs/pilot_features_*.yaml.
#
# Forward-only passes — no gradient — so this is the cheap stage. ~4000
# windows of 64 tokens through Gemma-2-2B + Gemma Scope at one layer.
#
# Usage:
#   scripts/experiments/submit_pilot_top_contexts.sh \
#       --feature-ids "0 5 17 42 ..." \
#       [--preset gemma-2-2b] [--layer 12] \
#       [--top-k 8] [--context-len 64] [--max-batches 4000] \
#       [--queue gpulong] [--gputype rtx4090with24gb] [--mem 16] \
#       [--dry-run]

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

PRESET="gemma-2-2b"
LAYER=12
FEATURE_IDS=""
TOP_K=8
CONTEXT_LEN=64
MAX_BATCHES=4000
QUEUE="gpulong"
GPUTYPE="rtx4090with24gb"
# 8 GB/core × 2 cores = 16 GB host RAM. rtx4090 nodes have 29.3 GB total
# host RAM; -m 8 fits comfortably and lets the job start immediately.
MEM=8
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)        PRESET="$2"; shift 2;;
        --layer)         LAYER="$2"; shift 2;;
        --feature-ids)   FEATURE_IDS="$2"; shift 2;;
        --top-k)         TOP_K="$2"; shift 2;;
        --context-len)   CONTEXT_LEN="$2"; shift 2;;
        --max-batches)   MAX_BATCHES="$2"; shift 2;;
        --queue)         QUEUE="$2"; shift 2;;
        --gputype)       GPUTYPE="$2"; shift 2;;
        --mem)           MEM="$2"; shift 2;;
        --dry-run|-n)    DRY_RUN=1; shift;;
        -h|--help)       sed -n '2,17p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$FEATURE_IDS" ]]; then
    echo "ERROR: --feature-ids \"id1 id2 ...\" is required." >&2
    exit 2
fi

mkdir -p "${REPO_ROOT}/out" "${REPO_ROOT}/data/${PRESET}"
TAG="top_contexts_${PRESET}_l${LAYER}"
OUT="${REPO_ROOT}/out/${TAG}_%j.out"
COMMENT="pilot ${TAG} (q=${QUEUE}/${GPUTYPE})"

PYCMD=(
    /usr/bin/env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
    stdbuf -oL -eL
    /usr/bin/python3
    "${REPO_ROOT}/scripts/analysis/top_activating_contexts.py"
    --preset "${PRESET}"
    --layer "${LAYER}"
    --top-k "${TOP_K}"
    --context-len "${CONTEXT_LEN}"
    --max-batches "${MAX_BATCHES}"
    --output-dir "${REPO_ROOT}/data/${PRESET}"
    --feature-ids ${FEATURE_IDS}
)

GPUTYPE_ARGS=()
[[ -n "$GPUTYPE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE")

ADDQUEUE_CMD=(
    addqueue
    -c "$COMMENT"
    -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
    -s -n 2 -m "$MEM"
    -o "$OUT"
    "${PYCMD[@]}"
)

echo "[submit] ${TAG}"
echo "[submit] queue=${QUEUE}/${GPUTYPE} mem=${MEM}G layer=${LAYER}"
echo "[submit] features: ${FEATURE_IDS}"
echo "[submit] outputs:  data/${PRESET}/top_contexts_layer${LAYER}.{json,txt}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would invoke:"
    printf '    %q' "${ADDQUEUE_CMD[@]}"; echo
    exit 0
fi

"${ADDQUEUE_CMD[@]}"
