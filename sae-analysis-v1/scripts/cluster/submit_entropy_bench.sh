#!/usr/bin/env bash
# Submit compare_entropies_multi_layer.py for a given preset to sondhigpu (H200).
#
# Usage:
#   scripts/cluster/submit_entropy_bench.sh --preset <name> [--smoke] [--num-batches N] [--layers "0 3 5"]
#
# Presets (see scripts/analysis/presets.py):
#   pythia-70m     (6 layers,  d_model=512,  default layers 0-5)
#   gpt2-small     (12 layers, d_model=768)
#   gemma-2-2b     (26 layers, d_model=2304)  -- larger Jacobian, needs more GPU mem
#   llama-3.2-1b   (16 layers, d_model=2048, SAE covers L0-L8)
#   qwen2-0.5b     (24 layers, d_model=896)
#
# --smoke runs 3 batches with a reduced layer set for quick validation.

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

# ---- Parse args ------------------------------------------------------------
PRESET=""
SMOKE=0
NBATCH_OVERRIDE=""
LAYERS_OVERRIDE=""
QUEUE_OVERRIDE=""
GPUTYPE_OVERRIDE=""
MEM_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)       PRESET="$2"; shift 2;;
        --smoke)        SMOKE=1; shift;;
        --num-batches)  NBATCH_OVERRIDE="$2"; shift 2;;
        --layers)       LAYERS_OVERRIDE="$2"; shift 2;;
        --queue)        QUEUE_OVERRIDE="$2"; shift 2;;
        --gputype)      GPUTYPE_OVERRIDE="$2"; shift 2;;
        --mem)          MEM_OVERRIDE="$2"; shift 2;;
        -h|--help)
            sed -n '2,14p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$PRESET" ]]; then
    echo "ERROR: --preset <name> is required." >&2
    echo "Available presets: pythia-70m gpt2-small gemma-2-2b llama-3.2-1b qwen2-0.5b" >&2
    exit 2
fi

# ---- Per-preset defaults ---------------------------------------------------
# Layer set and memory budget. d_model >= 2048 -> jacobian is noticeably larger.
case "$PRESET" in
    pythia-70m)    LAYERS="0 1 2 3 4 5"; MEM=8 ;;
    gpt2-small)    LAYERS="0 1 2 3 4 5 6 7 8 9 10 11"; MEM=16 ;;
    gemma-2-2b)    LAYERS="$(seq -s ' ' 0 25)"; MEM=48 ;;
    llama-3.2-1b)  LAYERS="0 1 2 3 4 5 6 7 8"; MEM=24 ;;
    qwen2-0.5b)    LAYERS="$(seq -s ' ' 0 11)"; MEM=16 ;;
    *)             echo "ERROR: unknown preset $PRESET" >&2; exit 2 ;;
esac

NBATCH=50
if [[ $SMOKE -eq 1 ]]; then
    NBATCH=3
    # Smoke: a sparse subset of layers (first, middle, last when available).
    case "$PRESET" in
        pythia-70m)   LAYERS="0 3 5" ;;
        gpt2-small)   LAYERS="0 5 11" ;;
        gemma-2-2b)   LAYERS="0 12 25" ;;
        llama-3.2-1b) LAYERS="0 4 8" ;;
        qwen2-0.5b)   LAYERS="0 5 11" ;;
    esac
fi

# Manual overrides take precedence.
[[ -n "$NBATCH_OVERRIDE"  ]] && NBATCH="$NBATCH_OVERRIDE"
[[ -n "$LAYERS_OVERRIDE"  ]] && LAYERS="$LAYERS_OVERRIDE"

if [[ $SMOKE -eq 1 ]]; then TAG="${PRESET}_smoke"; else TAG="${PRESET}"; fi
COMMENT="sae entropy ${TAG} (H200, num_batches=${NBATCH})"
OUT="${REPO_ROOT}/out/entropy_${TAG}_%j.out"
mkdir -p "${REPO_ROOT}/out" "${REPO_ROOT}/data"

PYCMD=(
    /usr/bin/env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}/scripts/analysis:${PYTHONPATH:-}"
    stdbuf -oL -eL
    /usr/bin/python3
    "${REPO_ROOT}/scripts/analysis/compare_entropies_multi_layer.py"
    --preset "${PRESET}"
    --layers ${LAYERS}
    --num-batches "${NBATCH}"
    --random-seed 0
    --log-every 1
    --heartbeat-interval 30
    --output-dir "${REPO_ROOT}/data"
)

QUEUE="${QUEUE_OVERRIDE:-sondhigpu}"
[[ -n "$MEM_OVERRIDE" ]] && MEM="$MEM_OVERRIDE"

GPUTYPE_ARGS=()
[[ -n "$GPUTYPE_OVERRIDE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE_OVERRIDE")

echo "[submit] preset=${PRESET} smoke=${SMOKE} queue=${QUEUE} mem=${MEM}G "\
"layers=(${LAYERS}) num_batches=${NBATCH}"
echo "[submit] command: ${PYCMD[*]}"

addqueue \
    -c "${COMMENT}" \
    -q "${QUEUE}" --gpus 1 "${GPUTYPE_ARGS[@]}" \
    -s -n 2 -m "${MEM}" \
    -o "${OUT}" \
    "${PYCMD[@]}"
