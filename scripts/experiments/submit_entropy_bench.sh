#!/usr/bin/env bash
# Submit compare_entropies_multi_layer.py for one preset via Oxford addqueue.
#
# Usage:
#   scripts/experiments/submit_entropy_bench.sh --preset <name> [--smoke] \
#       [--dry-run] [--num-batches N] [--layers "0 3 5"] \
#       [--queue Q] [--gputype G] [--mem GB]
#
# Presets (see scripts/analysis/presets.py):
#   pythia-70m     (6 layers,  d_model=512)
#   gpt2-small     (12 layers, d_model=768)
#   gemma-2-2b     (26 layers, d_model=2304)   -- larger Jacobian
#   llama-3.2-1b   (16 layers, d_model=2048, SAE covers L0-L8)
#   llama-3-8b     (32 layers, d_model=4096)   -- needs sondhigpu (H200, 141 GB)
#   qwen2-0.5b     (24 layers, d_model=896,  SAE covers L0-L11)
#
# Default GPU target: 24 GB RTX (gpulong queue, --gputype rtx4090with24gb).
# llama-3-8b defaults to sondhigpu because the model + its SAEs don't fit
# in 24 GB. Override --queue / --gputype on any preset.
#
# --smoke runs 3 batches with a sparse layer subset for quick validation.

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

# ---- Parse args ------------------------------------------------------------
PRESET=""
SMOKE=0
DRY_RUN=0
NBATCH_OVERRIDE=""
LAYERS_OVERRIDE=""
QUEUE_OVERRIDE=""
GPUTYPE_OVERRIDE=""
MEM_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)       PRESET="$2"; shift 2;;
        --smoke)        SMOKE=1; shift;;
        --dry-run|-n)   DRY_RUN=1; shift;;
        --num-batches)  NBATCH_OVERRIDE="$2"; shift 2;;
        --layers)       LAYERS_OVERRIDE="$2"; shift 2;;
        --queue)        QUEUE_OVERRIDE="$2"; shift 2;;
        --gputype)      GPUTYPE_OVERRIDE="$2"; shift 2;;
        --mem)          MEM_OVERRIDE="$2"; shift 2;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$PRESET" ]]; then
    echo "ERROR: --preset <name> is required." >&2
    echo "Available: pythia-70m gpt2-small gemma-2-2b llama-3.2-1b llama-3-8b qwen2-0.5b" >&2
    exit 2
fi

# ---- Per-preset defaults ---------------------------------------------------
# Memory is host RAM per core (addqueue -m); GPU memory is implied by gputype.
case "$PRESET" in
    pythia-70m)
        LAYERS="0 1 2 3 4 5"
        MEM=8
        DEF_QUEUE="gpulong"; DEF_GPUTYPE="rtx4090with24gb" ;;
    gpt2-small)
        LAYERS="0 1 2 3 4 5 6 7 8 9 10 11"
        MEM=16
        DEF_QUEUE="gpulong"; DEF_GPUTYPE="rtx4090with24gb" ;;
    gemma-2-2b)
        # 26 layers; Jacobian is d_model^2 * seq_len ~ 30 MB per layer per
        # batch -- fine, but the 2.6B model in fp32 (~10 GB GPU) plus 26 SAEs
        # makes the resident set ~22 GB. Tight on 24 GB.
        LAYERS="$(seq -s ' ' 0 25)"
        MEM=32
        DEF_QUEUE="gpulong"; DEF_GPUTYPE="rtx4090with24gb" ;;
    llama-3.2-1b)
        LAYERS="0 1 2 3 4 5 6 7 8"
        MEM=16
        DEF_QUEUE="gpulong"; DEF_GPUTYPE="rtx4090with24gb" ;;
    llama-3-8b)
        # 8B model in fp32 alone is ~32 GB. Default layer subset matches the
        # preset's default_layers (11 evenly-spaced of 31 available SAEs).
        LAYERS="0 3 6 9 12 15 18 21 24 27 30"
        MEM=64
        DEF_QUEUE="sondhigpu"; DEF_GPUTYPE="" ;;
    qwen2-0.5b)
        LAYERS="$(seq -s ' ' 0 11)"
        MEM=16
        DEF_QUEUE="gpulong"; DEF_GPUTYPE="rtx4090with24gb" ;;
    *)
        echo "ERROR: unknown preset $PRESET" >&2; exit 2 ;;
esac

NBATCH=50
if [[ $SMOKE -eq 1 ]]; then
    NBATCH=3
    case "$PRESET" in
        pythia-70m)   LAYERS="0 3 5" ;;
        gpt2-small)   LAYERS="0 5 11" ;;
        gemma-2-2b)   LAYERS="0 12 25" ;;
        llama-3.2-1b) LAYERS="0 4 8" ;;
        llama-3-8b)   LAYERS="0 15 30" ;;
        qwen2-0.5b)   LAYERS="0 5 11" ;;
    esac
fi

# Manual overrides take precedence.
[[ -n "$NBATCH_OVERRIDE"  ]] && NBATCH="$NBATCH_OVERRIDE"
[[ -n "$LAYERS_OVERRIDE"  ]] && LAYERS="$LAYERS_OVERRIDE"
QUEUE="${QUEUE_OVERRIDE:-$DEF_QUEUE}"
# Use ${var-default} (no colon) so --gputype "" can explicitly clear the
# preset default (needed e.g. when overriding --queue sondhigpu, which has
# only one GPU type).
GPUTYPE="${GPUTYPE_OVERRIDE-$DEF_GPUTYPE}"
[[ -n "$MEM_OVERRIDE" ]] && MEM="$MEM_OVERRIDE"

# ---- Build the python command ---------------------------------------------
if [[ $SMOKE -eq 1 ]]; then TAG="${PRESET}_smoke"; else TAG="${PRESET}"; fi
COMMENT="sae entropy ${TAG} (q=${QUEUE}, n=${NBATCH})"
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

GPUTYPE_ARGS=()
[[ -n "$GPUTYPE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE")

echo "[submit] preset=${PRESET} smoke=${SMOKE} queue=${QUEUE} "\
"gputype=${GPUTYPE:-(any)} mem=${MEM}G layers=(${LAYERS}) n=${NBATCH}"
echo "[submit] command: ${PYCMD[*]}"

ADDQUEUE_CMD=(
    addqueue
    -c "${COMMENT}"
    -q "${QUEUE}" --gpus 1 "${GPUTYPE_ARGS[@]}"
    -s -n 2 -m "${MEM}"
    -o "${OUT}"
    "${PYCMD[@]}"
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would invoke:"
    printf '    %q' "${ADDQUEUE_CMD[@]}"; echo
    exit 0
fi

"${ADDQUEUE_CMD[@]}"
