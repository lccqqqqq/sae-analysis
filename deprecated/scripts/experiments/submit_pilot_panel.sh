#!/usr/bin/env bash
# Stage 2 of the cherry-pick pilot: per-feature panel of H_pos / H_vocab /
# H_logit / density / Neuronpedia auto-interp, written to a single CSV.
#
# Reads the YAML at scripts/experiments/configs/pilot_features_*.yaml — fill
# the `features:` list in there before running this stage. H_pos requires
# forward+backward per firing event; H_vocab requires one corpus sweep.
# Comfortably fits on a 24 GB RTX 4090 for Gemma-2-2B + Gemma Scope.
#
# Usage:
#   scripts/experiments/submit_pilot_panel.sh \
#       [--config scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml] \
#       [--target-events 20] [--max-batches 4000] [--context-len 64] \
#       [--skip-neuronpedia] \
#       [--queue gpulong] [--gputype rtx4090with24gb] [--mem 16] \
#       [--dry-run]

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

CONFIG="${REPO_ROOT}/scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml"
TARGET_EVENTS=20
MAX_BATCHES=4000
CONTEXT_LEN=64
SKIP_NEURONPEDIA=0
QUEUE="gpulong"
GPUTYPE="rtx4090with24gb"
# 8 GB/core × 2 cores = 16 GB host RAM. rtx4090 nodes have 29.3 GB total
# host RAM; -m 8 fits comfortably and lets the job start immediately.
MEM=8
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)            CONFIG="$2"; shift 2;;
        --target-events)     TARGET_EVENTS="$2"; shift 2;;
        --max-batches)       MAX_BATCHES="$2"; shift 2;;
        --context-len)       CONTEXT_LEN="$2"; shift 2;;
        --skip-neuronpedia)  SKIP_NEURONPEDIA=1; shift;;
        --queue)             QUEUE="$2"; shift 2;;
        --gputype)           GPUTYPE="$2"; shift 2;;
        --mem)               MEM="$2"; shift 2;;
        --dry-run|-n)        DRY_RUN=1; shift;;
        -h|--help)           sed -n '2,17p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 2
fi

# Best-effort sanity check: warn if the config still has only the placeholder.
PRESET=$(/usr/bin/python3 -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG}'))['preset'])")
LAYER=$(/usr/bin/python3 -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG}'))['layer'])")
NFEAT=$(/usr/bin/python3 -c "import yaml,sys; print(len(yaml.safe_load(open('${CONFIG}'))['features']))")
echo "[submit] config preset=${PRESET} layer=${LAYER} n_features=${NFEAT}"
if [[ "$NFEAT" -lt 4 ]]; then
    echo "[WARN] config has only ${NFEAT} feature(s); the pilot is meant to span ~15-25 across tiers." >&2
    echo "[WARN] continuing anyway — pass --dry-run to inspect the addqueue command first." >&2
fi

mkdir -p "${REPO_ROOT}/out" "${REPO_ROOT}/data/${PRESET}"
TAG="pilot_panel_${PRESET}_l${LAYER}"
OUT="${REPO_ROOT}/out/${TAG}_%j.out"
COMMENT="pilot ${TAG} (q=${QUEUE}/${GPUTYPE})"

PY_ARGS=(
    --config "${CONFIG}"
    --target-events "${TARGET_EVENTS}"
    --max-batches "${MAX_BATCHES}"
    --context-len "${CONTEXT_LEN}"
)
[[ $SKIP_NEURONPEDIA -eq 1 ]] && PY_ARGS+=(--skip-neuronpedia)

PYCMD=(
    /usr/bin/env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
    stdbuf -oL -eL
    /usr/bin/python3
    "${REPO_ROOT}/scripts/analysis/feature_pilot_panel.py"
    "${PY_ARGS[@]}"
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

echo "[submit] queue=${QUEUE}/${GPUTYPE} mem=${MEM}G target_events=${TARGET_EVENTS}"
echo "[submit] output CSV: data/${PRESET}/feature_pilot_layer${LAYER}.csv"
echo "[submit] follow-up:  scripts/plot/pilot_scatter.py --csv <CSV> --context-len ${CONTEXT_LEN}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would invoke:"
    printf '    %q' "${ADDQUEUE_CMD[@]}"; echo
    exit 0
fi

"${ADDQUEUE_CMD[@]}"
