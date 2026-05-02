#!/usr/bin/env bash
# Wikitext CTX_LEN sweep on the existing wikitext experiment dir.
#
# No corpus-tensor flag (so topact_entropy.py loads wikitext as the
# corpus). Sweeps CTX_LEN values that may exceed Neuronpedia's 128-token
# cap — useful for studying where the H_pos-vs-ctx_len curve goes once
# we leave the SAE's training-context regime.
#
# Usage:
#   scripts/experiments/submit_wikitext_ctxsweep.sh \
#       [--expt-dir <path>] \
#       [--ctx-lens "256 512"] \
#       [--top-k 16] \
#       [--queue sondhigpu] [--mem 8] [--dry-run]
#
# Default expt-dir is the original wikitext run at
#   data/cherry_picked_feature_entropy/20260430_235742/
# (which has top_contexts_layer12.json from its Step 0 corpus scan).

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

EXPT_DIR="${REPO_ROOT}/data/cherry_picked_feature_entropy/20260430_235742"
YAML="${REPO_ROOT}/scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml"
CTX_LENS="256 512"
TOP_K=16
QUEUE="sondhigpu"
GPUTYPE=""
MEM=8
DRY_RUN=0
TIERS=(token phrase concept abstract)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt-dir)    EXPT_DIR="$2"; shift 2;;
        --yaml)        YAML="$2"; shift 2;;
        --ctx-lens)    CTX_LENS="$2"; shift 2;;
        --top-k)       TOP_K="$2"; shift 2;;
        --queue)       QUEUE="$2"; shift 2;;
        --gputype)     GPUTYPE="$2"; shift 2;;
        --mem)         MEM="$2"; shift 2;;
        --dry-run|-n)  DRY_RUN=1; shift;;
        -h|--help)     sed -n '2,17p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

EXPT_DIR=$(/usr/bin/python3 -c "import os; print(os.path.realpath('${EXPT_DIR}'))")
LAYER=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['layer'])")
JSON_PATH="${EXPT_DIR}/top_contexts_layer${LAYER}.json"

for f in "$JSON_PATH" "${EXPT_DIR}/manifest.json"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required input missing: $f" >&2
        exit 2
    fi
done

# Update manifest to record sweep.
/usr/bin/python3 - "${EXPT_DIR}/manifest.json" "$CTX_LENS" <<'PY'
import json, sys
manifest_path, ctx_lens_str = sys.argv[1:]
m = json.load(open(manifest_path))
new_lens = [int(x) for x in ctx_lens_str.split()]
prior = m.get("ctx_lens_swept", [])
m["ctx_lens_swept"] = sorted(set(prior + new_lens + [m.get("ctx_len", 64)]))
json.dump(m, open(manifest_path, "w"), indent=2)
print(f"[manifest] ctx_lens_swept={m['ctx_lens_swept']}")
PY

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "${REPO_ROOT}/out"

echo "================================================================"
echo "Wikitext CTX_LEN sweep on existing experiment"
echo "  expt_dir : ${EXPT_DIR}"
echo "  ctx_lens : ${CTX_LENS}"
echo "  top_k    : ${TOP_K}"
echo "  queue    : ${QUEUE}${GPUTYPE:+/$GPUTYPE}    mem: ${MEM} GB/core"
echo "================================================================"

GPUTYPE_ARGS=()
[[ -n "$GPUTYPE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE")

for TIER in "${TIERS[@]}"; do
    OUT="${REPO_ROOT}/out/cpfe_${TIMESTAMP}_wkctxsweep_${TIER}_%j.out"
    PYCMD=(
        /usr/bin/env
        PYTHONUNBUFFERED=1
        PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
        stdbuf -oL -eL
        /usr/bin/python3
        "${REPO_ROOT}/scripts/analysis/topact_entropy.py"
        --yaml "${YAML}"
        --top-contexts "${JSON_PATH}"
        --output-dir "${EXPT_DIR}"
        --ctx-len ${CTX_LENS}
        --top-k "${TOP_K}"
        --tier-filter "${TIER}"
    )
    CMD=(
        addqueue
        -c "cpfe ${TIMESTAMP} wkctxsweep ${TIER}"
        -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
        -s -n 2 -m "$MEM"
        -o "$OUT"
        "${PYCMD[@]}"
    )
    echo
    echo "  tier=${TIER}"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] would invoke:"
        printf '      %q' "${CMD[@]}"; echo
    else
        if ! "${CMD[@]}"; then
            echo "[WARN] submission failed for tier ${TIER}" >&2
        fi
    fi
done

echo
echo "================================================================"
echo "Watch:  q   |   tail -f out/cpfe_${TIMESTAMP}_wkctxsweep_*.out"
echo "Outputs land in: ${EXPT_DIR}"
echo "================================================================"
