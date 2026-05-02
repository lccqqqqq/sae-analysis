#!/usr/bin/env bash
# Cherry-picked feature-entropy experiment, Neuronpedia variant.
#
# Source of top-activating contexts: cached Neuronpedia API payloads under
# data/neuronpedia_cache/. No corpus scan, no Step 0; the adapter runs
# locally on CPU, then a small smoke-verify GPU job confirms the
# tokenizer round-trip + activation match before the four parallel
# entropy jobs are submitted.
#
# Outputs land under
#   data/cherry_picked_feature_entropy/<TIMESTAMP>_neuronpedia/
#       manifest.json
#       corpus_tokens.pt           # virtual corpus from the adapter
#       top_contexts_layer<L>.json # same shape as the wikitext run
#       adapter_log.json
#       ctx<N>_<tier>.csv          # one per tier from Step 1
#       influences_<tier>.npz      # one per tier from Step 1
#
# Usage:
#   scripts/experiments/submit_cherry_picked_entropy_neuronpedia.sh \
#       [--yaml <path>] [--ctx-len 64] [--top-k 32] \
#       [--queue sondhigpu] [--gputype ""] [--mem 8] \
#       [--dry-run]

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

# ---- defaults --------------------------------------------------------------
YAML="${REPO_ROOT}/scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml"
CTX_LEN=64
TOP_K=32
QUEUE="sondhigpu"
GPUTYPE=""              # sondhigpu has only h200; no --gputype needed
MEM=8
DRY_RUN=0
TIERS=(token phrase concept abstract)
DIR_SUFFIX="neuronpedia"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)        YAML="$2"; shift 2;;
        --ctx-len)     CTX_LEN="$2"; shift 2;;
        --top-k)       TOP_K="$2"; shift 2;;
        --queue)       QUEUE="$2"; shift 2;;
        --gputype)     GPUTYPE="$2"; shift 2;;
        --mem)         MEM="$2"; shift 2;;
        --dir-suffix)  DIR_SUFFIX="$2"; shift 2;;
        --dry-run|-n)  DRY_RUN=1; shift;;
        -h|--help)     sed -n '2,22p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

PRESET=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['preset'])")
LAYER=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['layer'])")
SAE_ID=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['sae_id'])")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPT_ROOT="${REPO_ROOT}/data/cherry_picked_feature_entropy"
EXPT_DIR="${EXPT_ROOT}/${TIMESTAMP}_${DIR_SUFFIX}"
mkdir -p "$EXPT_DIR" "${REPO_ROOT}/out"

GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
HOST=$(hostname)

# Manifest first so partial failures still leave a self-describing dir.
/usr/bin/python3 - "$YAML" "$EXPT_DIR/manifest.json" \
    "$TIMESTAMP" "$CTX_LEN" "$TOP_K" "$GIT_COMMIT" "$HOST" \
    "$QUEUE" "$GPUTYPE" "$MEM" <<'PY'
import json, sys, yaml
yaml_path, manifest_path, ts, ctx_len, top_k, git_commit, host, \
    queue, gputype, mem = sys.argv[1:]
cfg = yaml.safe_load(open(yaml_path))
manifest = {
    "experiment": "cherry_picked_feature_entropy",
    "variant": "neuronpedia",
    "timestamp": ts,
    "git_commit": git_commit,
    "host": host,
    "preset": cfg["preset"],
    "layer": cfg["layer"],
    "sae_id": cfg["sae_id"],
    "ctx_len": int(ctx_len),
    "top_k": int(top_k),
    "corpus": "monology/pile-uncopyrighted (via Neuronpedia activations)",
    "source": "data/neuronpedia_cache/<model>/<sae_id>/<fid>.json",
    "queue": queue, "gputype": gputype or None,
    "mem_per_core_gb": int(mem),
    "features": cfg["features"],
}
json.dump(manifest, open(manifest_path, "w"), indent=2)
print(f"[manifest] wrote {manifest_path}")
PY

ln -sfn "$(basename ${EXPT_DIR})" "${EXPT_ROOT}/latest_${DIR_SUFFIX}"

JSON_PATH="${EXPT_DIR}/top_contexts_layer${LAYER}.json"
CORPUS_PATH="${EXPT_DIR}/corpus_tokens.pt"

echo "================================================================"
echo "Cherry-picked feature-entropy experiment (Neuronpedia variant)"
echo "  expt_dir : ${EXPT_DIR}"
echo "  preset   : ${PRESET}    layer: ${LAYER}    sae_id: ${SAE_ID}"
echo "  ctx_len  : ${CTX_LEN}    top_k: ${TOP_K}"
echo "  queue    : ${QUEUE}${GPUTYPE:+/$GPUTYPE}    mem: ${MEM} GB/core"
echo "================================================================"

# ---- Step A: adapter (local CPU, instant) ----------------------------------
echo
echo "---- Step A: neuronpedia_to_top_contexts.py (local CPU) ----"
ADAPTER_CMD=(
    /usr/bin/env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
    /usr/bin/python3
    "${REPO_ROOT}/scripts/utils/neuronpedia_to_top_contexts.py"
    --yaml "${YAML}"
    --output-dir "${EXPT_DIR}"
    --top-k "${TOP_K}"
)
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would invoke:"
    printf '    %q' "${ADAPTER_CMD[@]}"; echo
else
    "${ADAPTER_CMD[@]}"
fi

# ---- Step B: verification (sondhigpu, ~30 sec) -----------------------------
echo
echo "---- Step B: verify_neuronpedia_match.py ----"
VERIFY_OUT="${REPO_ROOT}/out/cpfe_${TIMESTAMP}_neuronpedia_verify_%j.out"
VERIFY_PYCMD=(
    /usr/bin/env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
    stdbuf -oL -eL
    /usr/bin/python3
    "${REPO_ROOT}/scripts/analysis/verify_neuronpedia_match.py"
    --yaml "${YAML}"
    --top-contexts "${JSON_PATH}"
    --corpus-tensor "${CORPUS_PATH}"
)
GPUTYPE_ARGS=()
[[ -n "$GPUTYPE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE")
VERIFY_CMD=(
    addqueue
    -c "cpfe ${TIMESTAMP} neuronpedia verify (q=${QUEUE})"
    -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
    -s -n 2 -m "$MEM"
    -o "$VERIFY_OUT"
    --sbatch
    "${VERIFY_PYCMD[@]}"
)
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would invoke:"
    printf '    %q' "${VERIFY_CMD[@]}"; echo
    VERIFY_JOBID="<verify_jobid>"
else
    VERIFY_OUTPUT=$("${VERIFY_CMD[@]}")
    echo "$VERIFY_OUTPUT"
    VERIFY_JOBID=$(echo "$VERIFY_OUTPUT" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -z "$VERIFY_JOBID" ]]; then
        echo "[ERROR] could not parse verify job id" >&2
        exit 3
    fi
    echo "[submit] verify jobid=${VERIFY_JOBID}"
fi

# ---- Step C: 4 parallel tier entropy jobs (sondhigpu, --runafter verify) ---
echo
echo "---- Step C: topact_entropy.py per tier (4 parallel jobs) ----"
for TIER in "${TIERS[@]}"; do
    OUT="${REPO_ROOT}/out/cpfe_${TIMESTAMP}_neuronpedia_${TIER}_%j.out"
    PYCMD=(
        /usr/bin/env
        PYTHONUNBUFFERED=1
        PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
        stdbuf -oL -eL
        /usr/bin/python3
        "${REPO_ROOT}/scripts/analysis/topact_entropy.py"
        --yaml "${YAML}"
        --top-contexts "${JSON_PATH}"
        --corpus-tensor "${CORPUS_PATH}"
        --output-dir "${EXPT_DIR}"
        --ctx-len "${CTX_LEN}"
        --top-k "${TOP_K}"
        --tier-filter "${TIER}"
    )
    CMD=(
        addqueue
        -c "cpfe ${TIMESTAMP} neuronpedia ${TIER} ctx${CTX_LEN}"
        -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
        -s -n 2 -m "$MEM"
        -o "$OUT"
        --runafter "${VERIFY_JOBID}"
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
echo "All jobs issued. Verify will gate the four parallel entropy jobs."
echo "Watch:  q   |   tail -f out/cpfe_${TIMESTAMP}_neuronpedia_*.out"
echo "Results: ${EXPT_DIR}"
echo "================================================================"
