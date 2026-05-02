#!/usr/bin/env bash
# Cherry-picked feature-entropy experiment.
#
# Step 0:  one GPU job runs top_activating_contexts.py on all 20 features
#          from the pilot YAML; writes top_contexts_layer<L>.json.
# Step 1:  four parallel GPU jobs (one per tier: token / phrase / concept /
#          abstract) each run topact_entropy.py against the same JSON,
#          writing one CSV + one NPZ per tier into the experiment dir.
#
# Step 1 jobs are submitted with `--runafter <step0_jobid>` so they kick off
# the moment Step 0 completes.
#
# Outputs land under
#   data/cherry_picked_feature_entropy/<TIMESTAMP>/
#       manifest.json
#       top_contexts_layer<L>.json   # from Step 0
#       ctx<N>_<tier>.csv            # one per tier from Step 1
#       influences_<tier>.npz        # one per tier from Step 1
# and  data/cherry_picked_feature_entropy/latest -> <TIMESTAMP>
#
# Per the user's queue-routing rule: gpulong + rtx4090with24gb by default.
# 5 GPUs needed in total (1 for Step 0 + 4 parallel for Step 1); current
# `showgpus` shows >5 free rtx4090/rtx3090 slots.
#
# Usage:
#   scripts/experiments/submit_cherry_picked_entropy.sh \
#       [--yaml scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml] \
#       [--ctx-len 64] [--top-k 8] [--top-contexts-window 256] \
#       [--max-batches 4000] \
#       [--queue gpulong] [--gputype rtx4090with24gb] [--mem 8] \
#       [--dry-run]

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

# ---- defaults --------------------------------------------------------------
YAML="${REPO_ROOT}/scripts/experiments/configs/pilot_features_gemma2_2b_l12.yaml"
CTX_LEN=64
TOP_K=16
TOP_CONTEXTS_WINDOW=256   # window length used by top_activating_contexts.py;
                          # must be >= CTX_LEN so a peak near the end of the
                          # window has enough preceding tokens in the corpus.
MAX_BATCHES=4000
QUEUE="gpulong"
GPUTYPE="rtx4090with24gb"
MEM=8
DRY_RUN=0
TIERS=(token phrase concept abstract)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)                 YAML="$2"; shift 2;;
        --ctx-len)              CTX_LEN="$2"; shift 2;;
        --top-k)                TOP_K="$2"; shift 2;;
        --top-contexts-window)  TOP_CONTEXTS_WINDOW="$2"; shift 2;;
        --max-batches)          MAX_BATCHES="$2"; shift 2;;
        --queue)                QUEUE="$2"; shift 2;;
        --gputype)              GPUTYPE="$2"; shift 2;;
        --mem)                  MEM="$2"; shift 2;;
        --dry-run|-n)           DRY_RUN=1; shift;;
        -h|--help)              sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ ! -f "$YAML" ]]; then
    echo "ERROR: YAML not found: $YAML" >&2; exit 2
fi

# ---- introspect YAML -------------------------------------------------------
PRESET=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['preset'])")
LAYER=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['layer'])")
SAE_ID=$(/usr/bin/python3 -c "import yaml; print(yaml.safe_load(open('${YAML}'))['sae_id'])")
FEATURE_IDS=$(/usr/bin/python3 -c "
import yaml
cfg = yaml.safe_load(open('${YAML}'))
print(' '.join(str(int(e['feature_id'])) for e in cfg['features']))
")
NFEAT=$(echo "$FEATURE_IDS" | wc -w | tr -d ' ')

# ---- timestamped experiment dir + manifest ---------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPT_ROOT="${REPO_ROOT}/data/cherry_picked_feature_entropy"
EXPT_DIR="${EXPT_ROOT}/${TIMESTAMP}"
mkdir -p "$EXPT_DIR" "${REPO_ROOT}/out"

GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
HOST=$(hostname)

# Use python to dump manifest (preserves the YAML feature spec verbatim).
/usr/bin/python3 - "$YAML" "$EXPT_DIR/manifest.json" \
    "$TIMESTAMP" "$CTX_LEN" "$TOP_K" "$TOP_CONTEXTS_WINDOW" \
    "$MAX_BATCHES" "$GIT_COMMIT" "$HOST" \
    "$QUEUE" "$GPUTYPE" "$MEM" <<'PY'
import json, sys, yaml
yaml_path, manifest_path, ts, ctx_len, top_k, win, max_batches, \
    git_commit, host, queue, gputype, mem = sys.argv[1:]
cfg = yaml.safe_load(open(yaml_path))
manifest = {
    "experiment": "cherry_picked_feature_entropy",
    "timestamp": ts,
    "git_commit": git_commit,
    "host": host,
    "preset": cfg["preset"],
    "layer": cfg["layer"],
    "sae_id": cfg["sae_id"],
    "ctx_len": int(ctx_len),
    "top_k": int(top_k),
    "top_contexts_window": int(win),
    "max_batches": int(max_batches),
    "corpus": "data_loader.load_wikitext_train_text",
    "queue": queue, "gputype": gputype, "mem_per_core_gb": int(mem),
    "features": cfg["features"],
}
json.dump(manifest, open(manifest_path, "w"), indent=2)
print(f"[manifest] wrote {manifest_path}")
PY

# Update latest symlink.
ln -sfn "$TIMESTAMP" "${EXPT_ROOT}/latest"

JSON_PATH="${EXPT_DIR}/top_contexts_layer${LAYER}.json"

echo "================================================================"
echo "Cherry-picked feature-entropy experiment"
echo "  expt_dir : ${EXPT_DIR}"
echo "  preset   : ${PRESET}    layer: ${LAYER}    sae_id: ${SAE_ID}"
echo "  features : ${NFEAT} from ${YAML}"
echo "  ctx_len  : ${CTX_LEN}    top_k: ${TOP_K}"
echo "  step0_window: ${TOP_CONTEXTS_WINDOW}    max_batches: ${MAX_BATCHES}"
echo "  queue    : ${QUEUE}/${GPUTYPE}    mem: ${MEM} GB/core"
echo "================================================================"

# ---- Step 0 submission -----------------------------------------------------
STEP0_OUT="${REPO_ROOT}/out/cpfe_${TIMESTAMP}_step0_topctx_%j.out"
STEP0_PYCMD=(
    /usr/bin/env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}/scripts/analysis:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
    stdbuf -oL -eL
    /usr/bin/python3
    "${REPO_ROOT}/scripts/analysis/top_activating_contexts.py"
    --preset "${PRESET}" --layer "${LAYER}"
    --top-k "${TOP_K}"
    --context-len "${TOP_CONTEXTS_WINDOW}"
    --max-batches "${MAX_BATCHES}"
    --output-dir "${EXPT_DIR}"
    --feature-ids ${FEATURE_IDS}
)
GPUTYPE_ARGS=()
[[ -n "$GPUTYPE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE")
STEP0_CMD=(
    addqueue
    -c "cpfe ${TIMESTAMP} step0 topctx (q=${QUEUE}/${GPUTYPE})"
    -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
    -s -n 2 -m "$MEM"
    -o "$STEP0_OUT"
    --sbatch
    "${STEP0_PYCMD[@]}"
)

echo
echo "---- Step 0: top_activating_contexts.py ----"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would invoke:"
    printf '    %q' "${STEP0_CMD[@]}"; echo
    STEP0_JOBID="<step0_jobid>"
else
    STEP0_OUTPUT=$("${STEP0_CMD[@]}")
    echo "$STEP0_OUTPUT"
    STEP0_JOBID=$(echo "$STEP0_OUTPUT" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -z "$STEP0_JOBID" ]]; then
        echo "[ERROR] could not parse step 0 job id from addqueue output" >&2
        exit 3
    fi
    echo "[submit] step 0 jobid=${STEP0_JOBID}"
fi

# ---- Step 1 submissions (one per tier) -------------------------------------
echo
echo "---- Step 1: topact_entropy.py per tier (4 parallel jobs) ----"
for TIER in "${TIERS[@]}"; do
    OUT="${REPO_ROOT}/out/cpfe_${TIMESTAMP}_step1_${TIER}_%j.out"
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
        --ctx-len "${CTX_LEN}"
        --top-k "${TOP_K}"
        --tier-filter "${TIER}"
    )
    CMD=(
        addqueue
        -c "cpfe ${TIMESTAMP} step1 ${TIER} ctx${CTX_LEN}"
        -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
        -s -n 2 -m "$MEM"
        -o "$OUT"
        --runafter "${STEP0_JOBID}"
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
echo "All submissions issued. Watch with:"
echo "  q"
echo "  tail -f out/cpfe_${TIMESTAMP}_*.out"
echo "Results land under: ${EXPT_DIR}"
echo "================================================================"
