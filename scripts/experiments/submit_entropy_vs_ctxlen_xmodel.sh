#!/usr/bin/env bash
# Cross-model, one-prompt entropy_vs_context_len sweep.
#
# For a single seed, runs entropy_vs_context_len.py on every preset's default
# layer set. Because the analysis script samples its source window in
# CHARACTER space (not token space), the same --random-seed gives every
# preset the same WikiText character window, which each tokenizer renders
# independently into its own token stream. That's the "one prompt across
# models" invariant.
#
# One addqueue job per preset (default sondhigpu/H200). Inside each job a
# shell loop iterates layers, paying the per-layer model-reload cost from
# the OS page cache (~1-30 s per layer depending on model size; ~12 min
# total across all six presets — negligible vs. compute).
#
# Usage:
#   scripts/experiments/submit_entropy_vs_ctxlen_xmodel.sh \
#       [--seed N] [--max-context-len 128] [--min-context-len 8] [--step 8] \
#       [--smoke] [--dry-run] [--queue Q] [--mem GB] [--presets "p1 p2 ..."]
#
# --smoke runs only pythia-70m on layers (0 3 5), seed 0, context 32..64.

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

# ---- Defaults / arg parsing ------------------------------------------------
# Per-preset max-context defaults are set by preset_max_ctx() / preset_step()
# below. CLI --max-context-len / --step force a uniform override for all
# presets (useful for smoke tests and ad-hoc reruns).
SEED=42
MAX_CTX=""
MIN_CTX=8
STEP=""
SMOKE=0
DRY_RUN=0
QUEUE_OVERRIDE=""
MEM_OVERRIDE=""
PRESETS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)             SEED="$2"; shift 2;;
        --max-context-len)  MAX_CTX="$2"; shift 2;;
        --min-context-len)  MIN_CTX="$2"; shift 2;;
        --step)             STEP="$2"; shift 2;;
        --smoke)            SMOKE=1; shift;;
        --dry-run|-n)       DRY_RUN=1; shift;;
        --queue)            QUEUE_OVERRIDE="$2"; shift 2;;
        --mem)              MEM_OVERRIDE="$2"; shift 2;;
        --presets)          PRESETS_OVERRIDE="$2"; shift 2;;
        -h|--help)          sed -n '2,21p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ $SMOKE -eq 1 ]]; then
    SEED=0
    MAX_CTX=64
    MIN_CTX=32
    STEP=16
    PRESETS_OVERRIDE="${PRESETS_OVERRIDE:-pythia-70m}"
fi

if [[ -n "$PRESETS_OVERRIDE" ]]; then
    # shellcheck disable=SC2206
    PRESETS=($PRESETS_OVERRIDE)
else
    PRESETS=(pythia-70m qwen2-0.5b gpt2-small llama-3.2-1b gemma-2-2b llama-3-8b)
fi

# ---- Per-preset queue / mem / layer / context table -----------------------
# Routing policy: prioritise jobs starting immediately. Small models go to
# kocsisgpu (rtx2080-12gb, frequently fully idle, plenty for ≤1B-param fp32
# models). Memory-demanding models (gemma-2-2b, llama-3-8b) go to heraclesgpu
# which has rtx6000ada-48gb and a100-80gb GPUs.
preset_queue() {
    case "$1" in
        gemma-2-2b|llama-3-8b)  echo "heraclesgpu" ;;
        *)                       echo "kocsisgpu" ;;
    esac
}
preset_gputype() {
    # kocsisgpu has only rtx2080with12gb (no --gputype needed).
    # heraclesgpu mixes rtx6000ada-48gb and a100-80gb — pin per model:
    #   gemma-2-2b (~10 GB fp32) -> rtx6000ada (saves a100 for the bigger job)
    #   llama-3-8b (~32 GB fp32) -> a100with80gb
    case "$1" in
        llama-3-8b)    echo "a100with80gb" ;;
        gemma-2-2b)    echo "rtx6000adawith48gb" ;;
        *)             echo "" ;;
    esac
}
preset_mem() {
    # Per-CORE host memory in GB; addqueue runs with -n 2 cores so total is
    # 2× this. kocsisgpu nodes have ~123 GB / 32 cores = 3.8 GB/core; -n 2
    # -m 8 gives 16 GB total, which is plenty (model lives on GPU).
    # heraclesgpu has ~117 GB / 48 cores; keep total request reasonable so
    # we don't block other users on the same node.
    case "$1" in
        pythia-70m)    echo 8 ;;
        gpt2-small)    echo 8 ;;
        qwen2-0.5b)    echo 8 ;;
        llama-3.2-1b)  echo 12 ;;
        gemma-2-2b)    echo 16 ;;   # heraclesgpu, total 32 GB
        llama-3-8b)    echo 32 ;;   # heraclesgpu, total 64 GB (host RAM for fp32 model load)
        *)             echo 12 ;;
    esac
}
preset_max_ctx() {
    # Per-preset max-context default. Caps at the architectural max where
    # that's < 2048 (gpt2-small=1024). Other models support much longer
    # contexts (Qwen/Llama-3 reach 32k+) but per-feature gradient cost grows
    # linearly with context, so 2048 is a pragmatic ceiling that gives ~16×
    # the previous run-time. Override with --max-context-len for all presets.
    case "$1" in
        gpt2-small)    echo 1024 ;;
        *)             echo 2048 ;;
    esac
}
preset_step() {
    # Default step = max_ctx / 16, giving ~16-17 sweep points across the
    # context-length axis (matching the resolution of the original 8..128/8
    # sweep).
    case "$1" in
        gpt2-small)    echo 64 ;;
        *)             echo 128 ;;
    esac
}
preset_layers() {
    case "$1" in
        pythia-70m)    echo "0 1 2 3 4 5" ;;
        gpt2-small)    echo "0 1 2 3 4 5 6 7 8 9 10 11" ;;
        qwen2-0.5b)    echo "0 1 2 3 4 5 6 7 8 9 10 11" ;;
        llama-3.2-1b)  echo "0 1 2 3 4 5 6 7 8" ;;
        gemma-2-2b)    seq -s ' ' 0 25 ;;
        llama-3-8b)    echo "0 3 6 9 12 15 18 21 24 27 30" ;;
        *) echo "ERROR: unknown preset $1" >&2; exit 2 ;;
    esac
}

if [[ $SMOKE -eq 1 ]]; then
    # In smoke mode, override layers to a sparse set per preset.
    preset_layers() {
        case "$1" in
            pythia-70m)    echo "0 3 5" ;;
            gpt2-small)    echo "0 5 11" ;;
            qwen2-0.5b)    echo "0 5 11" ;;
            llama-3.2-1b)  echo "0 4 8" ;;
            gemma-2-2b)    echo "0 12 25" ;;
            llama-3-8b)    echo "0 15 30" ;;
        esac
    }
fi

# ---- Run-level setup ------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${REPO_ROOT}/data/ctxlen_xmodel/${TIMESTAMP}"
mkdir -p "$RUN_DIR"
mkdir -p "${REPO_ROOT}/out"

# Write run-level manifest. The actual character window is recoverable from
# (seed, char_budget = max_context_len * 12) against the WikiText loader.
# max_context_len / step are now per-preset; record the per-preset values.
HOST=$(hostname)
GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
PRESET_CFG_LINES=()
for P in "${PRESETS[@]}"; do
    P_MAX="${MAX_CTX:-$(preset_max_ctx "$P")}"
    P_STEP="${STEP:-$(preset_step "$P")}"
    PRESET_CFG_LINES+=("    \"${P}\": {\"max_context_len\": ${P_MAX}, \"step\": ${P_STEP}, \"char_budget\": $((P_MAX * 12))}")
done
PRESET_CFG_JSON=$(IFS=$',\n'; echo "${PRESET_CFG_LINES[*]}")
cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "experiment": "entropy_vs_context_len_xmodel",
  "timestamp": "${TIMESTAMP}",
  "seed": ${SEED},
  "min_context_len": ${MIN_CTX},
  "char_budget_per_max_ctx": 12,
  "loader": "data_loader.load_wikitext_train_text",
  "presets": [$(printf '"%s",' "${PRESETS[@]}" | sed 's/,$//')],
  "preset_config": {
${PRESET_CFG_JSON}
  },
  "host": "${HOST}",
  "git_commit": "${GIT_COMMIT}",
  "smoke": ${SMOKE}
}
EOF

# Update latest symlink at the cross-model namespace level.
ln -sfn "${TIMESTAMP}" "${REPO_ROOT}/data/ctxlen_xmodel/latest"

echo "================================================================"
echo "Cross-model entropy_vs_context_len sweep"
echo "  run_dir: ${RUN_DIR}"
echo "  seed:    ${SEED}    smoke: ${SMOKE}"
echo "  min_ctx: ${MIN_CTX}    (max/step are per-preset; see run_config.json)"
echo "  presets: ${PRESETS[*]}"
echo "================================================================"

# ---- Submit one job per preset --------------------------------------------
for PRESET in "${PRESETS[@]}"; do
    LAYERS=$(preset_layers "$PRESET")
    QUEUE="${QUEUE_OVERRIDE:-$(preset_queue "$PRESET")}"
    MEM="${MEM_OVERRIDE:-$(preset_mem "$PRESET")}"
    GPUTYPE="$(preset_gputype "$PRESET")"
    P_MAX_CTX="${MAX_CTX:-$(preset_max_ctx "$PRESET")}"
    P_STEP="${STEP:-$(preset_step "$PRESET")}"
    PRESET_OUT_DIR="${RUN_DIR}/${PRESET}"
    mkdir -p "$PRESET_OUT_DIR"

    TAG="${PRESET}"
    [[ $SMOKE -eq 1 ]] && TAG="${PRESET}_smoke"
    LOG_FILE="${REPO_ROOT}/out/ctxlen_xmodel_${TAG}_%j.out"
    COMMENT="ctxlen_xmodel ${TAG} seed=${SEED} q=${QUEUE} max=${P_MAX_CTX}"

    # addqueue expects a single executable file (it copies it to <name>.sh
    # at the submitter's CWD), so we write a per-preset run script into the
    # preset's output dir and submit that. PYTHONPATH lets the analysis
    # script import its sibling modules (presets, model_adapters, etc.) by
    # flat module name, matching submit_entropy_bench.sh's convention.
    RUN_SCRIPT="${PRESET_OUT_DIR}/run.sh"
    cat > "$RUN_SCRIPT" <<EOF_RUN
#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH='${REPO_ROOT}/scripts/analysis':\${PYTHONPATH:-}
for L in ${LAYERS}; do
    echo
    echo "=== ${PRESET} layer \$L ==="
    stdbuf -oL -eL /usr/bin/python3 \\
        '${REPO_ROOT}/scripts/analysis/entropy_vs_context_len.py' \\
        --preset '${PRESET}' --layer \$L \\
        --random-seed ${SEED} \\
        --max-context-len ${P_MAX_CTX} \\
        --min-context-len ${MIN_CTX} \\
        --step ${P_STEP} \\
        --output-dir '${PRESET_OUT_DIR}'
done
EOF_RUN
    chmod +x "$RUN_SCRIPT"

    GPUTYPE_ARGS=()
    [[ -n "$GPUTYPE" ]] && GPUTYPE_ARGS=(--gputype "$GPUTYPE")

    ADDQUEUE_CMD=(
        addqueue
        -c "$COMMENT"
        -q "$QUEUE" --gpus 1 "${GPUTYPE_ARGS[@]}"
        -s -n 2 -m "$MEM"
        -o "$LOG_FILE"
        "$RUN_SCRIPT"
    )

    echo
    echo "---- ${PRESET} (q=${QUEUE}${GPUTYPE:+/$GPUTYPE} mem=${MEM}G ctx=${MIN_CTX}..${P_MAX_CTX}/${P_STEP} layers=(${LAYERS})) ----"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] would invoke:"
        printf '    %q' "${ADDQUEUE_CMD[@]}"; echo
    else
        if ! "${ADDQUEUE_CMD[@]}"; then
            echo "[WARN] submission failed for ${PRESET}" >&2
        fi
    fi
done

echo
echo "================================================================"
echo "All ${#PRESETS[@]} preset submissions issued."
echo "Run dir: ${RUN_DIR}"
echo "Logs:    ${REPO_ROOT}/out/ctxlen_xmodel_*.out"
echo "Watch:   q   |   tail -f out/ctxlen_xmodel_<preset>_<jobid>.out"
echo "================================================================"
