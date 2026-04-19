#!/bin/bash
set -e
cd /mnt/users/clin/workspace/sae-analysis/sae-analysis-v1
echo "[INFO] Installing sparsify..."
/usr/bin/pip install --quiet --user sparsify 2>&1 | tail -5
echo "[INFO] sparsify installed:"
/usr/bin/python3 -c "import sparsify; print(sparsify.__version__)"
echo "[INFO] Starting compare_entropies_multi_layer.py on llama-3-8b..."
exec /usr/bin/python3 scripts/analysis/compare_entropies_multi_layer.py \
    --preset llama-3-8b --num-batches 50
