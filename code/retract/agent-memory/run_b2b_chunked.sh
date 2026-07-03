#!/bin/bash
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1
CHUNK=${CHUNK:-10}; TOTAL=${TOTAL:-30}
rm -f results/b2b_part_*.json; rm -rf /tmp/p7_b2b_*
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== B2b chunk offset=$off ==="
  P7_N_B2B=$CHUNK P7_OFFSET=$off P7_WORKERS_B2B=2 P7_PART=1 \
    P7_CHROMA_B2B=/tmp/p7_b2b_$off "$P" b2b_necessity.py
  rm -rf /tmp/p7_b2b_$off
done
echo "=== merging B2b ==="
"$P" merge_b2b.py
echo "=== B2b DONE ==="
