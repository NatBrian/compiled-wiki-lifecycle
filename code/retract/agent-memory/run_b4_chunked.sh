#!/bin/bash
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1
CHUNK=${CHUNK:-10}; TOTAL=${TOTAL:-30}
rm -f results/b4_part_*.json; rm -rf /tmp/p7_b4_*
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== B4 chunk offset=$off ==="
  P7_N_B4=$CHUNK P7_OFFSET=$off P7_WORKERS_B4=2 P7_N_CONSOLID=2 P7_PART=1 \
    P7_CHROMA_B4=/tmp/p7_b4_$off "$P" b4_attack.py
  rm -rf /tmp/p7_b4_$off
done
echo "=== merging B4 ==="; "$P" merge_b4.py; echo "=== B4 DONE ==="
