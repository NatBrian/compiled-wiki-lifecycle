#!/bin/bash
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1
CHUNK=${CHUNK:-8}; TOTAL=${TOTAL:-24}
rm -f results/b2_part_*.json; rm -rf /tmp/p7_b2_*
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== B2 chunk offset=$off ==="
  P7_N_B2=$CHUNK P7_OFFSET=$off P7_WORKERS_B2=2 P7_N_CONSOLID=2 P7_PART=1 \
    P7_CHROMA_B2=/tmp/p7_b2_$off "$P" b2_ablation.py
  rm -rf /tmp/p7_b2_$off
done
echo "=== merging B2 ==="
"$P" merge_b2.py
echo "=== B2 DONE ==="
