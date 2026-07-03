#!/bin/bash
# Chunked B1 driver: each chunk is a fresh process that os._exit()s, freeing the
# mem0/chroma leaked threads before the next chunk -> avoids thread exhaustion.
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1

CHUNK=${CHUNK:-8}
TOTAL=${TOTAL:-40}
rm -f results/b1_part_*.json
rm -rf /tmp/p7_b1
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== chunk offset=$off ==="
  P7_N_B1=$CHUNK P7_OFFSET=$off P7_WORKERS_B1=2 P7_N_CONSOLID=2 P7_PART=1 \
    P7_CHROMA_B1=/tmp/p7_b1_$off \
    "$P" b1_ladder.py
  rm -rf /tmp/p7_b1_$off
done
echo "=== merging ==="
"$P" merge_b1.py
echo "=== B1 DONE ==="
