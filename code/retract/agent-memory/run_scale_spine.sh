#!/bin/bash
# Scale the paper SPINE to n=250 for tight Wilson CIs (defeats sample-size kill).
#   1) B1 discrimination ladder  (A0..A4 + benign)   -> results/b1_ladder.json
#   2) E0b-AUTO consolidation backflow (A1/A3/A4)     -> results/e0b_auto.json
# Sequential (concurrent drivers exhaust the thread table). Chunked fresh
# processes that os._exit() between chunks -> no mem0/chroma thread leak buildup.
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1

CHUNK=${CHUNK:-10}
TOTAL=${TOTAL:-250}

echo "############ B1 ladder n=$TOTAL ############ $(date)"
rm -f results/b1_part_*.json; rm -rf /tmp/p7_b1_*
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== B1 chunk offset=$off === $(date)"
  P7_N_B1=$CHUNK P7_OFFSET=$off P7_WORKERS_B1=2 P7_N_CONSOLID=2 P7_PART=1 \
    P7_CHROMA_B1=/tmp/p7_b1_$off "$P" b1_ladder.py
  rm -rf /tmp/p7_b1_$off
done
echo "=== merging B1 ==="; "$P" merge_b1.py

echo "############ E0b-AUTO n=$TOTAL ############ $(date)"
rm -f results/e0b_auto_part_*.json; rm -rf /tmp/p7_auto_*
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== AUTO chunk offset=$off === $(date)"
  P7_N_AUTO=$CHUNK P7_OFFSET=$off P7_WORKERS_AUTO=2 P7_N_CONSOLID=2 P7_PART=1 \
    P7_CHROMA_AUTO=/tmp/p7_auto_$off "$P" e0b_auto.py
  rm -rf /tmp/p7_auto_$off
done
echo "=== merging AUTO ==="; "$P" merge_auto.py
echo "############ SCALE SPINE DONE ############ $(date)"
