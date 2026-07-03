#!/bin/bash
# B1 ladder on the 2nd backbone (Llama-3.1-8B @ :8103) -> cross-family generality.
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1
export P7_VLLM_URL=http://localhost:8103/v1 P7_MODEL=Llama-3.1-8B P7_TAG=_llama P7_CUSTOM_EXTRACT=1
CHUNK=${CHUNK:-8}; TOTAL=${TOTAL:-24}
rm -f results/b1_llama_part_*.json; rm -rf /tmp/p7_b1l_*
for ((off=0; off<TOTAL; off+=CHUNK)); do
  echo "=== B1-llama chunk offset=$off ==="
  P7_N_B1=$CHUNK P7_OFFSET=$off P7_WORKERS_B1=2 P7_N_CONSOLID=2 P7_PART=1 \
    P7_CHROMA_B1=/tmp/p7_b1l_$off "$P" b1_ladder.py
  rm -rf /tmp/p7_b1l_$off
done
echo "=== merging B1-llama ==="; P7_TAG=_llama "$P" merge_b1.py; echo "=== B1-llama DONE ==="
