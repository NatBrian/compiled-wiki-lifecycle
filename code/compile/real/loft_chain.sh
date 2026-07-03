#!/usr/bin/env bash
# LOFT accuracy-at-scale: dump (256K) + vector_rag (14B) on nq/hotpotqa at 32k/128k.
set -u
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export AGENT_MODEL=qwen2.5-14b AGENT_URL=http://127.0.0.1:8101/v1
export DUMP_MODEL=qwen2.5-14b DUMP_URL=http://127.0.0.1:8101/v1
PY=real/venvs/lightrag/bin/python
say(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
for ds in nq hotpotqa; do for sc in 32k; do for arm in vector_rag full_dump_cic; do
  out="results/results_loft_${arm}_${ds}_${sc}.json"
  [ -f "$out" ] && { say "loft $arm $ds $sc DONE"; continue; }
  for attempt in 1 2 3; do
    say "loft $arm $ds $sc attempt $attempt"
    $PY real/loft_run.py "$arm" "$ds" "$sc" >> logs/loft.log 2>&1
    [ -f "$out" ] && break; sleep 5
  done
done; done; done
say "### loft_chain complete ###"
