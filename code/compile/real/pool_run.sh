#!/usr/bin/env bash
# Retry-until-done runner for pooled-HoH arms. Each arm checkpoints per item
# (results/_ckpt_pool_pool_<arm>.jsonl) and resumes. Host may SIGKILL long python.
# Usage: pool_run.sh <N> <arm1> [arm2 ...]
set -u
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export AGENT_MODEL=qwen2.5-14b AGENT_URL=http://127.0.0.1:8101/v1
export DUMP_MODEL=qwen3-coder-30b DUMP_URL=http://127.0.0.1:8102/v1
N="$1"; shift
PY=real/venvs/lightrag/bin/python
say(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
for arm in "$@"; do
  out="results/results_pool_${arm}.json"
  for attempt in $(seq 1 40); do
    [ -f "$out" ] && { say "$arm: DONE"; break; }
    say "$arm: attempt $attempt"
    $PY real/pooled_hoh.py "$arm" "$N" >> "logs/pool_${arm}.log" 2>&1
    [ -f "$out" ] && { say "$arm: DONE (attempt $attempt)"; break; }
    say "$arm: exited w/o final, retry"; sleep 6
  done
done
say "### pool_run complete: $* ###"
