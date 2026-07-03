#!/usr/bin/env bash
# Serialized runner for the real-system baselines. They share ONE vLLM reader
# (GPU 4) + CPU embedders, so we run them one at a time to avoid contention and
# the host's under-load SIGKILL. Each writes results/results_real_<sys>_<lib>.json.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export AGENT_MODEL=qwen2.5-14b AGENT_URL=http://127.0.0.1:8101/v1
LIB="${1:-pydantic}"
LOG=logs
say(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }

run(){
  local sys="$1" venv="$2" script="$3"
  local out="results/results_real_${sys}_${LIB}.json"
  if [ -f "$out" ]; then say "$sys: already done ($out)"; return 0; fi
  say "### $sys ($LIB) start ###"
  "real/venvs/$venv/bin/python" "real/$script" "$LIB" > "$LOG/run_real_${sys}_${LIB}.log" 2>&1
  if [ -f "$out" ]; then say "$sys: DONE"; else say "$sys: FAILED (see $LOG/run_real_${sys}_${LIB}.log)"; fi
}

run lightrag lightrag run_lightrag.py
run mem0     mem0     run_mem0.py

say "### real chain ($LIB) complete ###"
