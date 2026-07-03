#!/usr/bin/env bash
# Single-system HoH runner with retry-until-done (resumes from checkpoint).
# Usage: hoh_one.sh <sys> <venv> <script> <N>
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
export AGENT_MODEL=qwen2.5-14b AGENT_URL=http://127.0.0.1:8101/v1 EMBEDDING_DIM=384 MEM0_TELEMETRY=false
sys="$1" venv="$2" script="$3" N="${4:-300}"
tag="real_${sys}_hoh"; out="results/results_${tag}.json"
say(){ echo "$(date '+%m-%d %H:%M:%S') [$sys] $*"; }
vllm_ok(){ curl -sf --max-time 5 "$AGENT_URL/models" >/dev/null 2>&1; }
for attempt in $(seq 1 40); do
  [ -f "$out" ] && { say "DONE"; exit 0; }
  if ! vllm_ok; then say "vLLM down, wait"; sleep 60; continue; fi
  nd=0; [ -f "results/_ckpt_${tag}.jsonl" ] && nd=$(wc -l < "results/_ckpt_${tag}.jsonl")
  say "attempt $attempt (resume from $nd)"
  "real/venvs/$venv/bin/python" "real/$script" "$N" >> "logs/hoh_${sys}.log" 2>&1
  [ -f "$out" ] && { say "DONE attempt $attempt"; exit 0; }
  say "exited w/o final, retry"; sleep 8
done
say "GAVE UP"; exit 1
