#!/usr/bin/env bash
# Resilient HoH n=300 runner for the real systems. Each driver checkpoints every
# item (results/_ckpt_<tag>.jsonl) and resumes; the host SIGKILLs long python
# ~every 1.5h, so we retry each until its final results_<tag>.json exists. Serial
# (one system at a time) to avoid CPU-embedder thrash on the shared host.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export AGENT_MODEL=qwen2.5-14b AGENT_URL=http://127.0.0.1:8101/v1 EMBEDDING_DIM=384 MEM0_TELEMETRY=false
N="${1:-300}"
say(){ echo "$(date '+%m-%d %H:%M:%S') $*"; }
vllm_ok(){ curl -sf --max-time 5 "$AGENT_URL/models" >/dev/null 2>&1; }

run(){
  local sys="$1" venv="$2" script="$3" tag="real_${1}_hoh"
  local out="results/results_${tag}.json"
  for attempt in $(seq 1 30); do
    [ -f "$out" ] && { say "$sys: DONE"; return 0; }
    if ! vllm_ok; then say "$sys: vLLM down, wait"; sleep 60; continue; fi
    local n_done=0; [ -f "results/_ckpt_${tag}.jsonl" ] && n_done=$(wc -l < "results/_ckpt_${tag}.jsonl")
    say "$sys: attempt $attempt (resume from $n_done)"
    "real/venvs/$venv/bin/python" "real/$script" "$N" >> "logs/hoh_${sys}.log" 2>&1
    [ -f "$out" ] && { say "$sys: DONE (attempt $attempt)"; return 0; }
    say "$sys: exited w/o final result, retry"; sleep 10
  done
  say "$sys: GAVE UP"; return 1
}

say "### HoH chain start (N=$N) ###"
run lightrag lightrag hoh_lightrag.py
run mem0     mem0     hoh_mem0.py
run raptor   raptor   hoh_raptor.py
say "### HoH chain complete ###"
