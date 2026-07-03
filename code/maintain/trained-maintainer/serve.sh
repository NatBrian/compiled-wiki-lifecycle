#!/usr/bin/env bash
# (Re)launch the Paper-4 vLLM server on MY GPUs (6,7) only. Serves Qwen2.5-14B base plus any
# LoRA adapters passed as name=path pairs. Detached (setsid) so it survives Claude crashes.
#
#   ./serve.sh [name=path ...]
# e.g. ./serve.sh maintainer-lora=/path/to/your/lora_maintainer_adapter \
#                  b1-trained=results/lora_b1_preserve
#
# NEVER touch GPUs 0-5 (another user). Kills only MY existing :8102 server first.
set -u
cd "$(dirname "$0")"   # bundled repo: scripts live in this dir directly
mkdir -p logs
PORT=8102
GPUS="${P4_GPUS:-6,7}"
TP="${P4_TP:-2}"

# kill my existing server on :8102 (mine only — match the api_server cmdline + my uid)
OLD=$(pgrep -u "$(id -u)" -f "vllm.entrypoints.openai.api_server.*port $PORT" || true)
if [ -n "$OLD" ]; then echo "killing my old server pids: $OLD"; kill $OLD 2>/dev/null; sleep 8; fi

LORA_ARGS=""
if [ "$#" -gt 0 ]; then LORA_ARGS="--enable-lora --max-lora-rank 16 --lora-modules $*"; fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2
setsid nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct --served-model-name qwen2.5-14b \
  $LORA_ARGS \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --port "$PORT" --host 127.0.0.1 \
  >> logs/vllm_8102.log 2>&1 < /dev/null &
echo $! > logs/vllm_8102.pid
echo "vLLM (GPUs $GPUS, TP=$TP) starting pid $(cat logs/vllm_8102.pid); adapters: ${*:-none}"
echo "wait for readiness: until curl -s http://127.0.0.1:$PORT/v1/models >/dev/null; do sleep 3; done"
