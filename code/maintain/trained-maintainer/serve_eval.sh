#!/usr/bin/env bash
# Paper-4 v2 eval servers on MY GPUs (6,7) only. Brings up:
#   :8102 (GPU6)  14B base + 4 LoRA adapters
#   :8103 (GPU7)  14B base + 4 LoRA adapters  (throughput: 2 servers)
#   :8106 (GPU7)  Qwen2.5-1.5B base           (maintainer-scale check, via --maint_port 8106)
# Detached (setsid) so a Claude crash can't kill them. NEVER touches GPUs 0-5 (yongyue).
set -u
cd "$(dirname "$0")"   # bundled repo: scripts live in this dir directly
mkdir -p logs
ADAPTERS="b1=results/lora_b1_preserve \
          maintainer-lora=/path/to/your/lora_maintainer_adapter \
          tight-f50=results/lora_probe_tight_f50 \
          tight-f25=results/lora_probe_tight_f25"

caps() { export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
                OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2; }

launch_14b () { # gpu port util
  caps
  CUDA_VISIBLE_DEVICES=$1 setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct --served-model-name qwen2.5-14b \
    --enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules $ADAPTERS \
    --tensor-parallel-size 1 --gpu-memory-utilization $3 \
    --max-model-len 16384 --port $2 --host 127.0.0.1 \
    >> logs/vllm_$2.log 2>&1 < /dev/null &
  echo $! > logs/vllm_$2.pid; echo "14B GPU$1 :$2 util$3 pid $(cat logs/vllm_$2.pid)"
}

launch_1p5b () { # gpu port util
  caps
  CUDA_VISIBLE_DEVICES=$1 setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --served-model-name qwen2.5-1.5b \
    --tensor-parallel-size 1 --gpu-memory-utilization $3 \
    --max-model-len 16384 --port $2 --host 127.0.0.1 \
    >> logs/vllm_$2.log 2>&1 < /dev/null &
  echo $! > logs/vllm_$2.pid; echo "1.5B GPU$1 :$2 util$3 pid $(cat logs/vllm_$2.pid)"
}

launch_14b 6 8102 0.55
launch_14b 7 8103 0.45
launch_1p5b 7 8106 0.12
echo "launched; poll: until curl -s http://127.0.0.1:8102/v1/models|grep -q tight-f50; do sleep 3; done"
