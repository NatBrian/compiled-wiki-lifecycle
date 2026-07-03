#!/bin/bash
# B2 make-or-break: compile vs dump vs rag, N-sweep on SciFact-Open candidates.
# Uses GPU passed as $1 (CUDA index on host). Honors non-disruption: caller verified free.
set -e
GPU="${1:-2}"; MODEL="${2:-Qwen/Qwen2.5-1.5B-Instruct}"
cd "$(dirname "$0")"
mkdir -p ../results
for N in 500 2000 8000 12236; do
  echo "=================== N=$N ==================="
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    python -u run_scifact_absence.py \
      --data ../benchmarks/scifact-open/data --corpus corpus_candidates.jsonl \
      --N "$N" --n_claims 50 --methods rag,dump,compile \
      --device cuda:0 --model "$MODEL" \
      --out ../results/b2_N${N}.json 2>&1 | grep -vE "Loading weights|FutureWarning|warnings.warn|torch_dtype|stable_cumsum"
done
echo "=================== SWEEP DONE ==================="
for N in 500 2000 8000 12236; do
  python - "$N" <<PY
import json,sys
N=sys.argv[1]; d=json.load(open(f"../results/b2_N{N}.json"))["summary"]
print(f"N={N:>6}  "+"  ".join(f"{m}:fa={d[m]['false_absence_rate']:.2f},tok={d[m]['avg_query_ctx_tokens']:.0f}" for m in ['rag','dump','compile']))
PY
done
