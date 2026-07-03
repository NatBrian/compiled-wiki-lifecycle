#!/usr/bin/env bash
# E6: judge-free HoH replication. 4 arms x 2 seeds, two queues (one per server).
set -u
cd "$(dirname "$0")" || exit 1   # bundled repo: scripts live in this dir directly
mkdir -p logs results
W=8
COMMON="--n_probe 80 --n_docs0 200 --docs_per_page 8 --batches 12 --batch_fillers 40 --workers $W"
run_one () {  # port qtag arm:seed...
  local port="$1" qtag="$2"; shift 2
  ( for pair in "$@"; do
      arm="${pair%%:*}"; seed="${pair##*:}"; out="results/e6_hoh_${arm}_seed${seed}.json"
      [ -f "$out" ] && { echo "[$qtag] $out exists"; continue; }
      echo "=== [$qtag] HoH $arm seed $seed @:$port $(date) ==="
      for a in 1 2 3; do [ -f "$out" ] && break
        python p5_hoh.py --arm "$arm" --seed "$seed" --port "$port" $COMMON --out "$out"; sleep 8; done
    done; echo "=== [$qtag] E6 DONE $(date) ===" ) >> "logs/e6_${qtag}.log" 2>&1 &
  echo "$!" > "logs/e6_${qtag}.pid"; echo "[$qtag] e6 pid $(cat logs/e6_${qtag}.pid)"
}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2
run_one 8102 e6A vanilla:0 lossgate_vanilla:0 conservative:0 lossgate_conservative:0
run_one 8104 e6B vanilla:1 lossgate_vanilla:1 conservative:1 lossgate_conservative:1
