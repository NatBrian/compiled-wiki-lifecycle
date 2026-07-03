#!/usr/bin/env bash
# A1 make-or-break: 4 arms x 3 seeds = 12 runs, two sequential queues (one per server),
# both detached so they survive Claude crashes. Polite workers to share the servers.
#   ./run_a1.sh
set -u
cd "$(dirname "$0")" || exit 1   # bundled repo: scripts live in this dir directly
mkdir -p logs results
W=6
COMMON="--n_docs0 200 --docs_per_page 8 --batches 12 --batch_fillers 40 --track_incorp 12 --workers $W"

run_queue () {  # $1=port  $2=qtag  then arm:seed pairs
  local port="$1" qtag="$2"; shift 2
  ( for pair in "$@"; do
      arm="${pair%%:*}"; seed="${pair##*:}"
      out="results/a1_${arm}_seed${seed}.json"
      if [ -f "$out" ]; then echo "[$qtag] $out exists, skip"; continue; fi
      echo "=== [$qtag] $arm seed $seed @:$port $(date) ==="
      for attempt in 1 2 3; do
        [ -f "$out" ] && break
        python p5_lossgate.py --arm "$arm" --seed "$seed" --port "$port" \
          $COMMON --out "$out"
        sleep 10
      done
    done
    echo "=== [$qtag] QUEUE DONE $(date) ===" ) >> "logs/a1_${qtag}.log" 2>&1 &
  echo "$!" > "logs/a1_${qtag}.pid"
  echo "[$qtag] queue launched pid $(cat logs/a1_${qtag}.pid) -> logs/a1_${qtag}.log"
}

export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2

setsid bash -c '
'"$(declare -f run_queue)"'
W='"$W"'
COMMON="'"$COMMON"'"
run_queue 8102 qA  vanilla:0 vanilla:1 vanilla:2 lossgate_vanilla:0 lossgate_vanilla:1 lossgate_vanilla:2
run_queue 8104 qB  conservative:0 conservative:1 conservative:2 lossgate_conservative:0 lossgate_conservative:1 lossgate_conservative:2
wait
' >> logs/a1_master.log 2>&1 < /dev/null &
echo "A1 master pid $! -> logs/a1_master.log"
