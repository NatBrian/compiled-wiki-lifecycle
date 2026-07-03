#!/usr/bin/env bash
# One sequential queue of (arm:seed) jobs on a given port. Detached caller handles setsid.
#   run_queue.sh <port> <qtag> <arm:seed> [arm:seed ...]
set -u
cd "$(dirname "$0")" || exit 1   # bundled repo: scripts live in this dir directly
mkdir -p logs results
port="$1"; qtag="$2"; shift 2
W=6
COMMON="--n_docs0 200 --docs_per_page 8 --batches 12 --batch_fillers 40 --track_incorp 12 --workers $W"
for pair in "$@"; do
  arm="${pair%%:*}"; seed="${pair##*:}"
  out="results/a1_${arm}_seed${seed}.json"
  if [ -f "$out" ]; then echo "[$qtag] $out exists, skip"; continue; fi
  echo "=== [$qtag] $arm seed $seed @:$port $(date) ==="
  for attempt in 1 2 3; do
    [ -f "$out" ] && break
    python p5_lossgate.py --arm "$arm" --seed "$seed" --port "$port" $COMMON --out "$out"
    sleep 10
  done
done
echo "=== [$qtag] QUEUE DONE $(date) ==="
