#!/usr/bin/env bash
# E4 (policy-agnostic: gate around a trained LoRA maintainer) + E5 (tau over-conservatism sweep).
# LoRA adapter 'maintainer-lora' is served on :8104 (the maintain stage's trained LoRA adapter). Base build/judge on base model.
set -u
cd "$(dirname "$0")" || exit 1   # bundled repo: scripts live in this dir directly
mkdir -p logs results
W=6
COMMON="--n_docs0 200 --docs_per_page 8 --batches 12 --batch_fillers 40 --track_incorp 12 --workers $W"

# E4: gate wrapped around the LoRA maintainer (rewrites routed to adapter; build/judge on base)
( out="results/e4_lossgate_lora_seed0.json"
  if [ ! -f "$out" ]; then
    echo "=== E4 lossgate(LoRA) @:8104 $(date) ==="
    for a in 1 2 3; do [ -f "$out" ] && break
      python p5_lossgate.py --arm lossgate_vanilla --maintainer_model maintainer-lora \
        --seed 0 --port 8104 $COMMON --out "$out"; sleep 8; done
  fi
  # ungated LoRA reference (the policy's average behavior, no gate)
  out2="results/e4_vanilla_lora_seed0.json"
  if [ ! -f "$out2" ]; then
    for a in 1 2 3; do [ -f "$out2" ] && break
      python p5_lossgate.py --arm vanilla --maintainer_model maintainer-lora \
        --seed 0 --port 8104 $COMMON --out "$out2"; sleep 8; done
  fi
  echo "=== E4 DONE $(date) ===" ) >> logs/e4.log 2>&1 &
echo "E4 pid $!" > logs/e4.pid; echo "E4 launched -> logs/e4.log"

# E5: tau sweep on lossgate_vanilla (tau=0 already from A1 seed0); add tau=1,2 on :8102
( for tau in 1 2; do
    out="results/e5_lossgate_vanilla_tau${tau}_seed0.json"
    [ -f "$out" ] && continue
    echo "=== E5 tau=$tau @:8102 $(date) ==="
    for a in 1 2 3; do [ -f "$out" ] && break
      python p5_lossgate.py --arm lossgate_vanilla --tau "$tau" --seed 0 --port 8102 \
        $COMMON --out "$out"; sleep 8; done
  done; echo "=== E5 DONE $(date) ===" ) >> logs/e5.log 2>&1 &
echo "E5 pid $!" > logs/e5.pid; echo "E5 launched -> logs/e5.log"
