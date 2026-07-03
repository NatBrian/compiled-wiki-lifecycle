#!/bin/bash
# Run the remaining feasible experiments AFTER the n=250 scale finishes.
# Sequential (shared vLLM/mem0; concurrent drivers exhaust the thread table).
#   1. stats_cluster  : re-run GEE/TOST on the n=250 spine        (no GPU)
#   2. sanity         : Block 0 gate-predicate gold-set check     (light)
#   3. cost           : per-hook weight-free tax (Table 8)        (light)
#   4. recon          : tombstone non-invertibility (Table 5)     (CPU embed)
#   5. mia            : Min-K%/Min-K%++ MIA-AUC (Table 5)          (vLLM logprobs)
#   6. k_sweep (n=60) : RSR/round curves (Fig 2), K_MAX=24         (vLLM+mem0, chunked)
set -u
cd "$(dirname "$0")"
P="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1

echo "############ REMAINING START $(date) ############"
echo "=== 1. stats_cluster (n=250) ==="; "$P" stats_cluster.py > logs/r_stats.log 2>&1; echo "exit=$?"
echo "=== 2. sanity ===";               "$P" sanity.py        > logs/r_sanity.log 2>&1; echo "exit=$?"
echo "=== 3. cost ===";                 "$P" cost.py          > logs/r_cost.log 2>&1; echo "exit=$?"
echo "=== 4. recon ===";   P7_N_RECON=250 "$P" recon.py      > logs/r_recon.log 2>&1; echo "exit=$?"
echo "=== 5. mia ===";     P7_N_MIA=250   "$P" mia.py        > logs/r_mia.log 2>&1; echo "exit=$?"

echo "=== 6. k_sweep n=40 K_MAX=12 (chunked) ==="
KN=${KN:-40}; KCHUNK=${KCHUNK:-10}; KMAX=${KMAX:-12}
rm -f results/k_sweep_part_*.json; rm -rf /tmp/p7_k_*
for ((off=0; off<KN; off+=KCHUNK)); do
  echo "--- ksweep chunk offset=$off $(date) ---"
  P7_N_K=$KCHUNK P7_OFFSET=$off P7_WORKERS_K=2 P7_K_MAX=$KMAX P7_PART=1 \
    P7_CHROMA_K=/tmp/p7_k_$off "$P" k_sweep.py >> logs/r_ksweep.log 2>&1
  rm -rf /tmp/p7_k_$off
done
"$P" merge_ksweep.py > logs/r_ksweep_merge.log 2>&1; echo "ksweep merge exit=$?"
echo "############ REMAINING DONE $(date) ############"