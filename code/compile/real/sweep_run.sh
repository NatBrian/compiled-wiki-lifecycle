#!/usr/bin/env bash
# Reader-size sweep on pooled-HoH for ONE small reader (parallel-friendly: one
# process per size). full_dump uses the SAME small reader (32k ctx holds the
# ~17k-token pool). 14B baseline = the already-finished main pool_* results.
# Usage: sweep_run.sh <sfx> <model> <url> [N]
set -u
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
sfx="$1" model="$2" url="$3" N="${4:-300}"
export TAG_SUFFIX="$sfx" AGENT_MODEL="$model" AGENT_URL="$url" DUMP_MODEL="$model" DUMP_URL="$url"
PY=real/venvs/lightrag/bin/python
say(){ echo "$(date '+%m-%d %H:%M:%S') [sweep$sfx] $*"; }
for arm in full_dump_cic vector_rag resolve_free wiki_karpathy; do
  out="results/results_pool_${arm}${sfx}.json"
  for attempt in $(seq 1 20); do
    [ -f "$out" ] && { say "$arm DONE"; break; }
    say "$arm attempt $attempt"
    $PY real/pooled_hoh.py "$arm" "$N" >> "logs/sweep${sfx}.log" 2>&1
    [ -f "$out" ] && { say "$arm DONE"; break; }
    sleep 5
  done
done
say "### sweep$sfx complete ###"
