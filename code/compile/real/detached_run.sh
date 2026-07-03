#!/usr/bin/env bash
# Launch a long experiment FULLY detached so a Claude-Code crash (core dump,
# SIGHUP, tool-process death) can never kill it. Uses setsid → new session,
# no controlling terminal, reparented to init. Survives parent death.
#
# Usage:
#   real/detached_run.sh <tag> <result_file> -- <command...>
# Example:
#   real/detached_run.sh pool_raptor results/results_pool_raptor.json -- \
#     real/venvs/raptor/bin/python real/pooled_raptor.py 300
#
# Re-running with the same <result_file> is a no-op if it already exists.
# Retry loop: re-invokes the command (checkpoint-resume) up to 30 times until
# <result_file> appears. Log → logs/<tag>.log. PID → logs/<tag>.pid.
set -u
TAG="$1"; RESULT="$2"; shift 2
[ "$1" = "--" ] && shift
CMD=("$@")
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs results

if [ -f "$RESULT" ]; then echo "[$TAG] $RESULT already exists — nothing to do"; exit 0; fi

# Thread caps: this box runs at load ~24 with 16K threads; uncapped BLAS/numba/
# rayon/tokenizers spawn 64 threads each and hit EAGAIN. Keep every layer small.
export TOKENIZERS_PARALLELISM=false \
       OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
       NUMBA_THREADING_LAYER=workqueue MALLOC_CONF=background_thread:false \
       AGENT_MODEL="${AGENT_MODEL:-qwen2.5-14b}" \
       AGENT_URL="${AGENT_URL:-http://127.0.0.1:8101/v1}"

setsid bash -c '
  for attempt in $(seq 1 30); do
    [ -f "'"$RESULT"'" ] && break
    "$@"
    sleep 20
  done
' bash "${CMD[@]}" >> "logs/$TAG.log" 2>&1 < /dev/null &

echo $! > "logs/$TAG.pid"
echo "[$TAG] detached pid $(cat logs/$TAG.pid) → logs/$TAG.log (result: $RESULT)"
