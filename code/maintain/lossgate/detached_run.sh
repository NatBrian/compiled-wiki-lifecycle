#!/usr/bin/env bash
# Launch a long Paper-5 job FULLY detached (setsid -> new session, reparented to init)
# so a Claude-Code crash can never kill it. Retry loop resumes until <result> exists.
#   ./detached_run.sh <tag> <result_file> -- <command...>
set -u
TAG="$1"; RESULT="$2"; shift 2
[ "$1" = "--" ] && shift
CMD=("$@")
cd "$(dirname "$0")" || exit 1             # bundled repo: scripts live in this dir directly
mkdir -p logs results
if [ -f "$RESULT" ]; then echo "[$TAG] $RESULT exists — nothing to do"; exit 0; fi
export TOKENIZERS_PARALLELISM=false \
       OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       RAYON_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
       NUMBA_THREADING_LAYER=workqueue MALLOC_CONF=background_thread:false
setsid bash -c '
  for attempt in $(seq 1 20); do
    [ -f "'"$RESULT"'" ] && break
    echo "=== attempt $attempt $(date) ==="
    "$@"
    sleep 15
  done
' bash "${CMD[@]}" >> "logs/$TAG.log" 2>&1 < /dev/null &
echo "[$TAG] launched pid $! -> logs/$TAG.log (result: $RESULT)"
echo $! > "logs/$TAG.pid"
