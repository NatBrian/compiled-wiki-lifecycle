#!/bin/bash
# Passive watcher: exits when a GPU has NO process owned by another user.
# Never launches anything. ME=$(whoami). Min free mem 40000 MiB (for 14B).
ME=$(whoami); MINFREE=40000; MAXITER=144; SLEEP=300
for it in $(seq 1 $MAXITER); do
  # uuid -> index
  declare -A IDX
  while IFS=, read -r idx uuid; do IDX["$(echo $uuid|tr -d ' ')"]=$(echo $idx|tr -d ' '); done \
    < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  # mark indices that have a non-ME process
  declare -A BUSY
  while IFS=, read -r uuid pid; do
    uuid=$(echo $uuid|tr -d ' '); pid=$(echo $pid|tr -d ' ')
    u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && BUSY[${IDX[$uuid]}]=1
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  # check each gpu for free+idle
  while IFS=, read -r idx free; do
    idx=$(echo $idx|tr -d ' '); free=$(echo $free|tr -d ' ')
    if [ -z "${BUSY[$idx]}" ] && [ "$free" -ge "$MINFREE" ]; then
      echo "GPU $idx FREE (no other-user proc, ${free}MiB free) at iter $it"; exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  unset BUSY IDX
  sleep $SLEEP
done
echo "no GPU freed after $((MAXITER*SLEEP/3600))h"; exit 1
