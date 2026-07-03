#!/bin/bash
# Parallel chunked downloader (HTTP Range). Numeric assembly order.
URL="$1"; OUT="$2"; N="${3:-24}"
SIZE=$(curl -sI "$URL" | awk '/[Cc]ontent-[Ll]ength/{print $2}' | tr -d '\r')
echo "size=$SIZE chunks=$N"
CHUNK=$(( (SIZE + N - 1) / N ))
mkdir -p "${OUT}.parts"; pids=()
for i in $(seq 0 $((N-1))); do
  START=$((i*CHUNK)); END=$((START+CHUNK-1)); [ $END -ge $SIZE ] && END=$((SIZE-1))
  part="${OUT}.parts/part_$(printf '%03d' $i)"
  if [ -f "$part" ] && [ "$(stat -c%s "$part")" -eq $((END-START+1)) ]; then continue; fi
  curl -s -C - -r "${START}-${END}" "$URL" -o "$part" & pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
: > "$OUT"
for i in $(seq 0 $((N-1))); do cat "${OUT}.parts/part_$(printf '%03d' $i)" >> "$OUT"; done
echo "assembled $(stat -c%s "$OUT")/$SIZE"
