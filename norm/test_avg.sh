#!/usr/bin/env bash
set -euo pipefail

count=${1:-2}

[[ $count =~ ^[1-9][0-9]*$ ]] && (( count >= 2 )) || {
  echo "COUNT must be at least 2" >&2
  exit 2
}

timings=()

for ((run = 1; run <= count; run++)); do
  elapsed=$(
    TIMEFORMAT='%3R'
    { time nargo compile --silence-warnings --force > /dev/null 2>&1; } 2>&1
  )

  if (( run > 1 )); then
    timings+=("$elapsed")
  fi
done

printf '%s\n' "${timings[@]}" |
  awk '{ sum += $1 } END { printf "Average: %.3f s\n", sum / NR }'