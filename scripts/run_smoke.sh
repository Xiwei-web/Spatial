#!/usr/bin/env bash
# 在已分配到 GPU 的作业里依次对若干模型跑冒烟测试；每个模型用 configs/adapters.yaml 指定的环境。
# 用法（通过调度系统，经 scripts/submit.sh 提交）：scripts/submit.sh gpuN ... bash scripts/run_smoke.sh <模型名> [模型名...]
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOGS="$(sed -n 's/^logs:[[:space:]]*\([^[:space:]#]*\).*/\1/p' configs/paths.yaml)/smoke"; mkdir -p "$LOGS"
for M in "$@"; do
  ENVN=$(awk -v m="$M" '$1==m":" { if (match($0, /env: *[a-z0-9]+/)) { s=substr($0, RSTART, RLENGTH); sub(/env: */, "", s); print s } }' configs/adapters.yaml)
  ENVN=${ENVN:-tf5}
  echo "=== $(date '+%F %T') $M (env $ENVN) on $(hostname) ==="
  ( source scripts/env.sh "$ENVN" && python scripts/smoke_test.py "$M" ) > "$LOGS/$M.log" 2>&1
  echo "rc=$? ; $(grep '^SUMMARY' "$LOGS/$M.log" | tail -1)"
done
