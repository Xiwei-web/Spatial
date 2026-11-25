#!/usr/bin/env bash
# 在一个已分配 GPU 的作业里，对若干模型依次执行某个“按模型运行”的脚本，自动按 configs/adapters.yaml 切换环境。
# 用法：bash scripts/run_group.sh <python 脚本路径> <日志子目录> <模型名>...
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SCRIPT="$1"; SUB="$2"; shift 2
LOGS="$(sed -n 's/^logs:[[:space:]]*\([^[:space:]#]*\).*/\1/p' configs/paths.yaml)/$SUB"; mkdir -p "$LOGS"
for M in "$@"; do
  ENVN=$(awk -v m="$M" '$1==m":" { if (match($0, /env: *[a-z0-9]+/)) { s=substr($0, RSTART, RLENGTH); sub(/env: */, "", s); print s } }' configs/adapters.yaml)
  ENVN=${ENVN:-tf5}
  echo "=== $(date '+%F %T') $SCRIPT $M (env $ENVN) ==="
  ( source scripts/env.sh "$ENVN" && python -u "$SCRIPT" "$M" ) > "$LOGS/$M.log" 2>&1
  echo "rc=$? ; $(grep -E '^SUMMARY' "$LOGS/$M.log" | tail -1)"
done
