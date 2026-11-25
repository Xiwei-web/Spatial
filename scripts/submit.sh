#!/usr/bin/env bash
# 统一的 Slurm 提交入口，日志目录从 configs/paths.yaml 读取（脚本里不写绝对路径）。
# 用法：scripts/submit.sh <cpu|gpu1|gpu2|gpu4> [--time=HH:MM:SS] [--name=xxx] <command...>
#   cpu  -> cscc-cpu-p, 8 核 32G
#   gpuN -> cscc-gpu-p 批处理（QoS cscc-gpu-qos，48h；long 分区本账号不可用），N 张 A100-40GB
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS="${INTERP_PATHS_FILE:-$REPO/configs/paths.yaml}"
LOGS="$(sed -n 's/^logs:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$PATHS")"
mkdir -p "$LOGS"
KIND="$1"; shift
TIME="04:00:00"; NAME="interp"
while [[ "${1:-}" == --* ]]; do
  case "$1" in --time=*) TIME="${1#*=}";; --name=*) NAME="${1#*=}";; *) echo "unknown $1"; exit 2;; esac; shift
done
case "$KIND" in
  cpu)  RES=(-p cscc-cpu-p -c 8 --mem=32G) ;;
  gpu1) RES=(-p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:1 -c 8 --mem=64G) ;;
  gpu2) RES=(-p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:2 -c 12 --mem=120G) ;;
  gpu3) RES=(-p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:3 -c 12 --mem=160G) ;;
  gpu4) RES=(-p cscc-gpu-p -q cscc-gpu-qos --gres=gpu:4 -c 16 --mem=200G) ;;
  *) echo "kind must be cpu|gpu1|gpu2|gpu3|gpu4"; exit 2 ;;
esac
# --export=ALL：HF_TOKEN 等只经由环境变量传入作业，不落盘
sbatch --parsable "${RES[@]}" -t "$TIME" -J "$NAME" -D "$REPO" --export=ALL \
  -o "$LOGS/%x_%j.out" --wrap "$(printf '%q ' "$@")"
