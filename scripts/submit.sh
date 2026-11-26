#!/usr/bin/env bash
# 统一的 Slurm 提交入口（所有 sbatch / srun 都经这里，不得绕过）。日志目录从 configs/paths.yaml 读，排除节点从 configs/slurm.yaml 读。
# 用法：scripts/submit.sh [--srun] <cpu|gpu1|gpu2|gpu3|gpu4> [--time=HH:MM:SS] [--name=xxx] <command...>
#   默认 sbatch：cpu -> cscc-cpu-p 8 核 32G；gpuN -> cscc-gpu-p 批处理（QoS cscc-gpu-qos，48h），N 张 A100-40GB；日志 <logs>/<name>_<jobid>.out
#   --srun：前台交互 / 调试（cscc-gpu-p 上自动进 gpu-debug-qos，3h、每人同时 2 个），输出到标准输出（调用方自己重定向）
#   环境变量 SUBMIT_DRYRUN=1：只打印将要执行的命令，不提交（测试用）
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS="${INTERP_PATHS_FILE:-$REPO/configs/paths.yaml}"
SLURM_CFG="${INTERP_SLURM_FILE:-$REPO/configs/slurm.yaml}"
LOGS="$(sed -n 's/^logs:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$PATHS")"
[ -f "$SLURM_CFG" ] || { echo "submit.sh: 找不到 $SLURM_CFG" >&2; exit 3; }
EXCLUDE="$(sed -n 's/^exclude_nodes:[[:space:]]*\[\(.*\)\].*/\1/p' "$SLURM_CFG" | tr -d ' ')"
[ -n "$EXCLUDE" ] || { echo "submit.sh: $SLURM_CFG 里读不到 exclude_nodes" >&2; exit 3; }
MODE=sbatch
[ "${1:-}" = "--srun" ] && { MODE=srun; shift; }
KIND="$1"; shift
TIME="04:00:00"; NAME="interp"
while [[ "${1:-}" == --* ]]; do
  case "$1" in --time=*) TIME="${1#*=}";; --name=*) NAME="${1#*=}";; *) echo "unknown $1"; exit 2;; esac; shift
done
case "$KIND" in
  cpu)  RES=(-p cscc-cpu-p -c 8 --mem=32G); QOS=() ;;
  gpu1) RES=(-p cscc-gpu-p --gres=gpu:1 -c 8 --mem=64G); QOS=(-q cscc-gpu-qos) ;;
  gpu2) RES=(-p cscc-gpu-p --gres=gpu:2 -c 12 --mem=120G); QOS=(-q cscc-gpu-qos) ;;
  gpu3) RES=(-p cscc-gpu-p --gres=gpu:3 -c 12 --mem=160G); QOS=(-q cscc-gpu-qos) ;;
  gpu4) RES=(-p cscc-gpu-p --gres=gpu:4 -c 16 --mem=200G); QOS=(-q cscc-gpu-qos) ;;
  *) echo "kind must be cpu|gpu1|gpu2|gpu3|gpu4"; exit 2 ;;
esac
# --export=ALL：HF_TOKEN 等只经由环境变量传入作业，不落盘
if [ "$MODE" = sbatch ]; then
  mkdir -p "$LOGS"
  CMD=(sbatch --parsable "${RES[@]}" ${QOS[@]+"${QOS[@]}"} --exclude="$EXCLUDE" -t "$TIME" -J "$NAME" -D "$REPO" --export=ALL -o "$LOGS/%x_%j.out" --wrap "$(printf '%q ' "$@")")
else
  CMD=(srun "${RES[@]}" --exclude="$EXCLUDE" -t "$TIME" -J "$NAME" -D "$REPO" --export=ALL "$@")
fi
if [ "${SUBMIT_DRYRUN:-0}" = 1 ]; then printf '%s ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
