#!/usr/bin/env bash
# 框坐标约定标定的一个作业：依次跑若干 “环境:模型”。用法：bbox_calib.sh tf5:Qwen2.5-VL-7B-Instruct tf4:InternVL3-8B …（断点续跑：已完成条目自动跳过）
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "host=$(hostname) visible=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse --short HEAD)"
for em in "$@"; do
  e="${em%%:*}"; m="${em#*:}"
  echo "=== $(date '+%T') [$e] $m"
  ( source scripts/env.sh "$e"; python -u scripts/bbox_calib_run.py "$m" ) 2>&1 | grep -v "FutureWarning\|warnings.warn\|Loading weights\|Loading checkpoint"
  echo "rc=${PIPESTATUS[0]} $m"
done
echo all_done
