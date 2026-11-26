#!/usr/bin/env bash
# 合成集推理的一个作业：依次跑若干 “环境:模型[:额外参数]”。用法：synth.sh <single_object|e1> tf5:Qwen2.5-VL-7B-Instruct tf4:InternVL3-8B …（断点续跑；任一模型出错即停）
# 额外参数用 “+” 连接，例如 tf5:GLM-4.6V-Flash:--datasets=whatsup_a,whatsup_b+--impl=sdpa
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
STAGE="$1"; shift
echo "host=$(hostname) visible=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
for em in "$@"; do
  e="${em%%:*}"; rest="${em#*:}"; m="${rest%%:*}"; extra=""
  [ "$rest" != "$m" ] && extra="${rest#*:}"; extra="${extra//+/ }"
  echo "=== $(date '+%T') [$e] $m $STAGE $extra"
  ( source scripts/env.sh "$e"; python -u scripts/synth_run.py "$m" "$STAGE" $extra ) 2>&1 | grep --line-buffered -v "FutureWarning\|warnings.warn\|Loading weights\|Loading checkpoint\|Setting .pad_token_id"
  rc=${PIPESTATUS[0]}; echo "rc=$rc $m"
  [ $rc -ne 0 ] && { echo "stopped"; exit $rc; }
done
echo all_done
