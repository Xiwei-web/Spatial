#!/usr/bin/env bash
# Qwen2.5-VL-7B 的 teacher-forced 与 fp32 生成（1 卡）。fp32 先跑（含门槛检查）；OOM 时分支块降到 5 再试一次，还 OOM 就停；门槛不过就停。bf16 用同样的块大小。
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh tf5
M=Qwen2.5-VL-7B-Instruct
echo "host=$(hostname) visible=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse --short HEAD)"
run() { echo "=== $(date '+%T') chunk=$2 $1 ${3:-}"; python -u scripts/teacher_forced_qwen25vl.py $M "$1" --chunk "$2" ${3:-} 2>&1; }
C=10
run fp32 $C --generate; rc=$?; echo "rc=$rc"
if [ $rc -eq 42 ]; then C=5; run fp32 $C --generate; rc=$?; echo "rc=$rc"; fi
[ $rc -ne 0 ] && { echo "stopped rc=$rc"; exit $rc; }
run bf16 $C; rc=$?; echo "rc=$rc"
[ $rc -ne 0 ] && { echo "stopped rc=$rc"; exit $rc; }
echo all_done
