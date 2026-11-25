#!/usr/bin/env bash
# 框实验精度决策用的三组 GPU 测试（2 张卡的一个作业内依次跑）：显存/耗时（1 卡、2 卡）与 teacher-forced 读出（bf16、fp32）。
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ONE="${CUDA_VISIBLE_DEVICES%%,*}"
echo "visible=$CUDA_VISIBLE_DEVICES one=$ONE host=$(hostname)"
r() { local env="$1" cvd="$2"; shift 2; echo "=== $(date '+%T') [$env gpu=$cvd] $*"; ( source scripts/env.sh "$env"; CUDA_VISIBLE_DEVICES="$cvd" python -u "$@" ) 2>&1; echo "rc=$?"; }
r tf4 "$ONE" scripts/teacher_forced_readout.py InternVL3-8B bf16
r tf4 "$CUDA_VISIBLE_DEVICES" scripts/teacher_forced_readout.py InternVL3-8B fp32
r tf5 "$ONE" scripts/bench_attention_memory.py Qwen2.5-VL-7B-Instruct 1gpu
r tf5 "$CUDA_VISIBLE_DEVICES" scripts/bench_attention_memory.py Qwen2.5-VL-7B-Instruct 2gpu
r tf4 "$ONE" scripts/bench_attention_memory.py InternVL3_5-8B 1gpu
r tf4 "$CUDA_VISIBLE_DEVICES" scripts/bench_attention_memory.py InternVL3_5-8B 2gpu
echo all_done
