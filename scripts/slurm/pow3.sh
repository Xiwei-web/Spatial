#!/usr/bin/env bash
# 功效模拟第三版的一个 CPU 作业：先重算试跑讨厌参数（只读已跑完的试跑），再对给定模型依次跑所有用试跑参数的情景。用法：pow3.sh 模型1,模型2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh tools
set -e
python -u scripts/e1_pilot_nuisance.py
for sc in synth_main_pilot synth_main_pilot_wins synth_overlap_pilot real_whatsup_a real_whatsup_b real_coco_spatial_two real_gqa_spatial_two \
          real_whatsup_a_pair real_whatsup_b_pair real_coco_spatial_two_pair real_gqa_spatial_two_pair; do
  python -u -m scripts.e1_power_sim_v3 --scenario $sc --models "$1"
done
echo all_done
