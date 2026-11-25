#!/usr/bin/env bash
# 准备阶段的一次性链条：建两个环境 → 下载 What's Up / 深度候选集元数据 / COCO → 下载补充模型。可重复执行（各步自带跳过）。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/build_env.sh tf5 || exit 1
bash scripts/build_env.sh tf4 || echo "[prep_chain] tf4 构建失败"
source scripts/env.sh tf5
exec 8>"$(python -m src.paths data)/.download.lock"; flock 8
bash scripts/download_data.sh whatsup;  echo "[prep_chain] whatsup rc=$?"
bash scripts/download_data.sh depth_meta; echo "[prep_chain] depth_meta rc=$?"
bash scripts/download_data.sh coco; echo "[prep_chain] coco rc=$?"
python scripts/download_models.py; echo "[prep_chain] models rc=$?"
