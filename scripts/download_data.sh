#!/usr/bin/env bash
# 阶段 D 数据下载（在计算节点上跑）。全部来自官方来源；压缩包保留在 data/_archives/。
# 用法：source scripts/env.sh tf5 && bash scripts/download_data.sh <whatsup|coco|depth_meta|gqa_sg|spatialrgpt|all>
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$(python -m src.paths data)"
ARCH="$DATA/_archives"; mkdir -p "$ARCH"
FAIL=0
gd() {  # gd <drive_id> <output>
  if [ -s "$2" ]; then echo "[skip] $2 已存在"; return 0; fi
  gdown --no-cookies "$1" --output "$2" || { echo "[FAIL] gdown $1 -> $2"; rm -f "$2.part"; FAIL=1; return 1; }
  ls -l "$2"
}
dl() {  # dl <url> <output>
  if [ -s "$2" ]; then echo "[skip] $2 已存在"; return 0; fi
  curl -L --fail --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2" || { echo "[FAIL] $1"; FAIL=1; return 1; }
  ls -l "$2"
}
whatsup() {
  # 官方来源：https://github.com/amitakamath/whatsup_vlms （dataset_zoo/aro_datasets.py 里的 Google Drive id）
  local W="$DATA/whatsup"; mkdir -p "$W"
  gd 1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm "$W/controlled_images_dataset.json"
  gd 1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z "$W/controlled_clevr_dataset.json"
  gd 1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq "$W/coco_qa_one_obj.json"
  gd 1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s "$W/coco_qa_two_obj.json"
  gd 1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p "$W/vg_qa_one_obj.json"
  gd 1sjVG5O3QMY8s118k7kQM8zzDZH12i_95 "$W/vg_qa_two_obj.json"
  gd 19KGYVQjrV3syb00GgcavB2nZTW5NXX0H "$ARCH/controlled_images.tar.gz" && [ ! -d "$W/controlled_images" ] && tar -xzf "$ARCH/controlled_images.tar.gz" -C "$W"
  gd 13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG "$ARCH/controlled_clevr.tar.gz"  && [ ! -d "$W/controlled_clevr" ]  && tar -xzf "$ARCH/controlled_clevr.tar.gz" -C "$W"
  # GQA-spatial 用到的图像：What's Up 官方只打包了所需子集
  gd 1idW7Buoz7fQm4-670n-oERw9U-2JLJvE "$ARCH/vg_images.tar.gz" && [ ! -d "$W/vg_images" ] && tar -xzf "$ARCH/vg_images.tar.gz" -C "$W"
}
coco() {
  local C="$DATA/coco"; mkdir -p "$C"
  dl http://images.cocodataset.org/zips/val2017.zip "$ARCH/coco_val2017.zip" && [ ! -d "$C/val2017" ] && unzip -q "$ARCH/coco_val2017.zip" -d "$C"
  dl http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$ARCH/coco_annotations_trainval2017.zip" \
    && [ ! -f "$C/annotations/instances_val2017.json" ] && unzip -q "$ARCH/coco_annotations_trainval2017.zip" 'annotations/instances_val2017.json' -d "$C"
}
depth_meta() {  # 只下说明文件和标注元数据，不下图像
  local D="$DATA/depth_candidates"; local H=https://huggingface.co/datasets
  mkdir -p "$D/spatialrgpt_bench" "$D/3dsrbench" "$D/cv_bench"
  dl $H/a8cheng/SpatialRGPT-Bench/resolve/01b38d2c7aa805223518c061f2a7602ad8f491ef/README.md "$D/spatialrgpt_bench/README.md"
  dl $H/a8cheng/SpatialRGPT-Bench/resolve/01b38d2c7aa805223518c061f2a7602ad8f491ef/SpatialRGPT-Bench_v1.json "$D/spatialrgpt_bench/SpatialRGPT-Bench_v1.json"
  dl $H/ccvl/3DSRBench/resolve/5c90e422b68d0484d50a3ff06b3096ea7d288026/README.md "$D/3dsrbench/README.md"
  dl $H/ccvl/3DSRBench/resolve/5c90e422b68d0484d50a3ff06b3096ea7d288026/3dsrbench_v1.csv "$D/3dsrbench/3dsrbench_v1.csv"
  dl $H/nyu-visionx/CV-Bench/resolve/bc284db50d036958861cb60cdd7b77612052ce0d/README.md "$D/cv_bench/README.md"
  dl $H/nyu-visionx/CV-Bench/resolve/bc284db50d036958861cb60cdd7b77612052ce0d/test_2d.jsonl "$D/cv_bench/test_2d.jsonl"
  dl $H/nyu-visionx/CV-Bench/resolve/bc284db50d036958861cb60cdd7b77612052ce0d/test_3d.jsonl "$D/cv_bench/test_3d.jsonl"
}
gqa_sg() {  # GQA scene graphs（官方下载页 https://cs.stanford.edu/people/dorarad/gqa/download.html，ver 1.1，约 45 MB）
  local G="$DATA/gqa"; mkdir -p "$G"
  dl https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip "$ARCH/gqa_sceneGraphs.zip" || return 1
  # 压缩包视为不可信：只解到它自己的新目录里
  if [ ! -d "$G/sceneGraphs" ]; then
    mkdir -p "$G/sceneGraphs.tmp" && unzip -q "$ARCH/gqa_sceneGraphs.zip" -d "$G/sceneGraphs.tmp" && mv "$G/sceneGraphs.tmp" "$G/sceneGraphs" \
      || { echo "[FAIL] unzip gqa_sceneGraphs.zip"; FAIL=1; return 1; }
  fi
  ( cd "$ARCH" && sha256sum gqa_sceneGraphs.zip )
  ( cd "$G" && find sceneGraphs -type f | LC_ALL=C sort | xargs sha256sum > SHA256SUMS.sceneGraphs )
}
spatialrgpt() {  # SpatialRGPT-Bench 完整快照（约 2.9 GB，固定 commit，匿名下载）。仅用于评测，不得再分发；HF 卡片未声明许可。
  local S="$DATA/depth/spatialrgpt_bench"; mkdir -p "$S"
  python - "$S" <<'PY' || { echo "[FAIL] snapshot_download SpatialRGPT-Bench"; FAIL=1; return 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download("a8cheng/SpatialRGPT-Bench", repo_type="dataset",
                  revision="01b38d2c7aa805223518c061f2a7602ad8f491ef", local_dir=sys.argv[1], max_workers=8)
PY
  ( cd "$S" && find . -type f -not -path './.cache/*' -not -name SHA256SUMS | LC_ALL=C sort | sed 's|^\./||' | xargs -d '\n' sha256sum > SHA256SUMS )
  echo "files: $(find "$S" -type f -not -path '*/.cache/*' -not -name SHA256SUMS | wc -l)  bytes: $(find "$S" -type f -not -path '*/.cache/*' -not -name SHA256SUMS -printf '%s\n' | awk '{s+=$1} END {print s}')"
}
vg_meta() {  # Visual Genome 图像元数据（coco_id），1.8 MB
  mkdir -p "$DATA/vg_meta"
  dl https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip "$ARCH/vg_image_data.json.zip" && unzip -q -n "$ARCH/vg_image_data.json.zip" image_data.json -d "$DATA/vg_meta"
}
case "${1:-all}" in
  vg_meta) vg_meta ;;
  whatsup) whatsup ;; coco) coco ;; depth_meta) depth_meta ;;
  gqa_sg) gqa_sg ;; spatialrgpt) spatialrgpt ;;   # 新增目标不并入 all（需单独确认后再下）
  all) whatsup; coco; depth_meta ;;
  *) echo "unknown target"; exit 2 ;;
esac
du -sh "$DATA"/* 2>/dev/null
exit $FAIL
