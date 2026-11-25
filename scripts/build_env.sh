#!/usr/bin/env bash
# 用法：bash scripts/build_env.sh <tf5|tf4|tools>     （应在计算节点上跑：scripts/submit.sh cpu scripts/build_env.sh tf5）
# 1) conda 建 python 3.11 前缀环境  2) uv 由 requirements/<env>.in 解析出带哈希的锁文件
# 3) uv pip sync 严格按锁文件安装。锁文件 requirements/<env>.lock 提交进 Git。
set -euo pipefail
NAME="${1:?env name}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh" "$NAME" 2>/dev/null || true
PATHS="${INTERP_PATHS_FILE:-$REPO/configs/paths.yaml}"
_p() { sed -n "s/^$1:[[:space:]]*\([^[:space:]#]*\).*/\1/p" "$PATHS"; }
ENV_DIR="$(_p envs)/$NAME"
CONDA="$(_p conda_exe)"
export CONDA_PKGS_DIRS="$(_p cache)/conda_pkgs"
mkdir -p "$(_p envs)"
exec 9>"$(_p envs)/.$NAME.lock"; flock 9      # 同一环境不并发构建
if [ -f "$ENV_DIR/.build_ok" ] && [ "${REBUILD:-0}" != 1 ]; then echo "[build_env] $NAME 已构建，跳过"; exit 0; fi
if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA" create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.11 pip
fi
"$ENV_DIR/bin/python" -m pip install --quiet "uv==0.13.10"
UV="$ENV_DIR/bin/uv"
LOCK="$REPO/requirements/$NAME.lock"
if [ ! -s "$LOCK" ] || [ "${RELOCK:-0}" = 1 ]; then
  "$UV" pip compile "$REPO/requirements/$NAME.in" -o "$LOCK" --generate-hashes \
      --python "$ENV_DIR/bin/python" --index-strategy unsafe-best-match --no-header
fi
"$UV" pip sync "$LOCK" --python "$ENV_DIR/bin/python" --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu128 --link-mode copy
"$ENV_DIR/bin/python" - <<'PY'
import torch, transformers, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda,
      "transformers", transformers.__version__, "cuda available", torch.cuda.is_available())
PY
touch "$ENV_DIR/.build_ok"
echo "[build_env] $NAME OK -> $ENV_DIR"
