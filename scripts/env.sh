#!/usr/bin/env bash
# 用法：source scripts/env.sh <env名>   （env名：tf5 | tf4）
# 从 configs/paths.yaml 读路径，激活环境，并把所有缓存指向 nfs-stor。
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_PATHS="${INTERP_PATHS_FILE:-$_REPO/configs/paths.yaml}"
_p() { sed -n "s/^$1:[[:space:]]*\([^[:space:]#]*\).*/\1/p" "$_PATHS"; }
export INTERP_REPO="$_REPO"
export INTERP_ENV_NAME="${1:?usage: source scripts/env.sh <tf5|tf4|tools>}"
export HF_HOME="$(_p hf_home)"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$(_p cache)/torch"
export PIP_CACHE_DIR="$(_p cache)/pip"
export UV_CACHE_DIR="$(_p cache)/uv"
export MPLCONFIGDIR="$(_p cache)/matplotlib"
export XDG_CACHE_HOME="$(_p cache)/xdg"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1   # 防止 trust_remote_code 在只读权重目录旁写 __pycache__
export PYTHONPATH="$_REPO${PYTHONPATH:+:$PYTHONPATH}"
_ENV_DIR="$(_p envs)/$INTERP_ENV_NAME"
if [ -x "$_ENV_DIR/bin/python" ]; then
  export PATH="$_ENV_DIR/bin:$PATH"
  export CONDA_PREFIX="$_ENV_DIR"
else
  echo "[env.sh] 环境 $_ENV_DIR 还不存在（先跑 scripts/build_env.sh）" >&2
fi
# HF_TOKEN 只从调用者的环境变量继承，这里不设置也不打印。
