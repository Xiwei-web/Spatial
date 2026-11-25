"""集中读取 configs/paths.yaml。代码中任何外部路径都必须经由这里。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def load_paths() -> dict[str, Path]:
    f = Path(os.environ.get("INTERP_PATHS_FILE", REPO_ROOT / "configs" / "paths.yaml"))
    with open(f) as fh:
        raw = yaml.safe_load(fh)
    return {k: Path(v) for k, v in raw.items()}


def get(key: str) -> Path:
    paths = load_paths()
    if key not in paths:
        raise KeyError(f"configs/paths.yaml 中没有键 {key!r}；已有：{sorted(paths)}")
    return paths[key]


if __name__ == "__main__":  # python -m src.paths <key>  供 shell 脚本取路径
    import sys

    print(get(sys.argv[1]))
