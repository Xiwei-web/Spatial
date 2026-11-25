"""按 configs/downloads/models.yaml 下载固定 commit 的模型快照到 new_models/<name>，并生成 SHA256SUMS。

用法：python scripts/download_models.py [name ...]      （不给名字则全部）
HF token 只从环境变量 HF_TOKEN 读取（huggingface_hub 默认行为），本脚本不读写任何 token 文件。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from src import paths


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    spec = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "downloads" / "models.yaml"))["models"]
    want = set(sys.argv[1:])
    api = HfApi()
    rc = 0
    for m in spec:
        if want and m["name"] not in want:
            continue
        dest = paths.get("new_models") / m["name"]
        info = api.model_info(m["hf_repo"], revision=m["revision"], files_metadata=True)
        if info.gated:
            print(f"[STOP] {m['hf_repo']} 是受限仓库（gated={info.gated}），跳过，需用户决定")
            rc = 1
            continue
        total = sum(s.size or 0 for s in info.siblings)
        print(f"[download] {m['hf_repo']}@{m['revision']} -> {dest}  ({total/1e9:.2f} GB, {len(info.siblings)} files)", flush=True)
        if total > 50e9:
            print("[STOP] 单项超过 50 GB，跳过，需用户决定")
            rc = 1
            continue
        snapshot_download(m["hf_repo"], revision=m["revision"], local_dir=dest, max_workers=8)
        # 核对：文件齐全、大小一致；LFS 文件核对 sha256
        lines, ok = [], True
        for s in sorted(info.siblings, key=lambda s: s.rfilename):
            f = dest / s.rfilename
            if not f.is_file() or (s.size is not None and f.stat().st_size != s.size):
                print(f"  [MISMATCH] {s.rfilename}")
                ok = False
                continue
            digest = sha256(f)
            if s.lfs is not None and s.lfs.sha256 != digest:
                print(f"  [SHA MISMATCH] {s.rfilename}")
                ok = False
            lines.append(f"{digest}  {s.rfilename}")
        (dest / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        print(f"[verify] {m['name']} files={len(lines)} bytes={total} ok={ok}", flush=True)
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
