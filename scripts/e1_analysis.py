"""E1 分析入口：读 E0 / E1 的 JSONL（一个或多个文件，同一模型），先过 test 安全阀，再整理条目、算统计量和 bootstrap CI。
用法：python scripts/e1_analysis.py <jsonl> [<jsonl> …] [--out out.json] [--n-boot N]
阈值与参数从 configs/e1.yaml 读取（当前为占位值）。"""
import argparse
import json

import yaml

from src import e0, paths
from src.e1 import require_prereg_if_test
from src.e1_analysis import analyze, build_items

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out"); ap.add_argument("--n-boot", type=int)
A = ap.parse_args()
cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "e1.yaml"))
records = [r for f in A.files for r in e0.load_records(f).values()]
require_prereg_if_test(records)
models = {r["model"] for r in records}
assert len(models) == 1, f"一次只分析一个模型：{models}"
items, skipped = build_items(records, cfg)
res = {"model": models.pop(), "files": A.files, "config_status": cfg["status"], "skipped": skipped, **analyze(items, cfg, n_boot=A.n_boot)}
print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])
if A.out:
    json.dump(res, open(A.out, "w"), indent=1, ensure_ascii=False)
