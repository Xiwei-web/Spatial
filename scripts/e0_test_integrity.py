"""E0 第二段输出的完整性检查（只看键和报错，不算任何指标）：每个模型 × 每块（A、B、默认实现 C），
用 e0_run --dry-run --dump-keys 得到任务清单的唯一键，与输出文件比对：每个唯一键恰好一条无报错的记录、没有多余的键、每行都能解析。
输出 manifests/e0_test_integrity.json。用法：python -m scripts.e0_test_integrity [--models A,B]"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter

from src import e0, paths

CHUNKS = {"A": ("whatsup_a,whatsup_b,coco_spatial_two,gqa_spatial_two", "eager"), "B": ("coco_spatial_one,gqa_spatial_one", "eager"), "C": ("whatsup_a,whatsup_b", "sdpa")}
ap = argparse.ArgumentParser(); ap.add_argument("--models", default=",".join(e0.FAMILY)); a = ap.parse_args()
O = paths.get("outputs") / "e0"
res = {}
for m in a.models.split(","):
    d = O / m
    res[m] = {}
    for c, (ds, impl) in CHUNKS.items():
        f = d / f"test_{ds.replace(',', '+')}{'' if impl == 'eager' else '_sdpa'}.jsonl"
        if not f.is_file():
            res[m][c] = {"file": f.name, "missing_file": True, "ok": False}
            continue
        with tempfile.NamedTemporaryFile(suffix=".json") as kf:
            subprocess.run([sys.executable, "-m", "scripts.e0_run", m, "test", f"--datasets={ds}", f"--impl={impl}", "--dry-run", f"--dump-keys={kf.name}"], check=True, capture_output=True)
            exp = {tuple(str(v) if v is not None else None for v in x) for x in json.load(open(kf.name))}
        bad, recs = 0, []
        for l in open(f, "rb"):
            try:
                recs.append(json.loads(l))
            except Exception:
                bad += 1
        good = [r for r in recs if not r.get("error")]
        cnt = Counter(tuple(str(x) if x is not None else None for x in e0.record_key(r)) for r in good)
        dup = sum(1 for v in cnt.values() if v > 1)
        miss, extra = len(exp - set(cnt)), len(set(cnt) - exp)
        impls = sorted({r["impl"] for r in good})
        res[m][c] = {"file": f.name, "expected": len(exp), "unique_keys": len(cnt), "duplicates": dup, "missing": miss, "unexpected": extra, "unparsable_lines": bad,
                     "error_records": len(recs) - len(good), "impl": impls, "gpus": sorted({r["gpu"] for r in good}), "jobs": sorted({str(r["slurm_job"]) for r in good}),
                     "ok": len(cnt) == len(exp) and not dup and not miss and not extra and not bad and impls == [impl]}
    print(m, {c: (v.get("ok"), v.get("unique_keys"), v.get("expected"), v.get("duplicates"), v.get("missing")) for c, v in res[m].items()}, flush=True)
json.dump(res, open(paths.REPO_ROOT / "manifests" / "e0_test_integrity.json", "w"), indent=1)
