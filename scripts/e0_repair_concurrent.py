"""修复 Qwen2.5-VL 的 test.jsonl：A、B 两块（438744、438745）在 NFS 上并发追加同一文件，有行交错 / 残缺。不删任何文件。
  python scripts/e0_repair_concurrent.py <模型> split     两块都结束后运行：逐行解析，完整记录按块写进 test_<块>.jsonl；残缺 / 交错的行逐条记下（行号、长度、能认出的字段）；
                                                          原文件改名为 test.jsonl.concurrent_corrupt 保留。之后用同样的块参数重新提交，断点续跑只补缺的记录。
  python scripts/e0_repair_concurrent.py <模型> verify    补跑之后运行：按唯一键 (stage, 数据集, 样本 id, 题型, 框的角色(目标物体), 轮换/顺序, 被问关系) 去重校验，
                                                          要求每个唯一键恰好一条无报错的记录、总条数等于任务清单（e0_run --dry-run --dump-keys）；列出补跑的键，并检查实现、精度、GPU 与原记录一致。
结果写 <outputs>/e0/<模型>/repair_concurrent.json。"""
import json
import re
import subprocess
import sys
from collections import Counter

from src import e0, paths

REQ = ("model", "stage", "dataset", "sample_id", "image_key", "task", "role", "rotation", "asked_relation", "prompt", "error", "seconds")
CHUNKS = {"whatsup_a+whatsup_b+coco_spatial_two+gqa_spatial_two": {"whatsup_a", "whatsup_b", "coco_spatial_two", "gqa_spatial_two"},
          "coco_spatial_one+gqa_spatial_one": {"coco_spatial_one", "gqa_spatial_one"}}
ORIG_JOBS = ("438744", "438745")
FIELD_RE = {k: re.compile(rb'"' + k.encode() + rb'": ("[^"]*"|null|\d+)') for k in ("dataset", "sample_id", "task", "role", "rotation", "asked_relation", "slurm_job")}
name, mode = sys.argv[1], sys.argv[2]
d = paths.get("outputs") / "e0" / name
rep_f = d / "repair_concurrent.json"


def keyl(r):
    return tuple(str(x) if x is not None else None for x in e0.record_key(r))


if mode == "split":
    src = d / "test.jsonl"
    out = {k: [] for k in CHUNKS}
    frags, stat = [], Counter()
    for i, l in enumerate(open(src, "rb")):
        try:
            r = json.loads(l)
            ok = all(k in r for k in REQ) and r["model"] == name and r["stage"] == "test"
        except Exception:
            ok = False
        if not ok:
            frags.append({"line": i, "bytes": len(l), "fields_seen": {k: [m.decode() for m in rx.findall(l)][:4] for k, rx in FIELD_RE.items() if rx.search(l)}})
            stat["corrupt_lines"] += 1
            continue
        out[[k for k, v in CHUNKS.items() if r["dataset"] in v][0]].append(l.rstrip(b"\n") + b"\n")
        stat["kept"] += 1
    for k, lines in out.items():
        f = d / f"test_{k}.jsonl"
        assert not f.exists(), f"{f} 已存在"
        f.write_bytes(b"".join(lines))
        stat[f"kept_{k}"] = len(lines)
    src.rename(d / "test.jsonl.concurrent_corrupt")
    json.dump({"split": {"stat": dict(stat), "corrupt_fragments": frags}}, open(rep_f, "w"), indent=1, ensure_ascii=False)
    print(dict(stat))
    print(json.dumps(frags, ensure_ascii=False)[:3000])
elif mode == "verify":
    rep = json.load(open(rep_f))
    rep["verify"] = {}
    for k in CHUNKS:
        kf = d / f"expected_keys_{k}.json"
        subprocess.run([sys.executable, "-m", "scripts.e0_run", name, "test", f"--datasets={k.replace('+', ',')}", "--dry-run", f"--dump-keys={kf}"], check=True, capture_output=True)
        exp = {tuple(str(v) if v is not None else None for v in x) for x in json.load(open(kf))}
        recs = [json.loads(l) for l in open(d / f"test_{k}.jsonl")]
        good = [r for r in recs if not r.get("error")]
        cnt = Counter(keyl(r) for r in good)
        dup = [list(x) for x, c in cnt.items() if c > 1]
        missing = sorted(exp - set(cnt), key=str)
        extra = sorted(set(cnt) - exp, key=str)
        jobs = Counter(r["slurm_job"] for r in good)
        rerun = [r for r in good if r["slurm_job"] not in ORIG_JOBS]
        ref = next(r for r in good if r["slurm_job"] in ORIG_JOBS)
        same = all((r["impl"], r["dtype"], r["gpu"], r["eager_fp32_patch"]) == (ref["impl"], ref["dtype"], ref["gpu"], ref["eager_fp32_patch"]) for r in rerun)
        rep["verify"][k] = {"expected": len(exp), "unique_keys": len(cnt), "duplicates": dup, "missing": [list(x) for x in missing], "unexpected": [list(x) for x in extra],
                            "errors_in_file": sum(1 for r in recs if r.get("error")), "records_by_job": dict(jobs), "rerun_keys": [list(keyl(r)) for r in rerun],
                            "rerun_same_impl_dtype_gpu": same, "reference": {"impl": ref["impl"], "dtype": ref["dtype"], "gpu": ref["gpu"]},
                            "ok": len(cnt) == len(exp) and not dup and not missing and not extra and same}
        print(k, {kk: (v if not isinstance(v, list) else len(v)) for kk, v in rep["verify"][k].items()})
    json.dump(rep, open(rep_f, "w"), indent=1, ensure_ascii=False)
