"""修复 Qwen2.5-VL 的 test.jsonl：A、B 两块（438744、438745）在 NFS 上并发追加同一文件，有行交错 / 残缺。
做法（不删任何文件）：逐行解析，只保留能完整解析、字段齐全且 dataset 属于该块的记录，按块写进新命名的分块文件；
原文件改名为 test.jsonl.concurrent_corrupt 保留；丢失的记录之后用同样的块参数重新提交，断点续跑只补缺的。
用法：python scripts/e0_repair_concurrent.py <模型>   （必须在该模型的 A、B 两块都结束之后运行）"""
import json
import sys
from collections import Counter

from src import paths

REQ = ("model", "stage", "dataset", "sample_id", "image_key", "task", "role", "rotation", "asked_relation", "prompt", "error", "seconds")
CHUNKS = {"whatsup_a+whatsup_b+coco_spatial_two+gqa_spatial_two": {"whatsup_a", "whatsup_b", "coco_spatial_two", "gqa_spatial_two"},
          "coco_spatial_one+gqa_spatial_one": {"coco_spatial_one", "gqa_spatial_one"}}
name = sys.argv[1]
d = paths.get("outputs") / "e0" / name
src = d / "test.jsonl"
out = {k: [] for k in CHUNKS}
stat = Counter()
for l in open(src, "rb"):
    try:
        r = json.loads(l)
    except Exception:
        stat["unparsable_line"] += 1
        continue
    if not all(k in r for k in REQ) or r["model"] != name or r["stage"] != "test":
        stat["incomplete_record"] += 1
        continue
    ch = [k for k, v in CHUNKS.items() if r["dataset"] in v]
    out[ch[0]].append(l.rstrip(b"\n") + b"\n")
    stat["kept"] += 1
for k, lines in out.items():
    f = d / f"test_{k}.jsonl"
    assert not f.exists(), f"{f} 已存在"
    f.write_bytes(b"".join(lines))
    stat[f"kept_{k}"] = len(lines)
src.rename(d / "test.jsonl.concurrent_corrupt")
print(dict(stat))
json.dump(dict(stat), open(d / "repair_concurrent.json", "w"), indent=1)
