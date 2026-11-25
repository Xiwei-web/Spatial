"""框坐标约定标定集 v1（只用 dev；固定清单，种子 20261125）。用法：python scripts/make_bbox_calib.py → splits/bbox_calib_v1.json
What's Up dev 200 张：A、B 各 100，每个关系 25（按关系分层），只取 auto_v3 里可用的样本；target、reference 各问一次，真值 box_gt_xyxy。
COCO-spatial two-object dev 100 张：先按图像抽，每张图再抽一个样本，两个物体的框都唯一匹配（loader 的 bbox 不为 None，即 mask 外接框）才可用；两个物体各问一次。"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src import paths
from src.data import load_split
from src.data.whatsup_annotations import load_annotations, truth_box

SEED = 20261125
ann = {(d["dataset"], d["sample_id"]): d for d in load_annotations("auto_v3")}
items, counts = [], {}
for ds in ("whatsup_a", "whatsup_b"):
    by_rel = defaultdict(list)
    for s in load_split(ds, "dev"):
        if (ds, s.sample_id) in ann:
            by_rel[s.relation].append(s)
    for rel in sorted(by_rel):
        pool = sorted(by_rel[rel], key=lambda s: s.sample_id)
        for s in random.Random(f"{SEED}:bbox_calib:{ds}:{rel}").sample(pool, 25):
            a = ann[(ds, s.sample_id)]
            for role in ("target", "reference"):
                o = a["objects"][role]
                assert o["name"] == getattr(s, role)
                items.append({"item_id": f"{ds}/{s.sample_id}/{role}", "subset": "whatsup", "dataset": ds, "sample_id": s.sample_id, "relation": rel, "role": role,
                              "object": o["name"], "image_key": f"{ds}:{s.sample_id}", "orig_wh": [a["width"], a["height"]], "truth_xyxy": [float(v) for v in truth_box(o)]})
    counts[ds] = {r: 25 for r in sorted(by_rel)}
pool = [s for s in load_split("coco_spatial_two", "dev") if s.bbox_target is not None and s.bbox_reference is not None]
by_img = defaultdict(list)
for s in pool:
    by_img[int(Path(s.image_path).stem)].append(s)
chosen = sorted(random.Random(f"{SEED}:bbox_calib:coco:images").sample(sorted(by_img), 100))
from PIL import Image                                                      # noqa: E402
for im in chosen:
    s = random.Random(f"{SEED}:bbox_calib:coco:{im}").choice(sorted(by_img[im], key=lambda s: s.sample_id))
    wh = Image.open(s.image_path).size
    for role in ("target", "reference"):
        items.append({"item_id": f"coco_spatial_two/{s.sample_id}/{role}", "subset": "coco", "dataset": "coco_spatial_two", "sample_id": s.sample_id, "relation": s.relation, "role": role,
                      "object": getattr(s, role), "image_key": f"coco:{im}", "orig_wh": list(wh), "truth_xyxy": [float(v) for v in getattr(s, f"bbox_{role}")]})
out = {"dataset": "bbox_calib_v1", "seed": SEED, "status": "active", "made_by": "scripts/make_bbox_calib.py",
       "purpose": "框坐标约定标定（只用 dev）；标定用的数，不是论文结果，不用于模型间比较；不得用于任何 test 相关操作",
       "truth": "What's Up：auto_v3 box_gt_xyxy（自动标注，经模型辅助预审和部分人工复核）；COCO：loader 的 bbox（mask 外接框）",
       "n_items": len(items), "n_images": len({i["image_key"] for i in items}), "n_items_by_subset": dict(Counter(i["subset"] for i in items)), "whatsup_per_relation": counts, "coco_pool_images": len(by_img), "items": items}
f = paths.REPO_ROOT / "splits" / "bbox_calib_v1.json"
json.dump(out, open(f, "w"), indent=0, ensure_ascii=False)
print({k: v for k, v in out.items() if k != "items"})
