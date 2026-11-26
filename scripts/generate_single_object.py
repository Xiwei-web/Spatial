"""单物体定位检查图像（src/synth/single_object.py）→ <data>/synth/single_object_v2/{images, metadata.jsonl}、manifests/single_object_v2.json。CPU，计算节点上跑。"""
import hashlib
import json
from collections import Counter

import yaml

from src import paths
from src.synth import generator_v2 as G
from src.synth import single_object as S

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "synth" / "synth_v2.yaml"))
OUT = paths.get("data") / "synth" / "single_object_v2"
(OUT / "images").mkdir(parents=True, exist_ok=True)
lib = G.LibraryV3()
metas = []
for r in S.design(CFG):
    m, png = S.generate(CFG, lib, r["index"])
    (OUT / m["image"]).write_bytes(png)
    metas.append(m)
with open(OUT / "metadata.jsonl", "w") as fo:
    fo.writelines(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n" for m in metas)
same = sum(S.generate(CFG, lib, i)[1] == (OUT / metas[i]["image"]).read_bytes() for i in range(0, len(metas), len(metas) // 20))
rep = {"n": len(metas), "by_size": dict(Counter(m["sqrt_area_px"] for m in metas)), "n_categories": len({m["object"]["category"] for m in metas}),
       "by_background": dict(Counter(m["background"]["type"] if "type" in m["background"] else str(m["background"])[:20] for m in metas)),
       "determinism_identical": same, "metadata_sha256": hashlib.sha256((OUT / "metadata.jsonl").read_bytes()).hexdigest(), "library_sha256": lib.sha256}
json.dump(rep, open(paths.REPO_ROOT / "manifests" / "single_object_v2.json", "w"), indent=1, ensure_ascii=False)
print(rep)
