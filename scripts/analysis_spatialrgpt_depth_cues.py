"""SpatialRGPT 深度样本（前/后）的同一项捷径检查（与 What's Up B 的 F6-1 相同；CPU，不跑任何模型）：
front/behind 在多大程度上能只凭两个 mask 的垂直位置差和面积比猜出来。样本少（跳过同名类别后 96 条），按图像来源分组交叉验证以避免同图泄漏。
输出 manifests/spatialrgpt_depth_cues.json"""
import json

import numpy as np
from pycocotools import mask as mu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import paths
from src.data.spatialrgpt import depth_samples, load_records

src_img = {r.id: r.extra["source_file_path"] for r in load_records()}
S = depth_samples()


def feats(s):
    out = []
    for rle in (s.mask_target, s.mask_reference):
        m = mu.decode(rle).astype(bool)
        ys, _ = np.nonzero(m)
        out.append((ys.mean(), ys.max() + 1, m.sum(), m.shape[0]))
    (yt, bt, at, H), (yr, br, ar, _) = out
    return (yt - yr) / H, (bt - br) / H, float(np.log(at / ar))


F = np.array([feats(s) for s in S])
y = np.array([s.relation == "front" for s in S]).astype(int)
bare = np.array([s.extra["relation_inferred_from_bare_no"] for s in S])
grp = np.array([src_img[s.sample_id] for s in S])
res = {"truth": "mask；真值关系见 src/data/spatialrgpt.py", "subsets": {}}
for tag, sel in (("all_used", np.ones(len(S), bool)), ("explicit_only", ~bare), ("inferred_from_bare_no_only", bare)):
    f, yy, g = F[sel], y[sel], grp[sel]
    r = {"n": int(sel.sum()), "n_front": int(yy.sum()), "n_source_images": len(set(g)), "majority_baseline": round(float(max(yy.mean(), 1 - yy.mean())), 3),
         "rule_target_centroid_lower_is_front": round(float(((f[:, 0] > 0).astype(int) == yy).mean()), 3),
         "rule_target_bottom_lower_is_front": round(float(((f[:, 1] > 0).astype(int) == yy).mean()), 3),
         "rule_target_larger_is_front": round(float(((f[:, 2] > 0).astype(int) == yy).mean()), 3), "cv": {}}
    for name, cols in (("centroid_dy", [0]), ("bottom_dy", [1]), ("log_area_ratio", [2]), ("centroid_dy+log_area_ratio", [0, 2]), ("all_three", [0, 1, 2])):
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        a = cross_val_score(clf, f[:, cols], yy, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=20261125))
        k = cross_val_score(clf, f[:, cols], yy, cv=GroupKFold(n_splits=5), groups=g)
        r["cv"][name] = {"stratified_5fold_x20": round(float(a.mean()), 3), "group_by_source_image": round(float(k.mean()), 3)}
    res["subsets"][tag] = r
    print(tag, r)
json.dump(res, open(paths.REPO_ROOT / "manifests" / "spatialrgpt_depth_cues.json", "w"), indent=1, ensure_ascii=False)
