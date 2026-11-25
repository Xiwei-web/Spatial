"""F6-1：What's Up B 的深度线索分析（CPU，不跑任何模型）。
问题：front/behind 能在多大程度上仅凭两个物体 mask 的“垂直位置差”和“面积比”猜出来？
特征（只有两个）：dy = (target 质心 y − reference 质心 y) / 图像高度；log_area_ratio = ln(target mask 面积 / reference mask 面积)
模型：标准化 + 逻辑回归。评估：分层 5 折交叉验证（重复 20 次），以及按 (target, reference) 物体组合分组的 5 折
（同一对物体的 front/behind 两张图不会同时出现在训练和测试里）。
标注来自 F1 的“自动标注”（未经人工验证）；同时给出“全部样本”和“仅自动质检通过”的结果。
用法：source scripts/env.sh tools && python scripts/analysis_whatsup_b_depth_cues.py"""
import json

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import paths

cfg = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "annotation" / "whatsup_auto.yaml"))
rows = [json.loads(l) for l in open(paths.get("data") / "whatsup_annotations" / cfg["version"] / "annotations.jsonl")]
rows = [r for r in rows if r["dataset"] == "whatsup_b" and r["relation"] in ("front", "behind")]
out = {"annotation": f"{cfg['version']} (automatic, not human-verified)", "n_front_behind_total": len(rows), "subsets": {}}
for tag, sub in (("all_with_both_masks", [r for r in rows if all(r["objects"][k]["mask_area"] for k in ("target", "reference"))]),
                 ("auto_qc_pass_only", [r for r in rows if r["auto_qc"] == "pass"])):
    T = [r["objects"]["target"] for r in sub]; R = [r["objects"]["reference"] for r in sub]
    dy = np.array([(t["mask_centroid_xy"][1] - q["mask_centroid_xy"][1]) / r["height"] for t, q, r in zip(T, R, sub)])
    la = np.log(np.array([t["mask_area"] / q["mask_area"] for t, q in zip(T, R)]))
    y = np.array([r["relation"] == "front" for r in sub]).astype(int)
    groups = np.array([f"{t['name']}|{q['name']}" for t, q in zip(T, R)])
    res = {"n": len(sub), "n_front": int(y.sum()), "n_behind": int((1 - y).sum()), "n_object_pairs": len(set(groups)),
           "majority_baseline": float(max(y.mean(), 1 - y.mean())),
           "feature_stats": {"dy_front_mean": float(dy[y == 1].mean()), "dy_behind_mean": float(dy[y == 0].mean()),
                             "log_area_ratio_front_mean": float(la[y == 1].mean()), "log_area_ratio_behind_mean": float(la[y == 0].mean()),
                             "share_front_with_target_lower_in_image(dy>0)": float((dy[y == 1] > 0).mean()),
                             "share_behind_with_target_higher_in_image(dy<0)": float((dy[y == 0] < 0).mean())},
           "cv_accuracy": {}}
    for fname, X in (("dy_only", dy[:, None]), ("log_area_ratio_only", la[:, None]), ("dy+log_area_ratio", np.c_[dy, la])):
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        a = cross_val_score(clf, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=20261125))
        g = cross_val_score(clf, X, y, cv=GroupKFold(n_splits=5), groups=groups)
        res["cv_accuracy"][fname] = {"stratified_5fold_x20_mean": float(a.mean()), "std_over_folds": float(a.std()),
                                     "group_5fold_by_object_pair_mean": float(g.mean()), "group_fold_min": float(g.min()), "group_fold_max": float(g.max())}
    # 最简单的单阈值规则：target 质心更低（dy>0）就判 front
    res["rule_dy_gt_0_is_front_accuracy"] = float(((dy > 0).astype(int) == y).mean())
    out["subsets"][tag] = res
json.dump(out, open(paths.REPO_ROOT / "manifests" / "whatsup_b_depth_cues.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
