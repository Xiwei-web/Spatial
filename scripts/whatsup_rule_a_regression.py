"""规则 (a) 修正后的回归（CPU，只读）：在 v2 的 820 张上，用已保存的前 5 个候选重算“新规则”的选择，
列出与 v2 实际选择不同的所有图，并与审核结论（annotations/whatsup/review_final_v2.csv）逐张对比。
审核结论的“正确选择”：final_verdict=ok → v2 当前选择正确；否则若审核表给了确定的候选编号 → 以其为准；
其余（none / ? / 空）→ 审核没有给出可比较的候选编号，记为“无法比较”。
输出：docs/parts/rule_a_regression.md、manifests/whatsup_rule_a_regression.json"""
import csv
import json

from src import paths
from src.annotate.selection import select_v2, select_v3

A = [json.loads(l) for l in open(paths.get("data") / "whatsup_annotations" / "auto_v2" / "annotations.jsonl")]
rev = {(r["dataset"], r["sample_id"]): r for r in csv.DictReader(open(paths.REPO_ROOT / "annotations" / "whatsup" / "review_final_v2.csv", encoding="utf-8-sig"))}
rows, stats = [], {"n": len(A), "v2_rule_reproduced": 0, "new_differs": 0, "agree": 0, "disagree": 0, "not_comparable": 0, "new_reason": {}}
for r in A:
    T, R = r["objects"]["target"], r["objects"]["reference"]
    t2, r2, _ = select_v2(T["candidates"], R["candidates"])
    stats["v2_rule_reproduced"] += (t2, r2) == (T.get("selected_candidate_rank") or 0, R.get("selected_candidate_rank") or 0)
    cur = (T.get("selected_candidate_rank") or 0, R.get("selected_candidate_rank") or 0)
    t3, r3, why = select_v3(T["candidates"], R["candidates"])
    stats["new_reason"][why] = stats["new_reason"].get(why, 0) + 1
    if (t3, r3) == cur:
        continue
    stats["new_differs"] += 1
    v = rev[(r["dataset"], r["sample_id"])]
    tp, rp = v["target_pick"].strip(), v["reference_pick"].strip()
    if v["final_verdict"] == "ok":
        truth, basis = cur, "审核 ok（v2 选择正确）"
    elif tp.isdigit() and rp.isdigit():
        truth, basis = (int(tp) - 1, int(rp) - 1), f"审核改选 T{tp}/R{rp}"
    else:
        truth, basis = None, f"审核未给出可比较的编号（T='{tp}' R='{rp}'，{v['final_verdict']}）"
    verdict = "无法比较" if truth is None else ("一致" if truth == (t3, r3) else "不一致")
    stats[{"一致": "agree", "不一致": "disagree", "无法比较": "not_comparable"}[verdict]] += 1
    rows.append([v["order"], r["dataset"], r["sample_id"], f"T{cur[0] + 1}/R{cur[1] + 1}", f"T{t3 + 1}/R{r3 + 1}", why,
                 f"{T['candidates'][0]['score']:.2f}", f"{R['candidates'][0]['score']:.2f}", v["final_verdict"], v["review_source"], basis, verdict])
rows.sort(key=lambda x: int(x[0]))
# 审核判定需要改选、而新规则没有改动的图（新规则漏掉的）
missed = []
for r in A:
    v = rev[(r["dataset"], r["sample_id"])]
    tp, rp = v["target_pick"].strip(), v["reference_pick"].strip()
    if v["final_verdict"] != "ok" and tp.isdigit() and rp.isdigit():
        T, R = r["objects"]["target"], r["objects"]["reference"]
        new = select_v3(T["candidates"], R["candidates"])[:2]
        if new != (int(tp) - 1, int(rp) - 1):
            missed.append([v["order"], r["sample_id"], f"审核 T{tp}/R{rp}", f"新规则 T{new[0] + 1}/R{new[1] + 1}", v["note"][:60]])
stats["review_picks_not_reproduced_by_new_rule"] = len(missed)
json.dump({"stats": stats, "rows": rows, "missed": missed}, open(paths.REPO_ROOT / "manifests" / "whatsup_rule_a_regression.json", "w"), indent=1, ensure_ascii=False)
md = ["# 规则 (a) 修正后的回归对比", "",
      "脚本 `scripts/whatsup_rule_a_regression.py`；规则实现 `src/annotate/selection.py`（`select_v2` / `select_v3`）。只用 v2 已保存的前 5 个候选重算，没有重新检测。", "",
      f"- v2 的 820 张里，用保存的候选能复现 v2 实际选择的：{stats['v2_rule_reproduced']} / {stats['n']}",
      f"- 新规则触发情况：{stats['new_reason']}",
      f"- 新规则与 v2 选择不同的图：**{stats['new_differs']}** 张；与审核结论一致 **{stats['agree']}**，不一致 **{stats['disagree']}**，无法比较 **{stats['not_comparable']}**",
      f"- 审核给出了确定改选编号、而新规则得不到同样结果的图：{len(missed)} 张（见第二张表）", "",
      "| order | 数据集 | 样本 | v2 选择 | 新规则选择 | 新规则动作 | target 最高分 | reference 最高分 | 审核结论 | 审核来源 | 审核的正确选择 | 对比 |", "|" + "---|" * 12]
md += ["| " + " | ".join(map(str, x)) + " |" for x in rows]
md += ["", "## 审核要求改选、新规则没有得到同样结果的图", "", "| order | 样本 | 审核 | 新规则 | 审核备注 |", "|---|---|---|---|---|"]
md += ["| " + " | ".join(map(str, x)) + " |" for x in missed] or ["| （无） | | | | |"]
(paths.REPO_ROOT / "docs" / "parts" / "rule_a_regression.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
