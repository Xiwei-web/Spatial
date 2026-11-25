# 规则 (a) 修正后的回归对比

脚本 `scripts/whatsup_rule_a_regression.py`；规则实现 `src/annotate/selection.py`（`select_v2` / `select_v3`）。只用 v2 已保存的前 5 个候选重算，没有重新检测。

- v2 的 820 张里，用保存的候选能复现 v2 实际选择的：820 / 820
- 新规则触发情况：{'top_score': 752, 'target_moved': 52, 'no_alternative_for_target': 9, 'reference_moved': 7}
- 新规则与 v2 选择不同的图：**7** 张；与审核结论一致 **6**，不一致 **0**，无法比较 **1**
- 审核给出了确定改选编号、而新规则得不到同样结果的图：1 张（见第二张表）

| order | 数据集 | 样本 | v2 选择 | 新规则选择 | 新规则动作 | target 最高分 | reference 最高分 | 审核结论 | 审核来源 | 审核的正确选择 | 对比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | whatsup_b | bowl_in-front_of_can | T1/R1 | T1/R2 | reference_moved | 0.92 | 0.26 | wrong_object | ai_prereview_with_human_rules | 审核改选 T1/R2 | 一致 |
| 7 | whatsup_b | cap_in-front_of_can | T1/R1 | T1/R2 | reference_moved | 0.85 | 0.29 | wrong_object | ai_prereview_with_human_rules | 审核改选 T1/R2 | 一致 |
| 8 | whatsup_b | mug_in-front_of_can | T1/R1 | T1/R2 | reference_moved | 0.88 | 0.27 | wrong_object | ai_prereview_with_human_rules | 审核改选 T1/R2 | 一致 |
| 13 | whatsup_b | mug_right_of_cup | T2/R1 | T1/R2 | reference_moved | 0.80 | 0.59 | wrong_object | ai_prereview_with_human_rules | 审核改选 T1/R2 | 一致 |
| 21 | whatsup_b | cup_in-front_of_can | T2/R1 | T1/R2 | reference_moved | 0.79 | 0.30 | wrong_object | ai_prereview_with_human_rules | 审核改选 T1/R2 | 一致 |
| 25 | whatsup_b | mug_left_of_cup | T2/R1 | T1/R2 | reference_moved | 0.80 | 0.61 | wrong_object | ai_prereview_with_human_rules | 审核改选 T1/R2 | 一致 |
| 682 | whatsup_b | mug_in-front_of_cup | T2/R1 | T1/R2 | reference_moved | 0.79 | 0.62 | wrong_object | human_confirmed | 审核未给出可比较的编号（T='' R=''，wrong_object） | 无法比较 |

## 审核要求改选、新规则没有得到同样结果的图

| order | 样本 | 审核 | 新规则 | 审核备注 |
|---|---|---|---|---|
| 66 | photograph_under_chair | 审核 T2/R1 | 新规则 T1/R1 | 按口径：白边计入。当前T1只框中心画面，应改选T2(含白边) |
