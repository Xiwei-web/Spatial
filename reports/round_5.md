# Round 5：两个裁定写入配置；E0 方案与 GPU 估算（只用 CPU）

状态：**完成**（2026-11-26）。没有提交作业，没有读 test 上的任何模型输出（只读了 test 划分的条数）。

## 一句话结论

两个裁定已写入配置（`7ba04a4`）。E0 方案写在 `docs/E0_PLAN.md`：全量约 173 GPU 小时，中档约 69，省钱档约 44，三档都超过 24 小时，需要 Xiwei 定。

## 一、裁定（提交 `7ba04a4`；`src/bbox_calib.py` 没改）

- VST-7B-RL：`calibrated` 块的 prompt_id 从 pending 改为 P2，约定 abs_model_xyxy。basis 写“规则判未定，planner 裁定（round_5）”，以及理由和“P2 是看到第 3 轮结果后才加的（论文里也要写）；剩余 73 条 3D 输出记解析失败”。
- RoboBrain2.0-7B：维持 abs_model_xyxy。basis 原样写“COCO 上 +0.006 [+0.002, +0.009]，差值显著但幅度可忽略（COCO 图缩放比偏离 1 不超过 3.8%）”，以及 What's Up 上 −0.427。
- 测试 `tests/test_bbox_calib.py` 改为要求 16 个模型都不再是 pending，10 项通过。

## 二、E0 方案（`docs/E0_PLAN.md`）

- 估算：`python scripts/e0_estimate.py` → `manifests/e0_estimate.json`。秒数取自 bbox_calib（框）和冒烟（关系）记录。
- 主要数字：test 共 5,072 个样本；框提问 4,449 次，四选一 20,284 次，二元 10,142 次（每个模型）。全量主跑 79.3 GPU 小时，其中 COCO-Pairs 占 35.2；默认实现那一遍按上界再算 79.3。
- 方案里列出了：提示词与解析的来源（What's Up 四选一模板事先写好、没用数据选过；解析器按 dev 冒烟输出补过格式）；还没有的三类提示词（两物体四选一、一物体四选一、二元问法），以及正式跑前要在 dev 上做的选择（解析检查约 2.8 GPU 小时）；GLM 关系解析的两个备选；入选按 What's Up 判定的建议和局限。

## 三、需要决定的事（转 Xiwei）

1. 用哪一档：全量约 173 / 中档约 69 / 省钱约 44 GPU 小时。墙钟按 2 个作业同时跑，分别约 87 / 35 / 22 小时，不含排队。
2. 入选按 What's Up dev 判定（我的建议，依据硬性规则 12），还是直接用 test。
3. 三类新提示词的措辞（我写死后提交给你看，再在 dev 上只查解析率），以及解析率 ≥ 95% 这条线。
4. GLM 关系回答的两个备选：加长 max_new_tokens，或用字母 logit。
5. GQA one-object 那 1 条 `behind` 剔除。
6. 合成集、深度集的规模与来源；估算单价为每 1,000 张合成图约 25.6 GPU 小时，每 100 条深度约 2.6 GPU 小时。
7. 默认实现那一遍跑哪些数据（全量还是只跑 What's Up）。
8. 微调模型（VST、RoboBrain、SenseNova-SI）是否算独立家族。
