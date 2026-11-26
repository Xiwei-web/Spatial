# E0 方案：能力筛选与行为基线（草案，待 Xiwei 审批）

2026-11-26，hpc-runner 起草（planner 第 5 轮指令；第 7 轮修改：COCO/GQA/COCO-Pairs 改为二选一，见 2.1）。只做了 CPU 上的统计和估算：没有提交作业，没有读任何 test 上的模型输出。数字来源：`scripts/e0_estimate.py` → `manifests/e0_estimate.json`。

## 0. 要先定的四件事

1. **用哪一档**（第 5 节）：全量约 140 GPU 小时；中档约 62 小时；省钱档约 42.5 小时。三档都超过 24 小时。
2. **入选判定用 dev 还是 test**：入选是“选择”操作，按硬性规则 12 应只用 dev。建议：入选按 What's Up **dev** 判定，论文第一张表的数字来自 test（第 7 节）。
3. **新提示词**：已写死在 `src/prompts.py`（round_6、round_7），planner 已预审；Xiwei 审批后先在 dev 上只检查解析率（第 2.3 节，约 2.8 GPU 小时）。
4. **GLM-4.6V 的关系回答**：16 个 token 内常常给不出字母（冒烟 13/20 可解析）。处理办法要在 dev 上按事先写好的规则定（第 2.3 节）。

## 1. 模型（16 个，坐标约定和框提示词一律取 `configs/bbox_conventions.yaml` 的 `calibrated` 块）

| 组 | 模型 | 框提示词 | 坐标约定 |
|---|---|---|---|
| 一 | Qwen2.5-VL-7B-Instruct | P0（Qwen 官方） | abs_model_xyxy |
| 一 | Qwen3-VL-8B-Instruct | P1（官方 + “Output only bbox_2d in JSON. Do not output 3D boxes.”） | norm1000_xyxy（规则判未定，planner 裁定） |
| 一 | InternVL3-8B、InternVL3_5-8B | P0（InternVL 官方） | norm1000_xyxy |
| 一 | SenseNova-SI-1.2 / 1.3-InternVL3-8B | P0（通用提示） | range_norm1_else_norm1000（逐条按数值范围） |
| 一 | GLM-4.6V-Flash | P0（GLM 官方） | norm1000_xyxy |
| 一 | VST-7B-RL | P2（通用 + 不输出 3D 框；**看到标定结果后才加**） | abs_model_xyxy（规则判未定，planner 裁定） |
| 一 | RoboBrain2.0-7B | P0（RoboBrain 官方） | abs_model_xyxy（规则判未定，planner 裁定；配置已由原图像素改过来） |
| 二 | Qwen3.5-2B / 4B / 9B | P0（Qwen 官方） | norm1000_xyxy |
| 二 | InternVL3-2B | P0 | norm1000_xyxy |
| 二 | SenseNova-SI-1.1-InternVL3-2B | P1（InternVL 官方） | norm1_xyxy |
| 二 | LLaVA-1.5-7B / 13B | P0 | norm1_xyxy（相对补边后的正方形） |

## 2. 任务

### 2.1 三种问法

- **框预测**：每个有真值框的物体问一次（两物体数据集里 target、reference 各一次）。同一张图里的同一个物体只问一次（按 图像 + 物体名 去重；COCO-Pairs 按 ann_id 去重）。指标按 2.2 节：IoU@0.5/0.7/0.9、平均 IoU、center-in-mask、中心误差、IoU@0.1、左右半区；另外用两个预测框的中心推出关系。
- **关系选择题**：
  - What's Up A/B 是**四选一**，原本就是四个互斥选项。每个样本问 4 次，选项按循环轮换（关系 → 字母的映射依次平移一位）。
  - COCO-spatial（one/two）、GQA-spatial（one/two）、COCO-Pairs 改为**二选一**，只给真关系和同一轴上的反关系，两种顺序各问一次。原因：原始数据每条只有这两个选项，标签只给出一个轴；例如左上角的物体标的是 top，但 left 也成立，四选一会出现两个正确答案，把对的回答判成错（planner 第 7 轮）。二选一的选项列表按轴的固定顺序排（left/right、above/below、top/bottom、front/behind），不按真关系排。
  - “轮换后准确率” = 所有顺序的平均。随机水平：四选一 25%，二选一 50%。同时保存第一步各字母 token 的 logit。
- **成对二元**：同一样本问真关系、反关系各一次（left↔right、on↔under、front↔behind、above↔below、top↔bottom），两问都答对才算对（随机水平 25%）。所有数据集都问。

### 2.2 每个数据集的选项集合与条数（test）

| 数据集（test） | 样本 | 图 | 关系 | 问法 | 不在选项里而剔除 | 框提问（去重，有真值） | 选择题（× 轮换次数） | 二元（真、反） |
|---|---|---|---|---|---|---|---|---|
| whatsup_a | 206 | 206 | left/right/on/under | 四选一 ×4 | 0 | 412 | 824 | 412 |
| whatsup_b | 204 | 204 | left/right/front/behind | 四选一 ×4 | 0 | 408 | 816 | 408 |
| coco_spatial_one | 1125 | 726 | left/right/top/bottom | 二选一（同轴）×2 | 0 | 896 | 2250 | 2250 |
| coco_spatial_two | 232 | 148 | left/right/above/below | 二选一（同轴）×2 | 0 | 346 | 464 | 464 |
| gqa_spatial_one | 579 | 524 | left/right/top/bottom | 二选一（同轴）×2 | 1 | 557 | 1156 | 1156 |
| gqa_spatial_two | 152 | 131 | left/right/front/behind | 二选一（同轴）×2 | 0 | 278 | 304 | 304 |
| coco_pairs | 2574 | 637 | left/right/above/below | 二选一（同轴）×2 | 0 | 1552 | 5148 | 5148 |
| 合计 | 5072 | | | | 4449 | 10962 | 10142 |

- What's Up：A 是 left/right/on/under，B 是 left/right/front/behind（B 的 front/behind 与垂直位置完全混淆，所有文档照注）。框提问只用 auto_v3 可用样本，真值 `box_gt_xyxy`（自动标注，经模型辅助预审和部分人工复核）。
- COCO-spatial：one-object 的关系相对整张图（left/right/top/bottom），two-object 是 left/right/above/below；都是二选一（同轴）。框真值是 mask 外接框，只用类别名唯一匹配的物体（test 全部有框）。
- GQA-spatial：one-object 的 test 里有 1 条 `behind`，不在 left/right/top/bottom 里，**剔除**；two-object 的 dev 里 3 条 `top` 同样剔除（planner 第 6 轮）。物体名照原样用，例如 “black trees” 这类复数带来的语法瑕疵不修。two-object 的 test 只有 left/right/front/behind；框来自 scene graph 的唯一匹配，没有框的不问框。
- COCO-Pairs：2,574 个有向对，互为孪生的两条（A 在 B 下 / B 在 A 上）都问；关系也只按一个轴标，二选一。框按物体去重，共 1,552 个。

### 2.3 提示词和解析的来源，以及还要在 dev 上做的选择

| 项 | 现状 | 是否只在 dev 上定过 |
|---|---|---|
| What's Up 四选一 | `src/prompts.py: relation_mcq`，所有模型同一模板，阶段 A/B/C 时由 runner 事先写好，没有用任何数据挑选 | 模板没有用数据选过。冒烟（What's Up dev 20 张）只用来确认能解析 |
| 四选一解析 `parse_mcq` | 冒烟之后按 dev 上看到的**格式**补过两种包装：`<answer>`（RoboBrain）、`<\|begin_of_box\|>`（GLM）。没有看对错 | 是（dev 20 张，只看格式） |
| 两物体二选一（COCO/GQA/COCO-Pairs） | `e0_mcq`：沿用 What's Up 句式，只保留同轴两行选项（round_6 写、round_7 改） | 事先写死，planner 已预审；没有用数据选 |
| 一物体二选一（相对整张图） | `e0_mcq`（同上） | 同上 |
| 二元问法与 yes/no 解析 | `binary_question`、`parse_yes_no`：先删掉字面短语 “yes or no” 再认第一个 yes/no 词 | 同上 |
| 框提示词与坐标约定 | `calibrated` 块，dev 标定（第 3、4、5 轮） | 是 |

**正式跑之前要在 dev 上做的选择**（都不看对错，只看能否解析；规则事先写进代码并提交）：

1. **三类新模板已写死并提交**（round_6、round_7，全文见 `reports/round_7.md`）。
2. **dev 解析检查**：每个数据集的 dev 取 50 个样本，选择题 1 次（固定顺序）加二元 2 次，16 个模型共约 **2.8 GPU 小时**。事先定好标准：解析率 ≥ 95% 才过。不过的模型只允许按第 3 条处理，不改模板措辞。
3. **答案获取方式的顺序**（已写进代码，`ANSWER_MODE_ORDER`）：默认 text16 → text64 → letter_logit，GLM-4.6V-Flash 从 text64 开始（planner 第 6 轮）。dev 上解析率 ≥ 95% 就停在当前一步。
   - letter_logit：每个字母取 “A”、“ A” 两种变体里 logit 最大的再比较；二元比较 {Yes, yes, “ Yes”, “ yes”} 的最大值与 {No, no, “ No”, “ no”} 的最大值；分词器里不是单个 token 的变体跳过。
   - 冒烟时 GLM 在 16 个 token 内多为 “The mug is located to the right of …”，没来得及给出字母。text64 会让 GLM 的关系耗时增加约 2–3 倍，下表没算进去。
4. GQA 剔除的 4 条已定（见 2.2）。

## 3. 实现

- **主跑**：bf16 + eager + fp32 打分补丁（`patch_eager_fp32`），贪心解码；框 max_new_tokens 96，关系 16（GLM 见 2.3）。与之后干预实验的基线是同一种实现、同一种精度（硬性规则 14 的前半）。
- **默认实现那一遍**（硬性规则 14 的后半）：每个模型另跑一次 sdpa、无干预。之前的检查统一用 sdpa 作“默认”参照；InternVL 远程代码自己的默认是 flash-attn（这里没装），沿用 sdpa 并写明。成本单独列在第 5 节。耗时按与主跑相同估算，这是上界；实测过的一例是 Qwen2.5-VL 的 bf16 teacher-forced，sdpa 约为 eager 的 0.73 倍。
- 每个模型一个作业（1 卡，cscc-gpu-p 批处理 QoS），断点续跑；提交前先 commit，记录里写干净的 commit。
- **图像前缀 KV cache 复用：不实现。** 原因：实现有风险（Qwen3.5 的递归状态缓存、InternVL 远程代码走 inputs_embeds 的生成，都要自己写）；而且 bf16 下复用与否输出不逐 token 相同。可行性和节省的估算在 `reports/round_6.md`，仅供参考。

## 4. 输出与统计

- 逐样本 JSONL：`<outputs>/e0/<模型>/<数据集>.jsonl`。每条带原始文本、解析结果、选项顺序（轮换编号与关系 → 字母映射）、四个字母 token 的 logit、提示词全文、模型 revision、config hash、git commit、作业号、输入尺寸与视图框、实现（eager / sdpa）、精度、耗时、出错信息。解析失败和出错照记，计入分母。
- 表和图只由脚本从 JSONL 生成。统计按图像 bootstrap 10,000 次（同图的样本一起重采样），给 95% 区间；模型之间的比较用配对 bootstrap 或 McNemar。

## 5. GPU 估算

秒数来自已有记录。框用 bbox_calib 里选定提示词的平均秒数，What's Up 与 COCO 分开；关系用冒烟 relation_mcq 的平均秒数，其余数据集按框的耗时比缩小。每个模型每个作业另加 0.05 小时加载。

| 模型 | 框秒数 WU / 其他 | 关系秒数 WU / 其他 | 全量主跑 | 默认实现一遍（全量） | 省钱主跑 | 省钱默认实现（只 WU） | dev 筛选（WU） | dev 解析检查 | 合成 / 1000 图 | 深度 / 100 条 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 1.74 / 1.27 | 0.73 / 0.53 | 4.98 | 4.98 | 1.39 | 0.94 | 0.94 | 0.22 | 2.18 | 0.22 |
| Qwen3-VL-8B-Instruct | 2.04 / 1.75 | 0.44 / 0.38 | 4.55 | 4.55 | 1.28 | 0.82 | 0.81 | 0.17 | 1.87 | 0.19 |
| InternVL3-8B | 0.88 / 0.85 | 0.16 / 0.15 | 2.02 | 2.02 | 0.57 | 0.36 | 0.36 | 0.10 | 0.76 | 0.08 |
| InternVL3_5-8B | 1.25 / 1.20 | 0.16 / 0.16 | 2.47 | 2.47 | 0.72 | 0.45 | 0.44 | 0.10 | 0.96 | 0.10 |
| SenseNova-SI-1.2-InternVL3-8B | 0.88 / 0.83 | 0.47 / 0.45 | 3.73 | 3.73 | 0.91 | 0.57 | 0.57 | 0.18 | 1.28 | 0.13 |
| SenseNova-SI-1.3-InternVL3-8B | 0.90 / 0.71 | 0.46 / 0.36 | 3.17 | 3.17 | 0.85 | 0.57 | 0.57 | 0.16 | 1.26 | 0.13 |
| GLM-4.6V-Flash | 1.83 / 1.42 | 0.87 / 0.67 | 5.98 | 5.98 | 1.59 | 1.06 | 1.06 | 0.26 | 2.46 | 0.25 |
| VST-7B-RL | 1.40 / 0.67 | 0.70 / 0.33 | 3.24 | 3.24 | 1.10 | 0.85 | 0.84 | 0.18 | 1.94 | 0.19 |
| RoboBrain2.0-7B | 1.20 / 0.72 | 0.95 / 0.57 | 4.67 | 4.67 | 1.34 | 0.98 | 0.97 | 0.25 | 2.26 | 0.23 |
| Qwen3.5-2B | 1.81 / 1.61 | 0.46 / 0.41 | 4.52 | 4.52 | 1.23 | 0.78 | 0.78 | 0.17 | 1.78 | 0.18 |
| Qwen3.5-4B | 2.44 / 2.22 | 0.95 / 0.86 | 7.95 | 7.95 | 2.00 | 1.25 | 1.25 | 0.31 | 2.93 | 0.29 |
| Qwen3.5-9B | 2.50 / 2.23 | 0.61 / 0.55 | 6.13 | 6.13 | 1.66 | 1.04 | 1.03 | 0.22 | 2.41 | 0.24 |
| InternVL3-2B | 0.85 / 0.81 | 0.10 / 0.10 | 1.64 | 1.64 | 0.50 | 0.31 | 0.31 | 0.08 | 0.64 | 0.06 |
| SenseNova-SI-1.1-InternVL3-2B | 0.93 / 0.89 | 0.43 / 0.41 | 3.57 | 3.57 | 0.88 | 0.55 | 0.55 | 0.17 | 1.22 | 0.12 |
| LLaVA-1.5-7B | 0.76 / 0.74 | 0.15 / 0.14 | 1.79 | 1.79 | 0.51 | 0.32 | 0.32 | 0.09 | 0.67 | 0.07 |
| LLaVA-1.5-13B | 1.01 / 0.98 | 0.22 / 0.22 | 2.54 | 2.54 | 0.69 | 0.43 | 0.43 | 0.11 | 0.93 | 0.09 |
| **合计（GPU 小时）** | | | **62.9** | **62.9** | **17.2** | **11.3** | **11.2** | **2.8** | **25.6** | **2.6** |

各数据集在全量主跑里的份额（16 个模型合计，不含加载）：What's Up A 5.3、B 5.2、COCO one 12.6、COCO two 3.4、GQA one 7.0、GQA two 2.5、**COCO-Pairs 26.2** 小时。

| 档 | 内容 | GPU 小时 | 墙钟（2 个作业同时跑，不含排队） | 取舍 |
|---|---|---|---|---|
| 全量 | 7 个数据集 × 3 种问法主跑（62.9）+ 默认实现全量（62.9）+ dev 筛选（11.2）+ dev 解析检查（2.8） | **≈ 140** | ≈ 70 小时 | 一切齐全，最贵 |
| 中档 | 主跑去掉 COCO-Pairs（36.7；COCO-Pairs 留给 E6）+ 默认实现只跑 What's Up（11.3）+ dev 筛选（11.2）+ 解析检查（2.8） | **≈ 62** | ≈ 31 小时 | COCO-Pairs 推迟到 E6 再跑；默认实现只覆盖主数据 |
| 省钱 | 主跑只跑 What's Up A/B、COCO-spatial two、GQA-spatial two（17.2）+ 默认实现只跑 What's Up（11.3）+ dev 筛选（11.2）+ 解析检查（2.8） | **≈ 42.5** | ≈ 21 小时 | 不报 one-object 子集（与 What's Up 原论文的对照不全）；COCO-Pairs 推迟到 E6 |

- **合成集**（等 Xiwei 决定）：每 1,000 张图（两个物体：框 2 次 + 选择题按四选一 4 次计、偏上界 + 二元 2 次），16 个模型合计约 25.6 GPU 小时（按 What's Up 的秒数，偏上界；合成图是 896×896）。trial_v1 只有 500 个场景 + 孪生共 1,500 张，全量尚未生成。
- **深度集**（来源等 Xiwei 决定）：每 100 条约 2.6 GPU 小时。
- 误差来源：关系秒数只来自冒烟 20 条；GLM 若把 max_new_tokens 提到 64，要另加；排队时间不可预测。

## 6. 入选标准（第 2.3 节）与建议

- 标准：开放权重、能取内部状态、不微调就能输出框；轮换后关系四选一准确率 ≥ 80%；IoU@0.5 ≥ 70%（不够时可放宽到 60%，照实报告）；入选模型覆盖至少 3 个家族。
- 注意：80% 这条线是按四选一（随机水平 25%）写的。按 What's Up 判定时仍是四选一；COCO/GQA/COCO-Pairs 改为二选一后随机水平是 50%，同一条 80% 线在那里更容易达到，不宜直接套用。
- **建议以 What's Up（A+B 合并）为判定数据集**：
  1. 它是机制实验（E2–E5）的主数据，入选要看的就是模型在这里能否同时做对框和关系；
  2. 框真值来自 v3 定稿的 mask（自动标注，经模型辅助预审和部分人工复核），关系标签由数据集本身给出，两者对得上；COCO-spatial 的框靠类别名唯一匹配，GQA 的框来自 scene graph，噪声来源更多；
  3. 图像受控、物体清楚，避免把检测难度混进“会不会”的判断。
- 局限，照实写：A 与 B 合计只有 410 张；B 的 front/behind 与垂直位置完全混淆；A 有 on/under。另外单独报告只看 left/right 的准确率作参考。
- **用 dev 判定**：入选是选择类操作（硬性规则 12）。建议在 What's Up dev（408 个样本，约 11.2 GPU 小时）上算入选指标、做决定，论文第一张表报 test。如果 Xiwei 认为 E0 的表本身就是入选依据、可以直接用 test，dev 筛选这一项可以省掉。
- 家族划分（覆盖 ≥ 3 个）：Qwen 系（Qwen2.5-VL、Qwen3-VL、Qwen3.5）、Qwen2.5-VL 微调系（VST、RoboBrain）、InternVL 系（InternVL3/3.5、SenseNova-SI）、GLM、LLaVA。微调模型是否算独立家族，需要定。
