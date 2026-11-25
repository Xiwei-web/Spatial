# 数据构建报告（DATA_REPORT，阶段 F）

日期：2026-11-25。范围：冒烟测试收尾（第 1 节）和数据构建 F1–F7。**本阶段没有在任何数据上计算模型正确率，没有调整任何提示词，没有生成合成集全量，没有对 COCO-Pairs 下采样，没有生成 LaMa 抹除图。**

**What's Up 的框和 mask 是“自动标注”（GroundingDINO + SAM 2.1），未经人工验证。** 在你填完 `review.csv` 之前，凡是依赖它的结果（F4 抠图库的 What's Up 部分、F6 两项分析）都建立在自动标注上。

各部分的详细文档在 `docs/parts/`：`F2_gqa_boxes.md`、`F3_coco_pairs.md`、`F4_synth.md`、`F5_coco_holdout.md`、`F7_spatialrgpt.md`。

你对上一份报告第 6 节的决定已落实：未下载 Gemma-3；只下载了 SpatialRGPT-Bench 完整数据；没有使用任何 ImageNet 副本；下载了 `sam2.1-hiera-large`（1.80 GB）和 `big-lama`（0.38 GB），GroundingDINO 只读使用你的本地权重和代码；LLaVA-1.5 改为先补成正方形；Cambrian-S 已从候选中去掉；没有删除任何文件。

---

## 1. 冒烟测试收尾

### 1.1 一个影响后续所有机制实验的发现：eager 注意力在 bf16 下会损坏部分模型的输出（已修复）

上一轮冒烟为了取注意力权重，统一用 `attn_implementation="eager"` + bf16 加载。排查 RoboBrain 的“输出截断”时发现这才是原因：

| 设置（RoboBrain2.0-7B，同一批 40 条框提示词，贪心） | 能解析出四个数 |
|---|---|
| 视觉塔 eager + 语言模型 eager，bf16（上一轮冒烟的设置） | 15 / 40 |
| 全部 sdpa，bf16 | 40 / 40 |
| 全部 sdpa，float32 | 40 / 40 |
| 视觉塔 eager + 语言模型 sdpa | 40 / 40 |
| 视觉塔 sdpa + 语言模型 eager | 14 / 40 |
| 视觉塔 sdpa + 语言模型 eager，但 q·k 打分在 float32 里算 | 40 / 40 |
| 视觉塔 eager + 语言模型 eager（打分在 float32 里算） | 40 / 40 |

- 定位：问题在**语言模型**的 eager 注意力（transformers 的 `eager_attention_forward` 在 bf16 里做 `matmul(q, k) * scaling`），与视觉塔无关。
- 修复：加载后把语言模型 eager 注意力的打分和 softmax 放到 float32 里计算，权重和 value 仍是 bf16（`src/models/adapter.py: patch_eager_fp32`）。注意力权重仍可取出，hook 不受影响。float32 模型上新旧实现数值一致（单元测试 `test_eager_fp32_patch_matches_reference_in_float32`）。
- 修复后与全 sdpa 的差异：末位置 logits 最大差 0.30～0.63（旧设置为 6.9），40 条输出里 22～23 条与全 sdpa 逐字相同（bf16 下两种实现本来就不是逐位相同的）。
- 我没有查明 bf16 下 eager 为什么会坏、sdpa 为什么不会（两者算式相同，只是缩放的位置不同）；上表是经验定位，不是机理解释。
- 这个补丁只在注意力模块所在的源码文件提供 `eager_attention_forward` 时生效；每个模型是否生效记录在冒烟 `summary.json` 的 `eager_fp32_scoring` 字段里。
- 证据文件：`outputs/diagnostics/attn_dtype_RoboBrain2.0-7B.json`、`eager_localize_RoboBrain2.0-7B.json`。只在 RoboBrain 上做了定位；其它模型是通过“修复后重跑冒烟”间接验证的（见 1.2）。

**所有模型都在修复后的设置下重跑了冒烟**，下表是重跑结果。

### 1.2 冒烟最终表

“通过”只表示管线跑通、状态拿得到，不表示框或答案正确；没有计算正确率。第 2、3 项旁边的数字是解析成功的次数。组别是你给的暂定分组。

<!-- SMOKE_TABLE_BEGIN -->
<!-- SMOKE_TABLE_END -->

<!-- SMOKE_NOTES -->

### 1.3 三个问题的原因

**(1) RoboBrain2.0-7B 输出截断：三个因素都不是原因。** 在旧设置下每次只改一个：

| 设置 | 能解析出四个数 | 平均生成 token 数 |
|---|---|---|
| 旧设置基线（`max_new_tokens=96`、默认 EOS、`repetition_penalty=1.0`） | 15 / 40 | 19.2 |
| 只改 `max_new_tokens=256` | 15 / 40 | 23.2 |
| 只改 EOS 为仅 `<|im_end|>` | 15 / 40 | 19.2 |
| 只关闭 EOS | 15 / 40 | 96.0（截断处之后是一串 `<|im_end|>`，没有后续坐标） |
| 只改 `repetition_penalty` 为仓库自带的 1.05 | 13 / 40 | 20.7 |

旧设置下重复一次，40 条逐 token 相同（可复现）。原因是 1.1 节的 eager + bf16；修复后 40 / 40。完整记录：`outputs/diagnostics/bbox_truncation_RoboBrain2.0-7B.json`（旧设置）。

**(2) Qwen3-VL-8B 输出 3D 框：只记录，未调提示词。** 修复后仍然存在：40 次里 11 次解析不出四个数，其中 10 次输出键为 `bbox_3d` 的 9 个数，1 次键为 `bbox_2d` 但也是 9 个数。触发的提示词是官方 cookbook 的 `Locate {object}, output its bbox coordinates using JSON format.`；涉及的物体：armchair、chair、table、knife、mug、pillow、remote、scissors、sunglasses；Subset A 9 次、Subset B 2 次；target 5 次、reference 6 次。原始输出在 `outputs/smoke/Qwen3-VL-8B-Instruct/records.jsonl`。

**(3) Qwen2.5-VL-7B 的 1/40 截断后重复特殊 token：旧设置下可复现，修复后消失。** 样本是 `whatsup_b / can_in-front_of_flower` 的 target（“Locate can, …”），输出为 `[539, 423, 690, 7` 后接重复的 `<|im_start|>`。旧设置下重复运行逐 token 相同；改 `max_new_tokens`、EOS、`repetition_penalty` 都不消除（都是 39 / 40）。修复后的冒烟里是 40 / 40。完整输入 token 与各设置的输出 token 保存在 `outputs/diagnostics/bbox_truncation_Qwen2.5-VL-7B-Instruct.json`。

### 1.4 视觉 token → 图像区域映射的行为验证（第一组 9 个模型）

做法（`scripts/verify_token_mapping.py`）：冒烟 20 张里固定的 5 张图，每张把一个宽高各占 25% 的矩形（位置由固定种子决定）换成均匀噪声，比较替换前后送入语言模型的视觉 token 嵌入（`hidden_states[0]`）的 L2 变化量。预测集合 P = 映射表里格子面积 ≥ 50% 落在矩形内的 token；实测集合 A = 变化量最大的 |P| 个 token；重合率 = |P ∩ A| / |P|。

<!-- MAPPING_TABLE -->

对比图：`outputs/token_mapping_verify/<模型>/compare.png`（上：噪声矩形；中：映射预测；下：实测变化）。

---

## 2. F1：What's Up Subset A / B 的自动标注

- 方法：GroundingDINO（SwinB，你的本地权重，只读）用数据集标注解析出的物体名称给候选框，取分数最高的；SAM 2.1（hiera-large）用该框做提示生成 mask。物体名称没有手改。配置 `configs/annotation/whatsup_auto.yaml`，脚本 `scripts/annotate_whatsup.py`（1 张卡，585 秒）与 `scripts/whatsup_qc.py`（CPU）。
- 你的 GroundingDINO 仓库没有编译 CUDA 算子；我在自己的包装代码里把可变形注意力固定到仓库自带的纯 PyTorch 实现，没有在你的仓库或权重目录里写任何文件。
- 标注：`data/whatsup_annotations/auto_v1/annotations.jsonl`（每张图：两个物体的框、mask RLE、检测分数、候选框数量、前 5 个候选、质检标记）。

### 2.1 质检统计

| | 总数 | 自动质检通过 | 待审 |
|---|---|---|---|
| Subset A | 412 | 238（57.8%） | 174 |
| Subset B | 408 | 371（90.9%） | 37 |
| 合计 | 820 | 609（74.3%） | **211** |

| 标记（一张图可有多个） | A | B | 合计 |
|---|---|---|---|
| 某个物体没检测到 | 0 | 0 | 0 |
| 多个分数接近的候选（第二个不重合候选与第一名相差 < 0.10） | 62 | 30 | 92 |
| 检测分数 < 0.35 | 29 | 24 | 53 |
| mask 面积 < 0.2% 或 > 60% | 81 | 0 | 81 |
| target 与 reference 的 mask IoU > 0.5 | 60 | 8 | 68 |
| 几何顺序与标签不一致（left/right/on/under） | 60 | 2 | 62 |

- 分数阈值 0.35 的依据：GroundingDINO 官方示例的默认 `box_threshold`；在本数据上约为 target 分数的 5% 分位（p5 = 0.358）。分数分布没有明显双峰（target 分数按 0.1 分箱：0, 1, 19, 62, 86, 103, 113, 119, 256, 61）。
- 待审按关系：left 43、right 52、on 53、under 44、front 7、behind 12。
- 统计文件：`manifests/whatsup_auto_qc_summary.json`。

### 2.2 审核材料

- 联系表：`/nfs-stor/xiwei.liu/interpretability/outputs/annotation_review/sheet_001.png` … `sheet_041.png`（每页 20 张；红 = target，蓝 = reference；标题含物体名、关系、标记）。待审的 211 张排在最前（第 1–11 页），标记多的更靠前。
- 审核表：同目录 `review.csv`，每行一张图，`human_verdict` 与 `note` 两列为空。重跑质检脚本时，如果该表已有填写内容，新表会写到 `review.NEW.csv`，不会覆盖你填的内容。
- **需要你审的最少是 211 张；通过的 609 张我只抽看了一页（第 20 页，20 张都对），建议你也抽查。**

### 2.3 我认为最可能出错的情况

1. **Subset A 里 target 的框落到家具上**：68 张图 target 框与 reference 框 IoU > 0.8（把 armchair / table / chair 当成了小物体）。全部被质检抓到。其中 56 张的候选列表里另有一个不与家具重合的候选。
2. **质检的误报**（标注本身可能是对的）：
   - 面积 < 0.2%：Subset A 的小物体（apple、banana、phone 等）本来就这么小；只带这一个标记的有 52 张，我看到的几张都是对的。
   - “on” 的几何检查：物体放在椅子座面上时，其质心低于椅子整体质心（椅背把质心抬高），会被判为不一致（on 30 张、under 15 张被标，其中一部分是这种情况，一部分是第 1 类真错误）。
   - Subset B 的可乐罐分数普遍只有 0.28～0.33，但我看到的框是准的。
3. **质检抓不到的错误**（通过的图里也可能有）：mask 只覆盖物体的一部分（例如盆栽只分割出叶子，没包括花盆）；桌子的 mask 可能把桌腿之间的空隙也算进去。这两种我是在联系表里看到了迹象，没有统计数量。

---

## 3. F2 与 F7

### 3.1 F2：GQA-spatial 的框（详见 `docs/parts/F2_gqa_boxes.md`）

- 下载：GQA `sceneGraphs.zip`，44.8 MB（低于 5 GB 阈值），sha256 `59f6a3f6…7c51989`。官方下载页没有许可正文，只有一个 CC BY 4.0 徽标链接。
- 规则：物体名称恰好匹配到场景图里的一个物体才给框；不使用关系标签在多个候选里挑选。

| 子集 | 唯一匹配 | 比例 |
|---|---|---|
| one-object：target | 1,121 / 1,160 | 96.6% |
| two-object：target | 290 / 291 | 99.7% |
| two-object：reference | 287 / 291 | 98.6% |
| two-object：两者都有 | 286 / 291 | 98.3% |

- 匹配不上的 44 处：29 处名称对不上（同义词，如 “mobile phone” 对 `cell phone`），15 处同名物体有多个。样本全部保留并标记为无框。
- one-object 中 1,119 个能做左右/上下核对的唯一框，位置全部与 caption 所说的一侧一致；用关系标签也无法消除那 15 处多候选。
- 1 张图（498187）场景图里的宽高字段有误，但框已在图像文件的像素坐标系里；32 个框越界 1 像素，已裁剪。
- 产物：`manifests/gqa_spatial_boxes.json`；加载器 `gqa_spatial_one / two` 已带框；抽样图 `outputs/sanity/gqa_boxes.png`。

### 3.2 F7：SpatialRGPT-Bench（详见 `docs/parts/F7_spatialrgpt.md`）

- **使用限制：仅用于评测，不再分发；HF 卡片未声明许可。**
- 下载：固定 commit `01b38d2c`，1,415 个文件，2.91 GB。`train-*` 与 `val-*` parquet 逐字节相同（重复 0.92 GB）。加载器只用 JSON + `images/`。
- 前/后关系样本：**110 条**（`front_predicate` 55 + `behind_predicate` 55）；按答案换算成真实关系为 front 63 / behind 47。其中 34 条的标准答案只有 “No.”，真实关系是按“否定即相反”推出并标记的。
- 带框 1,406 / 1,406（100%），带 mask 1,406 / 1,406（100%），数量与题目引用的区域数一致。数据自带的框不紧贴 mask（两者 IoU 中位数 0.80；2,546 个里 77 个 < 0.5）。
- 两个区域类别名相同的：深度样本里 14 条，全集 123 条。
- 改写规则只做了设计；20 条改写前后对照在 `docs/parts/F7_spatialrgpt.md` 的“20 条预览”，**等你确认后才全量改写**。
- “类别名第 k 个 ↔ 第 k 个区域”的对应关系数据集没有说明，只抽查了 12 条（都符合）。

---

## 4. F3：COCO-Pairs（详见 `docs/parts/F3_coco_pairs.md`，含关系 × 间距交叉表）

- 候选物体 4,966 个（36,781 个标注 → 非 crowd 36,335 → 类别在图中唯一 8,548 → 面积 1%～40% 5,050 → 短边 ≥ 300 后 4,966）。
- 有 ≥ 2 个候选的图像 1,274 张；无向对 2,562，有向对 **5,124**（两个方向都保留并互相标记）；覆盖全部 80 个类别。
- 没有过滤困难样本，没有下采样。

| 关系 | left | right | above | below |
|---|---|---|---|---|
| 有向对 | 1,506 | 1,506 | 1,056 | 1,056 |

| 归一化间距 | <0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–3 | 3–5 | ≥5 |
|---|---|---|---|---|---|---|---|
| 有向对 | 778 | 1,664 | 1,134 | 592 | 552 | 316 | 88 |

| 图中其它实例数 | 0 | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---|---|---|---|---|---|
| 有向对 | 632 | 1,200 | 1,258 | 1,154 | 724 | 156 |

- 关系轴上投影重叠（关系本身有歧义）2,908；框 IoU > 0 的 2,814；mask IoU > 0 的 2,070。
- 划分按图像：dev 635 张 / 2,542 对（743 / 743 / 528 / 528），test 639 张 / 2,582 对（763 / 763 / 528 / 528）。
- 与 COCO-spatial 的重叠：1,274 张里 834 张也在 COCO-spatial one-object 里（其中 284 张在 two-object 里），440 张两者都不在。**两套划分是各自独立做的，同一张图可能在一边是 dev、另一边是 test**（见第 8 节）。
- 口径说明：类别唯一性把 crowd 标注也计入；面积用 COCO 的 `area` 字段（与解码 mask 的面积平均相差 0.12%，4 个候选若用解码面积会落到范围外，两个值都存了）；归一化间距的分子是质心欧氏距离，另存了只沿关系轴的版本。

---

## 5. F4：参数化合成场景（详见 `docs/parts/F4_synth.md`）

- 生成器 `src/synth/`，配置 `configs/synth/trial_v0.yaml`；每个场景的随机种子由全局种子 + 场景序号导出，可单独重生成。重新生成 20 个场景（60 张图），PNG 与元数据逐字节相同。
- **抠图库目前只含 COCO**：1,333 个抠图、63 个类别，dev 656 / test 677，两侧不共用。构建时 F1 的标注还没出来，所以 What's Up 的部分没有纳入（代码路径已实现并用合成记录做了单元测试）。留出池的 1,000 张图已排除。有 659 个抠图来自 COCO-spatial 的图像、472 个来自 COCO-Pairs 的图像（已记录，未排除）。
- 试生成：500 个基础场景 + 各 2 张孪生图 = 1,500 张，只用 dev 抠图。

| 因素 | 每档基础场景数 |
|---|---|
| 间距（11 档） | 8–48 px 各 46，64–384 px 各 45 |
| 尺寸 8% / 12% / 18% | 167 / 167 / 166 |
| 关系轴 | 水平 252 / 垂直 248 |
| 干扰物 0 / 2 / 4 / 6 | 各 125 |
| 背景（4 种） | 各 125 |

- 间距 × 尺寸 × 轴 × 干扰物数的 264 个组合全部覆盖。发生遮挡的场景比例：8–32 px 为 100%，48 px 89%，64 px 76%，96 px 38%，128 px 9%，192 px 及以上 0%。实际间距与设定值相差至多 0.5 px（整数像素放置的限制）。
- 联系表（每行：原图 | 位置互换孪生 | 身份互换孪生）：`/nfs-stor/xiwei.liu/interpretability/outputs/synth_review/trial_v0/sheet_000.jpg` … `sheet_007.jpg`；前两张覆盖全部 11 个间距档。子代理只看过前两张，我没有逐张看。
- **全量估计**（由试生成实测外推，8 核）：5,000 个场景 = 15,000 张图，约 **2.19 GB** PNG + 74 MB 元数据，墙钟约 **4 分钟**（实测 0.36 核·秒 / 场景）。
- 已知局限：物体悬浮、比例和光照不真实；抠图边缘带 COCO 多边形标注的毛边，抠图质量没有人工检查；面积阈值（96×96）把 fork、knife、spoon 等小物体类别筛掉了；大间距时绝对位置与标签相关，只有位置互换孪生能抵消。

---

## 6. F5：COCO 留出池

5,000 张里，COCO-spatial one-object 占 1,451、two-object 占 295（是前者的子集）、COCO-Pairs 占 1,274，并集 1,891；剩 3,109 张，固定种子取 **1,000 张**，清单 `splits/coco_holdout.json`。没有挪用任何评测图像。

---

## 7. F6：两项数据分析

### 7.1 What's Up B 的深度线索（`manifests/whatsup_b_depth_cues.json`）

只用两个特征预测 front / behind：dy =（target 质心 y − reference 质心 y）/ 图像高度；ln(target 面积 / reference 面积)。标准化 + 逻辑回归。

| 样本 | 特征 | 分层 5 折 × 20 次 | 按物体组合分组 5 折 |
|---|---|---|---|
| 全部 204 张（front 102 / behind 102，102 个物体组合） | 仅 dy | 0.995 | 0.995 |
| | 仅面积比 | 0.856 | 0.859 |
| | dy + 面积比 | 0.982 | 0.975 |
| 仅自动质检通过的 185 张（front 95 / behind 90） | 仅 dy | 1.000 | 1.000 |
| | 仅面积比 | 0.887 | 0.886 |
| | dy + 面积比 | 1.000 | 1.000 |

- 多数类基线 0.50（全部）/ 0.51（通过）。
- 最简单的规则“target 质心在图像中更低就判 front”：全部 204 张 98.5%，质检通过的 185 张 100%。
- front 样本的 dy 均值 +0.27，behind 为 −0.30（图像高度的比例）。
- **含义：在 What's Up B 上，front / behind 几乎完全可以由垂直位置决定，答对不能说明用了深度信息。** 这建立在自动标注上；那 3 张不符合规则的图是否是标注错误，要等人工审核。

### 7.2 物体覆盖的视觉 token 数

<!-- COVERAGE_TABLE -->

---

## 8. 需要你决定或动手的事项（按阻塞程度排序）

1. **审核 F1 的自动标注**（阻塞：LaMa 抹除、坐标约定标定、所有需要真值框的实验）。至少 211 张待审，在联系表第 1–11 页。
2. **是否做标注 v2**：对 target 框与 reference 框重合（IoU > 0.8）的 68 张，改用下一个候选框并重新分割；其中 56 张有可用候选。要做的话最好在你开始填 `review.csv` 之前定，因为联系表和审核表要重新生成。你没说之前我不动 v1。
3. **COCO-Pairs 与 COCO-spatial 的划分不一致**：834 张共用图像可能一边是 dev 一边是 test。选项：(a) 让 COCO-Pairs 的划分服从 COCO-spatial；(b) 分析时用已记录的标记排除共用图像（剩 440 张图）；(c) 保持现状并限定两个数据集不混用于同一个选择—报告流程。
4. **What's Up B 的 front / behind 可由垂直位置决定**（7.1）：这影响“深度关系”实验的设计，是否需要另找或另造不带这一捷径的数据，请你定。
5. **SpatialRGPT 改写规则**：请看 20 条对照后确认；另外三点要定：是否接受 34 条 “No.” 样本的否定推断；同名类别的 14 条（全集 123 条）跳过还是用“左边的 / 右边的”区分；定位真值用框还是 mask。
6. **合成场景**：看联系表后确认生成器；另要定：现在把 What's Up 抠图加进去重建抠图库（会改变划分、要重做试生成），还是等人工审核后再加；来自 COCO-spatial / COCO-Pairs 图像的抠图是否要排除；自然背景用哪个来源。
7. **F1 质检阈值**：面积下限 0.2% 对 Subset A 的小物体误报多（52 张只带这一个标记）；“on” 的几何检查对椅子会误报。是否调整这两条规则由你定，我没有改。
8. **GQA 的 29 处同义词**：是否做一张手工同义词表来补框。
9. **四个 27B 以上大模型的冒烟**：批处理作业（435213）在 16:24 被取消，不是我取消的。是否重新提交？目前记为“大模型冒烟未做”。
10. **Qwen3-VL-8B 的 3D 框输出**：提示词留到 dev 上定（你已说明），这里只提醒它在框预测部分目前有约四分之一的输出不可用。
11. **GPU 配额**：仍只有 debug QoS 能较快拿到卡（单作业 3 小时、同时 2 个作业）。

---

## 9. 本阶段新增的磁盘占用

| 位置 | 占用 |
|---|---|
| `models/sam2.1-hiera-large`、`models/big-lama` | 1.80 GB + 0.38 GB |
| `envs/tools`（第三个环境，锁文件 `requirements/tools.lock`） | 约 7.8 G |
| `data/depth/spatialrgpt_bench` | 2.91 GB |
| `data/gqa` + 压缩包 | 约 0.6 G |
| `data/coco_pairs`、`data/whatsup_annotations` | 约 9 MB |
| `data/synth`（抠图库 + 试生成） | 见 `docs/parts/F4_synth.md` |
| `outputs/annotation_review` | 63 MB |
