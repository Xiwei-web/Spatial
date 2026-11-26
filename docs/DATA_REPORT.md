# 数据构建报告（DATA_REPORT，阶段 F）

日期：2026-11-25。范围：冒烟测试收尾（第 1 节）和数据构建 F1–F7。**本阶段没有在任何数据上计算模型正确率，没有调整任何提示词，没有生成合成集全量，没有对 COCO-Pairs 下采样，没有生成 LaMa 抹除图。**

**What's Up 的框和 mask 已定稿为 v3，状态是“自动标注，经模型辅助预审和部分人工复核”，不是全部人工验证。** 第 7 节两项分析基于 v3。第 8 节只列还需要你决定的事项。

各部分的详细文档在 `docs/parts/`：`F2_gqa_boxes.md`、`F3_coco_pairs.md`、`F4_synth.md`、`F5_coco_holdout.md`、`F7_spatialrgpt.md`、`rule_a_regression.md`（选框规则 (a) 修正的回归）、`bbox_impl_diff.md`（两种注意力实现的框差异与 float32 对照）。

你对上一份报告第 6 节的决定已落实：未下载 Gemma-3；只下载了 SpatialRGPT-Bench 完整数据；没有使用任何 ImageNet 副本；下载了 `sam2.1-hiera-large`（1.80 GB）和 `big-lama`（0.38 GB），GroundingDINO 只读使用你的本地权重和代码；LLaVA-1.5 改为先补成正方形；Cambrian-S 已从候选中去掉；合成场景抠图库只用 COCO；没有删除任何文件。

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
- 修复后与全 sdpa 的差异（RoboBrain）：末位置 logits 最大差 0.30～0.63（旧设置为 6.9），40 条输出里 22～23 条与全 sdpa 逐字相同（bf16 下两种实现本来就不是逐位相同的）。
- 我没有查明 bf16 下 eager 为什么会坏、sdpa 为什么不会（两者算式相同，只是缩放的位置不同）；上表是经验定位，不是机理解释。
- 这个补丁只在注意力模块所在的源码文件提供 `eager_attention_forward` 时生效；每个模型是否生效记录在冒烟 `summary.json` 的 `eager_fp32_scoring` 字段里。
- 证据文件：`outputs/diagnostics/attn_dtype_RoboBrain2.0-7B.json`、`eager_localize_RoboBrain2.0-7B.json`。只在 RoboBrain 上做了定位；其它模型是通过“修复后重跑冒烟”间接验证的（见 1.2）。

**除四个 27B 以上模型（未做）和 Cambrian-S（已去掉）外，所有模型都在修复后的设置下重跑了冒烟**，补丁在这 18 个模型上全部生效（`eager_fp32_scoring.applied = true`）。下表是重跑结果。

### 1.2 冒烟最终表

“通过”只表示管线跑通、状态拿得到，不表示框或答案正确；没有计算正确率。第 2、3 项旁边的数字是解析成功的次数。组别是你给的暂定分组。

<!-- SMOKE_TABLE_BEGIN -->
| 模型 | 组 | 环境 | 1 加载 | 2 框（解析出四个数的次数） | 3 关系（解析出选项的次数） | 4 隐状态 | 5 注意力/head | 6 视觉token映射 | 7 干预 | 8 确定性 | 显存 GB | 秒/样本 | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 1 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.491 | `f7bac9e` |
| Qwen3-VL-8B-Instruct | 1 | tf5 | 通过 | **失败** 29/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 16.33 / 22.72 | 1.866 | `f7bac9e` |
| RoboBrain2.0-7B | 1 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.139 | `f7bac9e` |
| VST-7B-RL | 1 | tf5 | 通过 | **失败** 19/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.489 | `f7bac9e` |
| Qwen3.5-2B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 4.12 / 9.08 | 1.439 | `579f5aa` |
| Qwen3.5-4B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 8.46 / 13.42 | 1.97 | `4a26260` |
| Qwen3.5-9B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 17.53 / 22.49 | 1.924 | `4a26260` |
| Qwen3.6-27B | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| Qwen3.8-27B | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| Qwen3.6-35B-A3B | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| GLM-4.6V-Flash | 1 | tf5 | 通过 | 通过 40/40 | **失败** 13/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 19.17 / 29.21 | 1.526 | `6842b9a` |
| gemma-4-31B-it | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| LLaVA-OneVision-7B | 3 | tf5 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | **失败** | 通过 | 14.96 / 38.59 | 1.164 | `4a26260` |
| LLaVA-1.5-7B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 13.16 / 16.25 | 0.567 | `4a26260` |
| LLaVA-1.5-13B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 24.87 / 29.32 | 0.76 | `4a26260` |
| InternVL3-8B | 1 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.712 | `f7bac9e` |
| InternVL3-2B | 2 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 3.89 / 4.97 | 0.618 | `4a26260` |
| InternVL3_5-8B | 1 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.89 / 16.97 | 0.953 | `f7bac9e` |
| SenseNova-SI-1.1-InternVL3-2B | 2 | tf4 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 3.89 / 4.97 | 0.786 | `4a26260` |
| SenseNova-SI-1.2-InternVL3-8B | 1 | tf4 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.789 | `f7bac9e` |
| SenseNova-SI-1.3-InternVL3-8B | 1 | tf4 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.804 | `f7bac9e` |
| MiniCPM-V-4_5 | 3 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | **失败** | 通过 | 通过 | 17.37 / 17.83 | 0.75 | `4a26260` |
| Cambrian-S-7B | excluded | — | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |

**失败项与说明**

- **Qwen3-VL-8B-Instruct** / 2_bbox：官方提示词=True，约定=norm1000_xyxy，解析出四个数 29/40，还原 29/40；
- **VST-7B-RL** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 19/40，还原 0/40；
- **Qwen3.5-2B** / 非标准层：18 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.5-4B** / 非标准层：24 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.5-9B** / 非标准层：24 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.6-27B**：未运行。未运行
- **Qwen3.8-27B**：未运行。未运行
- **Qwen3.6-35B-A3B**：未运行。未运行
- **GLM-4.6V-Flash** / 3_relation：解析出选项 13/20，四个选项 logit 齐全 20/20
- **gemma-4-31B-it**：未运行。未运行
- **LLaVA-OneVision-7B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **LLaVA-OneVision-7B** / 7_interventions：{}
- **LLaVA-OneVision-7B** / 7c_attention_knockout：OutOfMemoryError: CUDA out of memory. Tried to allocate 486.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 170.94 MiB is free. Including non-PyTorch memory, this process has 39.32 GiB memory in use. Of the allocated memory 38.56 GiB is allocated by PyTorch, and 272.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://py
- **SenseNova-SI-1.1-InternVL3-2B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **SenseNova-SI-1.2-InternVL3-8B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **SenseNova-SI-1.3-InternVL3-8B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **MiniCPM-V-4_5** / 6_visual_token_mapping：视觉 token 没有固定的图像区域对应
- **Cambrian-S-7B**：未运行。用户决定不适配，已从候选中去掉（2026-11-25）；model_type=cambrian_qwen 不在 transformers 中
<!-- SMOKE_TABLE_END -->

说明：
- 四个 27B 以上模型（Qwen3.6-27B、Qwen3.8-27B、Qwen3.6-35B-A3B、gemma-4-31B-it）：未做，让位给数据作业（用户取消 435213）。
- 第二组与 LLaVA-OneVision、MiniCPM 于 2026-11-25 19:39–19:53 在修复后的设置下重跑。前一次提交（435556）在 gpu-05 上 9 个模型全部 rc=1，之后避开该节点重新提交。
- LLaVA-OneVision-7B 的第 7 项失败只在注意力屏蔽一步，原因是单卡 40 GB 显存不足（输入长边已限到 768）；没有在 2 张卡上重试。
- GLM-4.6V-Flash：第一轮冒烟没有被限制分辨率（它的图像处理器不接受 `max_pixels`，代码退回默认设置，1,564 个视觉 token）。2026-11-25 按你的决定限制到与 Qwen2.5-VL 相同的像素预算并重跑（上表、1.4、1.5 都是重跑结果）：1148×840、1,230 个视觉 token（30×41），输入像素与 Qwen2.5-VL 逐字节相同。处理器按 2 帧计像素，所以配置里 `longest_edge = 2007040`。旧输出留档在 `outputs/archive/glm_default_resolution_20261125/`。

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

<!-- MAPPING_TABLE_BEGIN -->
| 模型 | 网格 | 重合率（均值 / 最低） | 最大变化 token 落在与矩形相交格子内的比例 | 预测图与实测图相关系数 | 质心距离（token，均值 / 最大） | 变化最大的单个 token 在矩形内 | 翻转对照重合率（左右 / 上下 / 两者） | 矩形内外平均变化量之比 | 第 1 / 2 / 3 层后的重合率 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 30×41 | 0.59 / 0.50 | 0.72 | 0.60 | 3.4 / 4.4 | 5/5 | 0.20 / 0.09 / 0.02 | 3.7 | 0.60 / 0.60 / 0.59 |
| Qwen3-VL-8B-Instruct | 27×36 | 0.40 / 0.33 | 0.47 | 0.43 | 4.7 / 6.8 | 3/5 | 0.15 / 0.11 / 0.03 | 2.8 | 0.49 / 0.52 / 0.59 |
| RoboBrain2.0-7B | 30×41 | 0.61 / 0.56 | 0.76 | 0.63 | 3.2 / 4.0 | 5/5 | 0.20 / 0.11 / 0.01 | 4.1 | 0.62 / 0.62 / 0.61 |
| VST-7B-RL | 30×41 | 0.67 / 0.66 | 0.80 | 0.68 | 2.4 / 3.2 | 5/5 | 0.21 / 0.10 / 0.01 | 5.1 | 0.67 / 0.68 / 0.67 |
| GLM-4.6V-Flash | 30×41 | 0.42 / 0.40 | 0.54 | 0.50 | 3.5 / 4.8 | 5/5 | 0.17 / 0.13 / 0.04 | 3.4 | 0.42 / 0.41 / 0.42 |
| InternVL3-8B | 16×16 | 0.74 / 0.60 | 0.91 | 0.70 | 1.1 / 1.9 | 5/5 | 0.20 / 0.12 / 0.00 | 3.5 | 0.76 / 0.76 / 0.74 |
| InternVL3_5-8B | 16×16 | 0.80 / 0.73 | 0.94 | 0.78 | 0.9 / 1.3 | 5/5 | 0.20 / 0.13 / 0.00 | 4.5 | 0.80 / 0.80 / 0.80 |
| SenseNova-SI-1.2-InternVL3-8B | 16×16 | 0.76 / 0.67 | 0.92 | 0.70 | 1.1 / 1.9 | 5/5 | 0.20 / 0.12 / 0.00 | 3.6 | 0.75 / 0.75 / 0.76 |
| SenseNova-SI-1.3-InternVL3-8B | 16×16 | 0.76 / 0.67 | 0.91 | 0.70 | 1.1 / 1.9 | 5/5 | 0.20 / 0.12 / 0.00 | 3.6 | 0.77 / 0.77 / 0.76 |
<!-- MAPPING_TABLE_END -->

- 没有发现映射错位的证据：预测图翻转后重合率明显低于正确映射；变化最大的单个 token 在 45 张图里有 43 张落在矩形内。
- 这个验证不能证明映射精确到单个 token：重合率本身不高（Qwen3-VL 0.40、GLM 0.42），系统性偏移一两个 token 的可能性排除不了。每模型只有 5 张图、一种矩形大小。
- Qwen3-VL 的重合率在第 1–3 层后逐层升高，与 DeepStack 在第 0、1、2 层之后再次注入视觉特征相符。

对比图：`outputs/token_mapping_verify/<模型>/compare.png`（上：噪声矩形；中：映射预测；下：实测变化）。

### 1.5 sdpa 参照实现与修复后 eager 的对比（第一组 9 个模型，bf16）

同一份权重、60 条提示（40 条框、20 条关系），贪心生成；`scripts/check_default_vs_eager.py`。参照统一用 sdpa（InternVL 远程代码自己的默认是 flash-attn，没装时退回 eager）。切换的是语言模型的注意力实现。

<!-- IMPL_TABLE_BEGIN -->
| 模型 | 框提示逐 token 一致 | 关系提示逐 token 一致 | 四个选项 logit 最大差 | 选项 logit 平均差 | 四选项第一名一致 |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 23/40 | 20/20 | 0.375 | 0.131 | 20/20 |
| Qwen3-VL-8B-Instruct | 20/40 | 20/20 | 0.750 | 0.369 | 20/20 |
| RoboBrain2.0-7B | 23/40 | 20/20 | 0.188 | 0.075 | 20/20 |
| VST-7B-RL | 15/40 | 20/20 | 0.500 | 0.197 | 20/20 |
| GLM-4.6V-Flash | 29/40 | 18/20 | 0.281 | 0.105 | 20/20 |
| InternVL3-8B | 34/40 | 20/20 | 0.375 | 0.200 | 20/20 |
| InternVL3_5-8B | 24/40 | 20/20 | 0.750 | 0.263 | 20/20 |
| SenseNova-SI-1.2-InternVL3-8B | 26/40 | 20/20 | 0.250 | 0.119 | 20/20 |
| SenseNova-SI-1.3-InternVL3-8B | 19/40 | 20/20 | 0.250 | 0.100 | 20/20 |

坐标差按各模型原始输出的数字计（单位是各自的输出约定）：

| 模型 | 框提示不一致条数 | 两边都能解析 | 坐标最大差：中位数 | 坐标最大差：最大值 | 相差 ≥ 50 的条数 |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 17 | 17 | 1 | 2 | 0 |
| Qwen3-VL-8B-Instruct | 20 | 13 | 1 | 10 | 0 |
| RoboBrain2.0-7B | 17 | 17 | 8 | 35 | 0 |
| VST-7B-RL | 25 | 11 | 3 | 94 | 1 |
| GLM-4.6V-Flash | 11 | 11 | 1 | 1 | 0 |
| InternVL3-8B | 6 | 6 | 3.5 | 226 | 2 |
| InternVL3_5-8B | 16 | 16 | 4 | 15 | 0 |
| SenseNova-SI-1.2-InternVL3-8B | 14 | 14 | 0.001 | 545.45 | 1 |
| SenseNova-SI-1.3-InternVL3-8B | 21 | 21 | 3 | 61 | 1 |
<!-- IMPL_TABLE_END -->

- 关系回答：180 条里 179 条逐 token 相同，选中的选项 180 条全部相同。
- 框预测：不是逐 token 相同的。相差 ≥ 50 的 5 条逐条列在 `docs/parts/bbox_impl_diff.md`。
- float32 下的对照、teacher-forced 读出（只做 InternVL3-8B）和显存测试见同一文件第 3–6 节。
- 由此定下的规则写在 `README.md`：干预组和对照基线必须用同一种注意力实现、同一种精度；每个模型另报一次默认实现下的无干预结果。

---

## 2. F1：What's Up Subset A / B 的标注（v3 定稿）

**状态：自动标注，经模型辅助预审和部分人工复核。** 任何文档引用这份标注都用这个说法（固定在 `configs/annotation/object_conventions.yaml`）。

### 2.1 做法与版本

- 自动标注：GroundingDINO（SwinB，你的本地权重，只读）用数据集标注解析出的物体名称给候选框；SAM 2.1（hiera-large）用框提示生成 mask。物体名称没有手改。可变形注意力用你仓库自带的纯 PyTorch 实现，没有在你的仓库或权重目录里写任何文件。

| 版本 | 内容 | 自动质检通过 / 待审 |
|---|---|---|
| v1 | 每个物体取最高分候选 | 609 / 211 |
| v2 | 按你的规则 (a)–(f)：target 与 reference 重合时换候选、放宽 Subset A 面积下限、改 on/under 几何检查、新增 mask–框一致性检查等 | 718 / 102 |
| **v3（定稿）** | v2 + 审核结论：改选 7 张、重新检测并经你确认 25 张（其中盆栽 8 张的 mask 为叶片框 mask ∪ 花盆框 mask）、剔除 2 张；reference 的 mask 减去 target 的 mask | — |

- v1、v2 保留不动。v3：`data/whatsup_annotations/auto_v3/annotations.jsonl`（sha256 记录在 `manifests/whatsup_auto_v3_summary.json`）；同目录的 `annotations_draft.jsonl`（初版）和 `annotations_pending_plants.jsonl`（你看盆栽叠加图时的版本）仅作留档。
- 重新生成：`annotate_whatsup.py` → `whatsup_qc.py` → `build_whatsup_v3.py` → `finalize_whatsup_v3.py --plants-confirmed`；人工输入在 `annotations/whatsup/`（`review_final_v2.csv`、`confirm_v3.csv`、`corrections.yaml`）。
- 下游统一用 `src/data/whatsup_annotations.py` 读取：只返回可用样本，未定稿或哈希不符会报错，取框时必须指明用检测框还是 mask 外接框。

### 2.2 审核是怎么做的

- v2 的 820 张全部经过模型辅助预审：41 张由你逐张确认，83 张按你定的口径规则由预审判定，**696 张只在缩略图上预审，未经你逐张检查**。
- v3 重新检测的 25 张：按模型预填的确认表、经你确认后执行；其中 order 12、30 与脚本建议不同。盆栽 8 张的 mask 由你看叠加图确认（定稿与你看过的版本逐像素相同）。

| v3 状态 | 张数 | 可用 |
|---|---|---|
| 沿用 v2（审核 ok） | 786 | 是 |
| 按审核表改选候选 | 7 | 是 |
| 按你的确认更新 | 25 | 是 |
| 剔除（pillow 两张） | 2 | 否 |

可用 **818** 张：Subset A 410、Subset B 408。逐张改动、原因和依据见 `manifests/whatsup_auto_v3_changelog.csv`。

### 2.3 标注口径

口径文件 `configs/annotation/object_conventions.yaml`：plant 包含花盆；photograph 包含白色衬纸；shoes 包含两只鞋；microphone 不含线缆；rose、flower 不含花茎；bread 不含面包袋的透明尾部；glasses、sunglasses 包含镜片、整副填实。
通用规则：**每个 mask 只包含该物体可见的像素；当 target 放在或挡在 reference 上时，reference 的 mask 不含被 target 覆盖的部分。**

**默认真值框 = 原始 mask（减去 target 之前）的外接框，存为 `box_gt_xyxy`**；检测框 `box_det_xyxy` 保留作对照。SpatialRGPT 和 COCO 也统一用 mask 外接框。两种框的 IoU（`manifests/whatsup_auto_v3_box_iou.json`、`manifests/truth_box_stats.json`）：

| 数据 | 物体数 | 最小 | 5% 分位 | 中位数 | IoU < 0.8 |
|---|---|---|---|---|---|
| What's Up A（检测框对 mask 外接框） | 820 | 0.812 | 0.891 | 0.969 | 0 |
| What's Up B（同上） | 816 | 0.754 | 0.924 | 0.966 | 1 |
| COCO-spatial one / two（COCO 自带框） | 2,247 / 880 | 0.940 / 0.975 | 0.987 / 0.986 | 0.994 | 0 |
| COCO-Pairs（不重复物体，COCO 自带框） | 3,101 | 0.880 | 0.982 | 0.994 | 0 |
| SpatialRGPT 深度样本（数据集自带框） | 192 | 0.329 | 0.556 | 0.795 | 99 |

- What's Up 里 IoU < 0.8 的唯一一个是 order 98 `cup_behind_flower` 的 flower（0.754）：mask 含花和花萼、不含花茎，检测框左侧多框了一段空地。
- SpatialRGPT 自带框明显比 mask 松，改用 mask 外接框后真值变化很大。
- COCO-Pairs 的几何量已按 mask 外接框重算（第 4 节）。

### 2.4 从 reference 的 mask 里减去 target

- 规则：有交集就减，所有关系类型；只改 reference 的 mask（及其面积、质心），target 的 mask 和两者的框都不变。原 mask 在 reference 的 `mask_reference_raw`。
- 实际改动 **109** 张，全部在 Subset A（on 83、under 26）。其中交集超过 target 面积 1% 的 50 张，其余 59 张只有少量边缘像素。Subset B 为 0。
- 减完后 reference 面积不足原来 50% 的：**0** 张。面积减少最多的 10 张：

| order | 样本 | 原面积 | 新面积 | 减少 | 剩余比例 |
|---|---|---|---|---|---|
| 16 | scarf_on_chair | 67,679 | 39,807 | 27,872 | 58.8% |
| 12 | pan_on_armchair | 172,969 | 166,854 | 6,115 | 96.5% |
| 266 | laptop_on_armchair | 170,000 | 165,570 | 4,430 | 97.4% |
| 44 | headphones_on_armchair | 177,305 | 173,305 | 4,000 | 97.7% |
| 197 | cap_under_armchair | 181,983 | 178,198 | 3,785 | 97.9% |
| 28 | banjo_on_armchair | 166,989 | 163,860 | 3,129 | 98.1% |
| 14 | glass_under_armchair | 179,785 | 176,672 | 3,113 | 98.3% |
| 423 | water-bottle_under_armchair | 179,638 | 176,820 | 2,818 | 98.4% |
| 161 | bottle_on_armchair | 173,074 | 170,696 | 2,378 | 98.6% |
| 271 | laptop_under_chair | 67,329 | 65,052 | 2,277 | 96.6% |

- 有 4 张减完后 mask 的外接框会变；真值框按规定取减之前的外接框，所以这 4 张的框与减后的 mask 不严格贴合。全部 109 张的清单：`manifests/whatsup_auto_v3_subtraction.csv`。

### 2.5 质检

- 新增并默认执行的一条：target 与 reference 的 mask 交集占 target 面积超过 1% 就标记（`target_covered_by_reference`）。减之前标记 50 张，减之后 0 张。原来的重叠检查用 IoU > 0.5，target 很小时抓不到。
- 定稿后剩余的标记（不影响可用性，仅供参考）：检测分数 < 0.35 的 74 张，多个分数接近的候选 45 张，target 无可换候选 3 张（order 6、7、8，是 v2 选框时留下的记录，审核已改选），几何不一致 2 张（已剔除的 pillow），mask 与检测框不一致 1 张。
- 几何顺序（left/right/on/under）与标签不一致的只有两张已剔除的 pillow（`manifests/whatsup_auto_v3_geometry.csv`）。front/behind 没有对应的检查。

### 2.6 剔除的两张与已知不足

- `pillow_left_of_chair`、`pillow_right_of_chair`：原始数据左右与标签相反（不是下载、解压或加载造成的）。按你的决定剔除、不改标签；更正表 `annotations/whatsup/corrections.yaml`，由 `load_split` 过滤，划分文件不变。两张都在 test，Subset A 因此 dev 204 / test 206。
- 眼镜类：`glasses_on_armchair` 的 target mask 较粗（镜片中间有空洞，镜腿间的沙发面被包进来一点），减完后镜片空洞处留在扶手椅的 mask 里。口径已定为“包含镜片，整副填实”，v3 没有因此重做：52 个眼镜类物体里，有封闭空洞的只有它明显（空洞占 mask 的 14.9%），其余都不超过 1.4%。镜腿与镜框之间的开口不算封闭空洞，没有统计。
- 盆栽的藤蔓末端含少量叶片间的地面，order 340 有一片叶子与主体断开（你已确认不用修）。
- 通过缩略图预审的 696 张里可能仍有未发现的错误。

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

- 匹配不上的 44 处：29 处名称对不上（同义词，如 “mobile phone” 对 `cell phone`；按你的决定不补），15 处同名物体有多个。样本全部保留并标记为无框。
- one-object 中 1,119 个能做左右/上下核对的唯一框，位置全部与 caption 所说的一侧一致；用关系标签也无法消除那 15 处多候选。
- 1 张图（498187）场景图里的宽高字段有误，但框已在图像文件的像素坐标系里；32 个框越界 1 像素，已裁剪。
- 产物：`manifests/gqa_spatial_boxes.json`；加载器 `gqa_spatial_one / two` 已带框；抽样图 `outputs/sanity/gqa_boxes.png`。

### 3.2 F7：SpatialRGPT-Bench（详见 `docs/parts/F7_spatialrgpt.md`）

- **使用限制：仅用于评测，不再分发；HF 卡片未声明许可。**
- 下载：固定 commit `01b38d2c`，1,415 个文件，2.91 GB。`train-*` 与 `val-*` parquet 逐字节相同（重复 0.92 GB）。加载器只用 JSON + `images/`。
- **用户决定后可用 96 条**：跳过两个区域同名的 14 条；靠 “No.” 推断的样本保留、标记并单独报告（跳过同名后 29 条，另有答案明确的 67 条）；真值框用 mask 外接框。
- **也有垂直位置捷径**（`manifests/spatialrgpt_depth_cues.json`）：“target 的 mask 下边缘更低 = front”在 96 条上 88.5%，答案明确的 67 条上 92.5%，靠推断的 29 条上 79.3%。只用质心垂直差的逻辑回归（按来源图像分组 5 折）：0.842。面积比几乎没有信息（0.564）。程度比 What's Up B 轻，但不能假定它没有捷径。
- 下载时的原始统计：前/后关系样本 **110 条**（`front_predicate` 55 + `behind_predicate` 55）；按答案换算成真实关系为 front 63 / behind 47。其中 34 条的标准答案只有 “No.”，真实关系是按“否定即相反”推出并标记的。
- 带框 1,406 / 1,406（100%），带 mask 1,406 / 1,406（100%），数量与题目引用的区域数一致。数据自带的框不紧贴 mask（两者 IoU 中位数 0.80；2,546 个里 77 个 < 0.5）。
- 两个区域类别名相同的：深度样本里 14 条，全集 123 条。
- 改写规则只做了设计；20 条改写前后对照在 `docs/parts/F7_spatialrgpt.md` 的“20 条预览”，**等你确认后才全量改写**。
- “类别名第 k 个 ↔ 第 k 个区域”的对应关系数据集没有说明，只抽查了 12 条（都符合）。

- **mask 有零星碎点，会撑大外接框**：深度样本的 192 个物体里 57 个有面积 < 1% 的碎块；去掉后 32 个的外接框会变，其中 19 个与原外接框 IoU < 0.95，6 个 < 0.8。目前的真值框按全部 mask 像素取，没有去碎点。与自带框 IoU 最低的 20 个物体的联系表在 `outputs/data_review/spatialrgpt_box/`（`lowest20_depth_used.jpg` 为实际会用的 192 个物体，`lowest20_all_regions.jpg` 为全集 2,546 个区域）。

### 3.3 CV-Bench 3D 深度子集（只查元数据，`manifests/cvbench_depth_cues.json`）

- 许可：HF 卡片写 apache-2.0；图像来自 Omni3D（Hypersim、SUN RGB-D、nuScenes），Omni3D 代码库标 CC BY-NC 4.0，各源数据集另有自己的许可，未逐一核实。
- Depth 题（哪个物体离相机更近）600 条、508 张来源图；两个物体的 2D 框都在元数据里（xyxy），没有 mask。图像上已画红 / 蓝框，干净原图要另从 Omni3D 源数据取。
- 捷径：规则“2D 框下边缘更低 = 更近”。

| 来源 | 条数 | 下边缘规则 | 框中心更低 | 框更大 | 与下边缘规则相反的条数 |
|---|---|---|---|---|---|
| 全部 | 600 | 85.3% | 80.8% | 59.5% | 88 |
| Hypersim | 200 | 80.0% | 79.5% | 44.0% | 40 |
| SUNRGBD | 200 | 82.0% | 86.5% | 67.0% | 36 |
| nuScenes | 200 | 94.0% | 76.5% | 67.5% | 12 |

- 与捷径相反的只有 **88** 条（< 150），按你的要求停在这里，没有构造平衡子集，也没有造深度数据。
- **结论（Xiwei 2026-11-26）**：CV-Bench 深度子集上，“更低判为更近”规则的准确率是 85.3%，与规则相反的只有 88 条（< 150），没有构造平衡子集。它的图像上已经画了红 / 蓝框，不适合需要物体 token 的实验。**前后关系降级为描述性分析**：What's Up B 的 front/behind 与垂直位置完全混淆，SpatialRGPT（88.5%）和 CV-Bench（85.3%）都有强的垂直位置捷径，三者都只作描述，不作“模型使用了深度信息”的证据。

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
- **划分（v2，统一）**：COCO-spatial 与 COCO-Pairs 的图像取并集 1,891 张，按图像一次性分配（种子 20261125），dev 945 / test 946；`splits/coco_images_v2.json`，脚本 `scripts/make_joint_coco_split.py`。按“属于哪些数据集 × 主轴 × 候选物体数”分层。

| 数据集 | dev | test |
|---|---|---|
| COCO-Pairs | 637 张 / 2,550 有向对（左 757 右 757 上 518 下 518） | 637 张 / 2,574 有向对（左 749 右 749 上 538 下 538） |
| COCO-spatial one | 1,122（{'bottom': 362, 'left': 275, 'right': 296, 'top': 189}） | 1,125（{'bottom': 382, 'left': 301, 'right': 268, 'top': 174}） |
| COCO-spatial two | 208（{'above': 39, 'below': 37, 'left': 69, 'right': 63}） | 232（{'above': 47, 'below': 38, 'left': 81, 'right': 66}） |
| 合成抠图 | 653 | 680 |

- 旧划分的问题不止两套不一致：旧 COCO-spatial 按样本划分，one-object 有 318 张图、two-object 有 52 张图同时出现在 dev 和 test；旧抠图划分有 110 张图跨侧。旧文件保留并标记 `deprecated`，读取会报错。测试 `tests/test_joint_coco_split.py` 保证同一张图在所有数据集里同侧。
- 按图像划分后 COCO-spatial 的关系类别不再严格对半（见上表）。
- 1,274 张 COCO-Pairs 图里 834 张也在 COCO-spatial one-object 里（其中 284 张在 two-object 里），440 张两者都不在。
- **几何量按 mask 外接框重算**（间距、框 IoU、包含比例、投影重叠；旧值在 `*_detbox` 字段，旧文件 `pairs_detbox_20261125.jsonl` 保留）。关系标签由 mask 质心决定，**0 对变化**；间距档变化 **14 个有向对**（7 个无向对，{'2-3 -> 3-5': 4, '0.5-1 -> 1-1.5': 6, '1-1.5 -> 1.5-2': 4}）；投影重叠与否变化 16 个有向对。归一化间距最大变化 0.070。逐条见 `manifests/coco_pairs_geometry_change.csv`。上面的间距表是重算前的，重算后：{'1-1.5': 1136, '0.5-1': 1658, '1.5-2': 596, '<0.5': 778, '3-5': 320, '>=5': 88, '2-3': 548}；投影重叠 2,892，框 IoU > 0 2,796。
- 口径说明：类别唯一性把 crowd 标注也计入；面积用 COCO 的 `area` 字段（与解码 mask 的面积平均相差 0.12%，4 个候选若用解码面积会落到范围外，两个值都存了）；归一化间距的分子是质心欧氏距离，另存了只沿关系轴的版本。
- **GQA-spatial 的划分**（`splits/gqa_images_v2.json`，`scripts/make_gqa_split.py`）：按图像，1,208 张，dev 600 / test 608。597 张有 coco_id，其中因 coco_id 在 COCO 统一划分里而被强制分侧的 **25 张**（21 张跟随 COCO-spatial / COCO-Pairs 并集，4 张跟随抠图来源图；dev 9、test 16）；其余随机分（无 coco_id 611 张，coco_id 不在统一划分里 572 张）。

| 子集 | dev 图像 / 样本 | test 图像 / 样本 |
|---|---|---|
| gqa_spatial_one | 519 / 581 | 524 / 579 |
| gqa_spatial_two | 127 / 139 | 131 / 152 |

- **跨数据集一致性**（`scripts/check_split_consistency.py`，已写成测试）：把图像换成统一键后共 4,376 张，1,269 张出现在多个数据集，**跨侧 0 张**；各数据集 dev ∪ test 都等于全部样本。
- **留出池有 1 张与 GQA-spatial 重叠**（COCO 126226 = VG 2406339，在 gqa_spatial_one 里）：建留出池时只排除了 COCO-spatial 和 COCO-Pairs。没有改动，等你定是否剔除这 1 张。

---

## 5. F4：参数化合成场景（详见 `docs/parts/F4_synth.md`）

- 生成器 `src/synth/`，配置 `configs/synth/trial_v0.yaml`；每个场景的随机种子由全局种子 + 场景序号导出，可单独重生成。重新生成 20 个场景（60 张图），PNG 与元数据逐字节相同。
- **抠图库只用 COCO**（你的决定）：1,333 个抠图、63 个类别；留出池的 1,000 张图已排除。划分继承所在图像（v2）：dev 653 / test 680，749 个来自评测图像并集，其余来源图像也按图像分配。`book` 类只有 test 有（10 个），其余 62 类两侧都有。
- **暂不扩量，等你看联系表。** trial_v0 用的是旧 dev 抠图，其中 326 个现在属于 test：它只能用来检查生成器，不能用于评测。之后的配置必须写 `cutout_split_name: synth_cutouts`，否则生成器拒绝运行。
- 试生成：500 个基础场景 + 各 2 张孪生图 = 1,500 张，只用 dev 抠图。

| 因素 | 每档基础场景数 |
|---|---|
| 间距（11 档） | 8–48 px 各 46，64–384 px 各 45 |
| 尺寸 8% / 12% / 18% | 167 / 167 / 166 |
| 关系轴 | 水平 252 / 垂直 248 |
| 干扰物 0 / 2 / 4 / 6 | 各 125 |
| 背景（4 种） | 各 125 |

- 间距 × 尺寸 × 轴 × 干扰物数的 264 个组合全部覆盖。发生遮挡的场景比例：8–32 px 为 100%，48 px 89%，64 px 76%，96 px 38%，128 px 9%，192 px 及以上 0%。实际间距与设定值相差至多 0.5 px（整数像素放置的限制）。
- trial_v0 的统计、联系表与局限见 `docs/parts/F4_synth.md`；它只作生成器检查用。

### 5.1 抠图库 library_v1（`scripts/build_synth_library_v1.py`，`manifests/synth_library_v1.json`）

- 成员与 v0 相同（1,333 个 COCO 抠图），从原图按原分辨率重裁（v0 入库时把长边缩到 256）；dev / test 继承所在图像的划分。
- 过滤：原始长边 ≥ 128 px；放大倍数 ≤ 1.5（按最大尺寸档 sqrt 面积 161 px 判，因此每个保留的抠图在所有尺寸都能用）；宽高比 ≤ 3:1；贴到灰底（128）正方形上用 CLIP 在 80 个 COCO 类别上零样本分类，正确类别第一且概率 ≥ 0.5。CLIP 用 `laion/CLIP-ViT-L-14-laion2B`（MIT，1.7 GB），没有用 OpenAI CLIP，因为 LLaVA-1.5 的视觉塔就是它。
- 单项不通过：长边 32、放大 170、宽高比 89、CLIP 187（一个抠图可同时不通过多项）。全部通过 962 个；两侧各 ≥ 3 个的类别 **53 / 63**，最终 **923** 个（dev 456 / test 467）。
- **排除的类别**：airplane（dev 1 / test 1）、backpack（dev 2 / test 1）、book（dev 0 / test 7）、bottle（dev 2 / test 4）、frisbee（dev 2 / test 2）、microwave（dev 2 / test 2）、mouse（dev 1 / test 2）、potted plant（dev 0 / test 5）、skateboard（dev 0 / test 2）、surfboard（dev 2 / test 1）。
- 每类过滤前后的数量在 `manifests/synth_library_v1.json` 的 `per_category`；逐个抠图的过滤记录在 `data/synth/cutouts_v1/filter_report.jsonl`。联系表（每类最多 6 个）：`outputs/synth_review/library_v1/sheet_00…04.jpg`。
- **CLIP 过滤不严**：80 选 1 的概率大多饱和在 1.00，我在联系表里看到残缺的抠图也能通过（被枝条遮住的熊、只剩一块床面的床、只露半个的苹果）；另有左右镜像的近重复（两张香蕉）。过滤主要去掉的是类别判错的。需要更干净的库时，要人工看联系表或另设标准。

### 5.2 trial_v1（`configs/synth/trial_v1.yaml`，`src/synth/generator_v1.py`，`manifests/synth_trial_v1.json`）

- 500 个基础场景 + 各 2 张孪生图 = 1,500 张，dev 250 / test 250（场景内全部抠图同侧）。重生成 20 个场景（60 张）逐字节相同；30 秒（8 核），0.33 GB。
- **边缘间距** = 两个 mask 在关系轴上投影区间的空隙（0 = 相接）。主网格 0、4、8、16、32、64、128、256 px，不重叠；另记录 mask 间最短距离、质心距离，以及按各模型视觉 token 边长换算的间距（896×896 画布：Qwen2.5-VL / RoboBrain / VST / GLM 28 px，Qwen3-VL / Qwen3.5 32 px，LLaVA-1.5 37.3 px，InternVL 系 56 px；前两组用处理器实算核对，后两组按模型结构推算）。重叠子集按下层遮挡比例分 <10%（下限 0.5%）、10–30%、30–50%。
- 尺寸按 sqrt(mask 面积)：72 / 108 / 161 px，实际误差 -0.50～0.51 px；放大倍数最大 1.50；所有物体离边最近 16 px。
- **身份互换**：质心偏差 0.41（0.03–0.70） px（中位数，最小–最大），sqrt 面积比 0.992～1.007。
- **身份互换孪生图的边缘间距不受控：Xiwei 2026-11-26 接受**，实际间距记录在每个场景的元数据里，并在 `manifests/synth_trial_v1.json` 的 `identity_swap_edge_gap_uncontrolled` 字段写明。合成集 v2 的规格另定，生成器没有改动。
- **位置互换**：主网格保持边缘间距不变（对调占位区间），形状不同所以质心不是恰好互换，质心偏移中位数 17 px、最大 113 px；重叠子集是质心互换（最大偏移 0.64 px）且保持同一遮挡档。

| 间距档 | 场景数 | dev / test | 质心沿轴距离中位数（px） | 身份互换孪生的实际间距（中位数，最小–最大） | 遮挡比例（base，中位数，最小–最大） |
|---|---|---|---|---|---|
| gap0 | 46 | 23 / 23 | 132.0 | 14（0–69） | 0 |
| gap4 | 46 | 23 / 23 | 146.3 | 11（0–85） | 0 |
| gap8 | 46 | 23 / 23 | 149.9 | 19（0–62） | 0 |
| gap16 | 46 | 23 / 23 | 150.2 | 20（0–96） | 0 |
| gap32 | 46 | 23 / 23 | 159.7 | 28（1–110） | 0 |
| gap64 | 45 | 22 / 23 | 188.6 | 56（21–154） | 0 |
| gap128 | 45 | 22 / 23 | 263.6 | 117（69–194） | 0 |
| gap256 | 45 | 23 / 22 | 395.7 | 265（183–328） | 0 |
| occ10 | 45 | 23 / 22 | 113.1 | — | 0.045（0.006–0.097） |
| occ30 | 45 | 22 / 23 | 83.3 | — | 0.220（0.117–0.293） |
| occ50 | 45 | 23 / 22 | 52.2 | — | 0.387（0.312–0.490） |

- **身份互换孪生的边缘间距没有控制**：质心和面积对齐后，换上的物体形状不同，实际间距会偏离设定值（见表，例如 gap0 中位数 14 px、最大 69 px）。质心对齐与边缘间距不能同时保证，按你的要求以质心为准，实际值记在元数据 `gap.projection_gap_px`。
- 每类在两侧实际用作主物体的抠图数在 manifest 的 `per_category`；53 个类别全部用到。
- 联系表：`outputs/synth_review/trial_v1/sheet_000…007.jpg`（64 个场景，每行 base | 位置互换 | 身份互换，前两张覆盖全部 11 档）。我看了前两张：间距、遮挡和标签与图一致，抠图质量问题同上。
- 没有扩量。
---

## 6. F5：COCO 留出池

5,000 张里，COCO-spatial one-object 占 1,451、two-object 占 295（是前者的子集）、COCO-Pairs 占 1,274，并集 1,891；剩 3,109 张，固定种子取 **1,000 张**，清单 `splits/coco_holdout.json`。没有挪用任何评测图像。

---

## 7. F6：两项数据分析

### 7.1 What's Up B 的深度线索（v3；`manifests/whatsup_b_depth_cues_v3.json`）

只用两个特征预测 front / behind：dy =（target 质心 y − reference 质心 y）/ 图像高度；ln(target 面积 / reference 面积)。标准化 + 逻辑回归。样本是全部 204 张（front 102 / behind 102，102 个物体组合，均可用），多数类基线 0.50。

| 特征 | 分层 5 折 × 20 次 | 按物体组合分组 5 折 | v1 标注上的旧结果（分层 / 分组） |
|---|---|---|---|
| 仅 dy | 1.000 | 1.000 | 0.995 / 0.995 |
| 仅面积比 | 0.898 | 0.888 | 0.856 / 0.859 |
| dy + 面积比 | 1.000 | 1.000 | 0.982 / 0.975 |

- 规则“target 质心在图像中更低就判 front”：**204 / 204**。v1 上有 3 张例外（98.5%），v3 中没有例外。
- front 样本的 dy 均值 +0.28，behind 为 −0.30。
- **含义：What's Up B 的 front / behind 与垂直位置完全混淆，答对不能说明用了深度信息。** 按你的决定保留这些样本；前后关系不只依赖 SpatialRGPT（它也有较强的垂直位置捷径，见 3.2）。
- 注意：v3 修正的框里有 5 张是 front 样本（order 6、7、8、21、682），修正依据是物体身份（哪个是杯子、哪个是罐子），不是这条规则。

### 7.2 物体覆盖的视觉 token 数

**最终版：基于 v3 定稿（818 张）**，reference 的 mask 已减去 target。v1 上的暂定结果保留在 `manifests/whatsup_token_coverage_v1.json`。GLM 处理器按 2 帧计像素，表中的 `max_pixels` 指单图像素预算；另列限制分辨率之前所用的默认分辨率（`old_default_resolution`，现已不用）。

<!-- COVERAGE_TABLE_BEGIN -->
标注：auto_v3 定稿（自动标注，经模型辅助预审和部分人工复核）；下表用全部 818 张（A / B 分开的数字在 JSON 里）。格式：中位数（5%–95% 分位）。

| 模型 | 设置 | 网格 | 视觉 token 数 | target 覆盖（有重叠） | target 覆盖（≥50% 面积） | reference 覆盖（有重叠） | reference 覆盖（≥50% 面积） | 同时与两个物体重叠 | 两质心距离（token） | target 没有任何 ≥50% token 的比例 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | max_pixels/2 | 21×29 | 609 | 19 (4–68) | 9 (0–50) | 60 (9–117) | 31 (2–88) | 0 (0–8) | 7.4 (0.7–11.1) | 9.9% |
| Qwen2.5-VL-7B-Instruct | smoke(max_pixels=1003520) | 30×41 | 1230 | 33 (6–125) | 20 (1–101) | 111.5 (14–210) | 68 (5–179) | 0 (0–11) | 10.5 (0.9–15.7) | 2.3% |
| Qwen2.5-VL-7B-Instruct | max_pixels*2 | 34×46 | 1564 | 39 (6–156) | 26 (2–127) | 139 (16–266) | 81 (7–225) | 0 (0–12) | 11.8 (1.1–17.6) | 0.4% |
| Qwen3-VL-8B-Instruct | max_pixels/2 | 19×25 | 475 | 15 (3–55) | 8 (0–38) | 51 (8–93) | 21 (1–67) | 0 (0–7) | 6.6 (0.6–9.6) | 14.4% |
| Qwen3-VL-8B-Instruct | smoke(max_pixels=1003520) | 27×36 | 972 | 27 (5–102) | 16 (1–80) | 95 (11.9–171) | 51 (4–140) | 0 (0–10) | 9.4 (0.8–13.8) | 2.6% |
| Qwen3-VL-8B-Instruct | max_pixels*2 | 30×40 | 1200 | 31 (6–122) | 19 (1–97.1) | 111.5 (13–207) | 58 (5–173) | 0 (0–11) | 10.4 (0.9–15.3) | 2.0% |
| RoboBrain2.0-7B | max_pixels/2 | 21×29 | 609 | 19 (4–68) | 9 (0–50) | 60 (9–117) | 31 (2–88) | 0 (0–8) | 7.4 (0.7–11.1) | 9.9% |
| RoboBrain2.0-7B | smoke(max_pixels=1003520) | 30×41 | 1230 | 33 (6–125) | 20 (1–101) | 111.5 (14–210) | 68 (5–179) | 0 (0–11) | 10.5 (0.9–15.7) | 2.3% |
| RoboBrain2.0-7B | max_pixels*2 | 34×46 | 1564 | 39 (6–156) | 26 (2–127) | 139 (16–266) | 81 (7–225) | 0 (0–12) | 11.8 (1.1–17.6) | 0.4% |
| VST-7B-RL | max_pixels/2 | 21×29 | 609 | 19 (4–68) | 9 (0–50) | 60 (9–117) | 31 (2–88) | 0 (0–8) | 7.4 (0.7–11.1) | 9.9% |
| VST-7B-RL | smoke(max_pixels=1003520) | 30×41 | 1230 | 33 (6–125) | 20 (1–101) | 111.5 (14–210) | 68 (5–179) | 0 (0–11) | 10.5 (0.9–15.7) | 2.3% |
| VST-7B-RL | max_pixels*2 | 34×46 | 1564 | 39 (6–156) | 26 (2–127) | 139 (16–266) | 81 (7–225) | 0 (0–12) | 11.8 (1.1–17.6) | 0.4% |
| GLM-4.6V-Flash | max_pixels/2 | 21×29 | 609 | 19 (4–68) | 9 (0–50) | 60 (9–117) | 31 (2–88) | 0 (0–8) | 7.4 (0.7–11.1) | 9.9% |
| GLM-4.6V-Flash | smoke(max_pixels=1003520) | 30×41 | 1230 | 33 (6–125) | 20 (1–101) | 111.5 (14–210) | 68 (5–179) | 0 (0–11) | 10.5 (0.9–15.7) | 2.3% |
| GLM-4.6V-Flash | max_pixels*2 | 34×46 | 1564 | 39 (6–156) | 26 (2–127) | 139 (16–266) | 81 (7–225) | 0 (0–12) | 11.8 (1.1–17.6) | 0.4% |
| GLM-4.6V-Flash | old_default_resolution | 34×46 | 1564 | 39 (6–156) | 26 (2–127) | 139 (16–266) | 81 (7–225) | 0 (0–12) | 11.8 (1.1–17.6) | 0.4% |
| InternVL3-8B | fixed_448_single_view | 16×16 | 256 | 10 (3–33) | 4 (0–20) | 32 (5–59) | 9 (0–36) | 0 (0–6) | 5 (0.5–6.3) | 26.7% |
| InternVL3_5-8B | fixed_448_single_view | 16×16 | 256 | 10 (3–33) | 4 (0–20) | 32 (5–59) | 9 (0–36) | 0 (0–6) | 5 (0.5–6.3) | 26.7% |
| SenseNova-SI-1.2-InternVL3-8B | fixed_448_single_view | 16×16 | 256 | 10 (3–33) | 4 (0–20) | 32 (5–59) | 9 (0–36) | 0 (0–6) | 5 (0.5–6.3) | 26.7% |
| SenseNova-SI-1.3-InternVL3-8B | fixed_448_single_view | 16×16 | 256 | 10 (3–33) | 4 (0–20) | 32 (5–59) | 9 (0–36) | 0 (0–6) | 5 (0.5–6.3) | 26.7% |
<!-- COVERAGE_TABLE_END -->

A、B 差别很大，上表合在一起会掩盖它（冒烟设置下；“≥50%”= 格子面积至少一半被 target 的 mask 覆盖）：

| 模型 | 子集 | target 覆盖（≥50%）中位数 | target 覆盖（有重叠）中位数 | target 没有任何 ≥50% token 的图 | target 的每个重叠 token 同时也与 reference 重叠的图 |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B（RoboBrain、VST、GLM-4.6V 相同，30×41） | A | 4 | 11 | 4.6% | 10.7% |
| | B | 49 | 67 | 0% | 0% |
| Qwen3-VL-8B（27×36） | A | 3 | 10 | 5.1% | 11.2% |
| | B | 38 | 55 | 0% | 0% |
| InternVL3 / 3.5 / SenseNova（16×16） | A | **0** | 5 | **53.2%** | **20.5%** |
| | B | 10 | 19 | 0% | 0% |

- **InternVL 系 + Subset A**：超过一半的图里，没有一个视觉 token 以 target 为主；约五分之一的图里，target 的 token 全部与 reference 共用。按 token 替换或屏蔽 target 的实验，在这个组合上必然同时动到背景或 reference。
- Subset B 没有这个问题。
- 与 v1 暂定版相比，target 的覆盖数变小（例如 30×41 网格“≥50%”中位数 28 → 20），推测主要是因为 v1 有一批 target 框在家具上（没有单独验证）。

---

## 8. 需要你决定或动手的事项（按阻塞程度排序）

1. **深度检验的数据**（已定，Xiwei 2026-11-26：前后关系降级为描述性分析，见 3.3）：SpatialRGPT（88.5%）、CV-Bench 3D（85.3%）都有垂直位置捷径，What's Up B 完全混淆。CV-Bench 与捷径相反的只有 88 条（< 150），没有构造子集。需要你定数据来源。
2. **框预测精度**：teacher-forced 对比已完成（`docs/parts/bbox_impl_diff.md` 第 4–6 节，作业 435836）。InternVL3-8B：fp32 下两种实现期望坐标最大差 0.0064（0–1000），bf16 下均值 0.45、最大 5.0；bf16 对 fp32 均值约 0.8、最大约 14。Qwen2.5-VL-7B 也已测（作业 436204，第 7 节）：fp32 下两种实现期望坐标最大差 3.9e-4 像素、自由生成 40/40 框逐 token 相同；bf16 对 fp32 框中心差的 95% 分位 0.012 个视觉 token（n=80）。按 planner 2026-11-26 的固定标准：Qwen2.5-VL-7B 通过（边缘），InternVL3-8B 不通过（cx 偏移约 0.004 个 token，区间不含 0）。其它模型没有测 fp32 的实现一致性。Qwen2.5-VL-7B 的 float32 + eager 单卡 40 GB 能跑但没有余量。请你定。
3. **SpatialRGPT 改写规则**（已定，Xiwei 2026-11-26）：只改写 96 条深度样本，同名物体的配对跳过，按 F7 的规则全量改写，原文保留。结果：96 条状态全部为 ok；25 条含复数类名，按规则改了动词；30 条答案含代词，没有改。含原文的文件在 `<data>/depth/spatialrgpt_derived/depth_rewritten_v1.jsonl`，因为只允许评测、不得再分发，所以不进仓库；不含原文的清单在 `manifests/spatialrgpt_depth_rewrite_v1.json`。
4. **合成场景**：请看 `outputs/synth_review/trial_v1/` 与 `library_v1/`；要定：抠图库是否需要比 CLIP 更严的筛选；身份互换孪生的边缘间距不受控是否可接受（5.2）；自然背景用哪个来源。
5. **留出池里有 1 张与 GQA-spatial 重叠的图**（已定，Xiwei 2026-11-26：剔除，剩 999 张，不补；新文件 `splits/coco_holdout_v2.json`，旧文件保留，标为作废，读取时报错；一致性检查里评测数据与留出池的重叠变为 0）（COCO 126226）：是否从留出池剔除（剩 999 张）。
6. **SpatialRGPT 真值框是否去碎点**（已定，Xiwei 2026-11-26）：只在计算框时去掉面积 < 1% 的碎块（8 连通的连通块，按该 mask 总面积算；mask 本身不改）。96 条深度样本的 192 个物体里，57 个有碎块，**32 个的框随之改变**（其中 19 个与原框 IoU < 0.95，6 个 < 0.8）。另有 **6 个物体从框相关的指标里排除**，但保留在关系问答里，原因写在 `extra["box_metric_exclude_*"]`：desk（gTR3O1nDxz）、bowl（iwxK9EjtZu），以及含骑车人的自行车（同一骑车人出现在 2 帧，各 2 道题，共 4 个区域）。代码在 `src/data/spatialrgpt.py`。
7. **没有官方框约定的模型**：下一阶段用带真值框的 dev 子集标定；Qwen3-VL-8B 的 3D 框输出也留到那时。
   - 2026-11-26 已在 dev 上标定（`reports/round_3.md`，清单 `splits/bbox_calib_v1.json`，结果 `manifests/bbox_calib_*.json`）：14 个模型选出约定，Qwen3-VL-8B、RoboBrain2.0-7B 为“未定”；RoboBrain 与配置不一致（未改配置）。这些是 dev 标定用的数，不是结果。
   - 2026-11-26 第 4 轮（`reports/round_4.md`）：planner 裁定 Qwen3-VL-8B 用 (P1, norm1000_xyxy)、RoboBrain2.0-7B 用 (P0, abs_model_xyxy)（配置已改）；SenseNova-1.2/1.3 用逐条按数值范围换算；VST 补测 P2 后仍判“未定”（提示词 pending）。各模型选定的（提示词, 约定）在 `configs/bbox_conventions.yaml` 的 `calibrated` 块。

2026-11-25 已决定并落实：真值框（mask 外接框）；COCO 划分统一；What's Up B 的 front / behind 保留并注明混淆；SpatialRGPT 的同名跳过与 “No.” 标记；合成场景暂不扩量；GLM-4.6V 的像素预算；GQA 同义词不补；眼镜口径；坏节点由你上报；GQA 按图像划分；COCO-Pairs 几何按 mask 外接框重算；trial_v1。此前了结的：F1 审核与 v2 / v3、质检阈值、四个大模型的冒烟（未做，让位给数据作业）。

---

## 9. 本阶段新增的磁盘占用

| 位置 | 占用 |
|---|---|
| `models/sam2.1-hiera-large`、`models/big-lama` | 1.80 GB + 0.38 GB |
| `envs/tools`（第三个环境，锁文件 `requirements/tools.lock`） | 约 7.8 G |
| `data/depth/spatialrgpt_bench` | 2.91 GB |
| `data/gqa` + 压缩包 | 约 0.4 G |
| `data/coco_pairs`、`data/whatsup_annotations`（v1–v3） | 约 27 MB |
| `data/synth`（抠图库 + 试生成） | 308 MB |
| `outputs/`（含 v1、v2 联系表、候选图、v3 确认图、诊断记录） | 246 MB |
| **本项目合计**（models 91 G、envs 24 G、cache 8.7 G、data 6.8 G、其余很小） | **约 131 G** |

### 抠图库 v2（2026-11-26，Xiwei 决定，planner 转达；预审中，结论出来之前不用于生成场景）

- 来源：COCO val2017（沿用原图像级划分）+ COCO train2017（按图像、固定种子分 dev/test）；v1 库保留不动，v2 存为 `<data>/synth/cutouts_v2`。规则 a–g 与泄漏排除见 `configs/synth/library_v2.yaml`、`src/synth/library_v2.py`，在看任何图之前提交。
- **每类最多 60 个 train 候选**（下载前按固定种子随机抽）：这是 runner 为控制下载量定的参数，不是 Xiwei / planner 给的，planner 已认可。它截掉了 8,528 个已通过 a–d 预筛的 train 候选。
- 下载：只下载候选所在的 3,006 张 train2017 图像（477 MB），标注包复用已有的。
- 泄漏：train2017 里 image_id 等于 GQA-spatial（dev 与 test）VG coco_id 的 51 个实例已排除。下载的 3,006 张 train 图与 5,114 张评测图（What's Up A/B、COCO-spatial、COCO-Pairs、COCO 留出池、SpatialRGPT）做整图感知哈希比对，最小汉明距离 ≥ 8，阈值 ≤ 4 下没有近重复。COCO-spatial、COCO-Pairs、留出池都来自 val2017，与 train2017 的 image_id 天然不重叠；SpatialRGPT 的图像来自 SUNRGBD / hypersim / nuScenes / ARKitScenes / KITTI，不是 COCO。
- 结果：2,976 个抠图，59 类两侧各 ≥ 3（门槛 30 类）；dev 1,486 / test 1,490；train 来源 2,706，val 来源 270；2,228 个通过 18 px 检查，56 类可进 36 px 档。逐类、逐规则的数量见 `manifests/synth_library_v2.json`。

### 抠图库 v3（2026-11-26，planner 定的口径；**已定稿，见本节末尾“定稿”**；合成场景 v2 的规格出来之前不生成场景）

- **来源写法：自动规则过滤，加模型辅助逐格预审排除（两轮，未经人工复核）；人工抽查结果另报。** 不得写成人工审核。
- 组成：v3 = v2（2,976 个）− knife、potted plant 两类 − 逐格预审排除的（`annotations/synth/library_v2_full_prereview.csv`，exclude=1；两轮预审中任一轮判 BAD 或 UNSURE 即排除）− 规则 c、d、e 剔除的；两侧达不到各 ≥ 3 的类别整类排除。不提高候选上限，不新增下载。
- 两轮预审在重叠的 1,374 格上，“排除与否”的一致率 0.842，kappa 0.65（planner 提供）。
- 剔除数量：整类 30 个；预审排除 941 个（其中规则 c / d / e 也会剔除的分别有 33 / 26 / 24 个）；规则 c、d、e 在预审之外**额外**剔除 12 个（c 4、d 8、e 0）。规则 d 的凸度阈值按 dev 定阈值的结果不用（T_sol = 0），只看连通块数（面积 ≥ 1% 的块）与 CLIP pile 概率 > 0.5。
- 结果：**1,980 个**抠图（dev 951 / test 1,029；train 来源 1,807，val 来源 173），**54 类**（门槛 30）；1,657 个通过 18 px 检查，52 类可进 36 px 档。
- 与 v2 相比少了 5 类：knife、potted plant（整类剔除），backpack（dev 1 / test 3）、frisbee（dev 7 / test 2）、tennis racket（预审后两侧都是 0），后三类是两侧达不到各 ≥ 3。逐类数量与增减见 `manifests/synth_library_v3.json`。
- **不用规则 a、b、f 的理由**（依据 dev 上的数字，见 `manifests/synth_library_v3_tune.json`）：a（遮挡接触）和 b（接触边长直边）在 v2 候选上不起作用，dev BAD 召回为 0，因为 v2 已要求外接框与同图其他实例不相交，切口来自未标注的遮挡物；f（SAM 2.1 重新分割的 IoU）在 T_sam = 0.85 时 dev 上召回 0.393、OK 损失 0.138，即每抓到 1 个 BAD 要误删约 1.6 个 OK，代价太高，由逐格预审替代。所有规则合起来在 OK 损失 ≤ 0.25 的约束下，dev BAD 召回最高只有 0.512（目标 0.80），没有放宽阈值。
- 审阅材料：全量联系表 `outputs/synth_review/library_v3_full/`（54 页，附 index.csv）；人工抽查表 `outputs/synth_review/library_v3_spotcheck/`（固定种子从 v3 随机抽 150 个，不分类别；只显示编号和类别，不显示 CLIP 概率和预审结论），用来估计残余的坏格比例。
- **定稿（2026-11-26）**：
  - **Xiwei 人工抽查**：从 v3 草稿（1,980 个，提交 0bd0f57）里用固定种子随机抽 150 个，联系表上不显示 CLIP 概率和预审结论。结果坏 3 个（2.0%，Wilson 95% CI 0.7%–5.7%），坏加拿不准共 6 个（4.0%，CI 1.9%–8.5%），达到定稿标准（≤ 5%，即 ≤ 7 个）。
  - **cow 类单类复核**：cow 在抽查里 6 个坏了 2 个，planner 因此对 cow 类单独复核（模型预审，未经人工复核）。
  - **定稿时再剔除 23 个**，清单在 `annotations/synth/library_v3_final_exclusions.csv`：
    - 抽查里的坏 3 个和拿不准 3 个，按 Xiwei 的要求一并剔除（source=human_spotcheck）；
    - planner 的 cow 复核 17 个（source=planner_cow_review）。
    - 按类别：cow 19、dog 1、sheep 1、cake 1、sandwich 1。
  - **要照实注意**：这 23 个剔除是在抽查之后做的，抽查估计的是剔除前草稿的残余坏格比例，不是定稿后的比例。
  - **定稿结果**：**1,957 个**（dev 941 / test 1,016；train 来源 1,784，val 来源 173），**54 类**，没有类别掉到两侧各 3 个以下；1,639 个通过 18 px 检查，52 类可进 36 px 档；cow 剩 dev 13 / test 12。
  - **文件**：定稿写在 `<data>/synth/cutouts_v3/library_final.jsonl`，草稿 `library.jsonl` 保留不动；`splits/synth_library_v3.json` 与 `manifests/synth_library_v3.json` 标为 final；全量联系表 `outputs/synth_review/library_v3_full/` 已按定稿重出，页眉标 FINAL。
  - **来源写法**：自动规则过滤，加模型辅助逐格预审排除（两轮，未经人工复核）；再按 Xiwei 人工抽查剔除 6 个，按 planner 的 cow 单类复核（模型预审，未经人工复核）剔除 17 个。

