# 冒烟测试结果（SMOKE_RESULTS）

由 `scripts/build_smoke_table.py` 从 `outputs/smoke/<模型>/summary.json` 生成。**只表示管线是否跑通，不是评测；不含任何正确率，不得用于比较模型。**

显存列为“加载后 / 全程峰值”GB（各卡之和，bf16，eager 注意力）；耗时为 60 次生成（40 次框 + 20 次关系）的平均秒数，含预处理。

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

## 失败项与说明

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
