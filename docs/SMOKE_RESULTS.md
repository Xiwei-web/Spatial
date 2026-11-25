# 冒烟测试结果（SMOKE_RESULTS）

由 `scripts/build_smoke_table.py` 从 `outputs/smoke/<模型>/summary.json` 生成。**只表示管线是否跑通，不是评测；不含任何正确率，不得用于比较模型。**

显存列为“加载后 / 全程峰值”GB（各卡之和，bf16，eager 注意力）；耗时为 60 次生成（40 次框 + 20 次关系）的平均秒数，含预处理。

| 模型 | 环境 | 1 加载 | 2 框 | 3 关系 | 4 隐状态 | 5 注意力/head | 6 视觉token映射 | 7 干预 | 8 确定性 | 显存 GB | 秒/样本 | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | tf5 | 通过 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 15.45 / — | — | `efc744e` |
| Qwen3-VL-8B-Instruct | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| RoboBrain2.0-7B | tf5 | 通过 | **失败** | **失败** | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.061 | `462d086` |
| VST-7B-RL | tf5 | 通过 | **失败** | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.635 | `462d086` |
| Qwen3.5-2B | tf5 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 4.12 / 9.08 | 1.503 | `52229e2` |
| Qwen3.5-4B | tf5 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 8.46 / 13.42 | 1.933 | `60f2f9a` |
| Qwen3.5-9B | tf5 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 17.53 / 22.49 | 1.901 | `60f2f9a` |
| Qwen3.6-27B | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| Qwen3.8-27B | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| Qwen3.6-35B-A3B | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| GLM-4.6V-Flash | tf5 | 通过 | 通过 | **失败** | 通过 | 通过 | 通过 | 通过 | 通过 | 19.17 / 34.43 | 1.694 | `462d086` |
| gemma-4-31B-it | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| LLaVA-OneVision-7B | tf5 | 通过 | **失败** | 通过 | 通过 | **失败** | 通过 | **失败** | 通过 | 14.96 / 38.38 | 3.202 | `462d086` |
| LLaVA-1.5-7B | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| LLaVA-1.5-13B | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| InternVL3-8B | tf4 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| InternVL3-2B | tf4 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| InternVL3_5-8B | tf4 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.89 / 16.97 | 0.946 | `60f2f9a` |
| SenseNova-SI-1.1-InternVL3-2B | tf4 | 通过 | **失败** | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 3.89 / 4.97 | 0.825 | `60f2f9a` |
| SenseNova-SI-1.2-InternVL3-8B | tf4 | 通过 | **失败** | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.773 | `5337b5e` |
| SenseNova-SI-1.3-InternVL3-8B | tf4 | 通过 | **失败** | 通过 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.77 | `462d086` |
| MiniCPM-V-4_5 | tf4 | 通过 | **失败** | 通过 | 通过 | 通过 | **失败** | 通过 | 通过 | 17.37 / 17.83 | 0.809 | `462d086` |
| Cambrian-S-7B | — | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |

## 失败项与说明

- **Qwen3-VL-8B-Instruct**：未运行。未运行
- **RoboBrain2.0-7B** / 2_bbox：官方提示词=True，约定=abs_orig_xyxy，解析出四个数 15/40，还原 15/40；
- **RoboBrain2.0-7B** / 3_relation：解析出选项 0/20，四个选项 logit 齐全 20/20
- **VST-7B-RL** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 11/40，还原 0/40；
- **Qwen3.5-2B** / 非标准层：18 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.5-4B** / 非标准层：24 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.5-9B** / 非标准层：24 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.6-27B**：未运行。未运行
- **Qwen3.8-27B**：未运行。未运行
- **Qwen3.6-35B-A3B**：未运行。未运行
- **GLM-4.6V-Flash** / 3_relation：解析出选项 0/20，四个选项 logit 齐全 20/20
- **gemma-4-31B-it**：未运行。未运行
- **LLaVA-OneVision-7B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **LLaVA-OneVision-7B** / 5_attention_and_heads：{"nonstandard": {"has_nonstandard": false}}
- **LLaVA-OneVision-7B** / 7_interventions：{}
- **LLaVA-OneVision-7B** / 5a_attention_weights：OutOfMemoryError: CUDA out of memory. Tried to allocate 5.70 GiB. GPU 0 has a total capacity of 39.49 GiB of which 546.31 MiB is free. Including non-PyTorch memory, this process has 38.95 GiB memory in use. Of the allocated memory 35.54 GiB is allocated by PyTorch, and 2.91 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorc
- **LLaVA-OneVision-7B** / 5b_head_hook：OutOfMemoryError: CUDA out of memory. Tried to allocate 5.70 GiB. GPU 0 has a total capacity of 39.49 GiB of which 190.31 MiB is free. Including non-PyTorch memory, this process has 39.30 GiB memory in use. Of the allocated memory 30.37 GiB is allocated by PyTorch, and 8.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorc
- **LLaVA-OneVision-7B** / 7b_self_patch：OutOfMemoryError: CUDA out of memory. Tried to allocate 4.19 GiB. GPU 0 has a total capacity of 39.49 GiB of which 3.37 GiB is free. Including non-PyTorch memory, this process has 36.12 GiB memory in use. Of the allocated memory 27.70 GiB is allocated by PyTorch, and 7.92 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.
- **LLaVA-OneVision-7B** / 7c_attention_knockout：OutOfMemoryError: CUDA out of memory. Tried to allocate 5.70 GiB. GPU 0 has a total capacity of 39.49 GiB of which 546.31 MiB is free. Including non-PyTorch memory, this process has 38.95 GiB memory in use. Of the allocated memory 35.54 GiB is allocated by PyTorch, and 2.91 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorc
- **LLaVA-1.5-7B**：未运行。未运行
- **LLaVA-1.5-13B**：未运行。未运行
- **InternVL3-8B**：未运行。未运行
- **InternVL3-2B**：未运行。未运行
- **SenseNova-SI-1.1-InternVL3-2B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **SenseNova-SI-1.2-InternVL3-8B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 31/40，还原 0/40；
- **SenseNova-SI-1.3-InternVL3-8B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 34/40，还原 0/40；
- **MiniCPM-V-4_5** / 2_bbox：官方提示词=True，约定=norm1000_xyxy，解析出四个数 0/40，还原 0/40；
- **MiniCPM-V-4_5** / 6_visual_token_mapping：视觉 token 没有固定的图像区域对应
- **Cambrian-S-7B**：未运行。model_type=cambrian_qwen 不在 transformers 中，需要官方 cambrian-s 代码库及其依赖环境；本阶段未适配
