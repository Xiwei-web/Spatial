# Round 6：写死关系类提示词与解析规则（只用 CPU），前缀复用估算

状态：**完成**（2026-11-26）。没有跑任何 GPU 作业，没有读任何模型在 dev 或 test 上的新输出。模板和解析在 Xiwei 审批前不使用。

## 一句话结论

三类新模板、yes/no 解析、GLM 的备选顺序已写进 `src/prompts.py`，40 项单元测试全部用手写字符串。16 个模型的图像 token 都在问题文字之前，关系问法可以复用图像前缀；粗估可省掉关系部分约 47%：全量约 25 GPU 小时，省钱档约 4.5 GPU 小时，主跑和默认实现各算一次。

## 一、模板全文（`src/prompts.py`：`e0_mcq`、`binary_question`）

- 两物体四选一沿用 What's Up 的句式，rotation=0 时与原 `relation_mcq` 逐字相同（有测试）。一物体四选一的关系相对整张图。
- 轮换：rotation = r 时选项列表循环左移 r 位，字母始终是 A–D；4 次轮换里每个关系在 A、B、C、D 各出现一次（有测试）。
- 关系短语（两物体）：left “to the left of”、right “to the right of”、on、under、front “in front of”、behind、above、below。一物体：left “on the left side of the image”、right “on the right side of the image”、top “at the top of the image”、bottom “at the bottom of the image”。
- 反关系：left↔right、on↔under、front↔behind、above↔below、top↔bottom。
- 剔除（照实记录，planner 第 6 轮）：GQA one-object test 的 1 条 behind，GQA two-object dev 的 3 条 top。

下面是 rotation=0 的全文示例，每个数据集一个：

**whatsup_a**（选项 left/right/on/under；示例 target=mug，reference=table，真关系 on）

```
Look at the image. Where is the mug relative to the table?
A. The mug is to the left of the table.
B. The mug is to the right of the table.
C. The mug is on the table.
D. The mug is under the table.
Answer with the option's letter only.
```

二元：`Is the mug on the table? Answer yes or no.` / `Is the mug under the table? Answer yes or no.`

**whatsup_b**（选项 left/right/front/behind；示例 target=cup，reference=bowl，真关系 front）

```
Look at the image. Where is the cup relative to the bowl?
A. The cup is to the left of the bowl.
B. The cup is to the right of the bowl.
C. The cup is in front of the bowl.
D. The cup is behind the bowl.
Answer with the option's letter only.
```

二元：`Is the cup in front of the bowl? Answer yes or no.` / `Is the cup behind the bowl? Answer yes or no.`

**coco_spatial_two**（选项 left/right/above/below；示例 target=dog，reference=car，真关系 left）

```
Look at the image. Where is the dog relative to the car?
A. The dog is to the left of the car.
B. The dog is to the right of the car.
C. The dog is above the car.
D. The dog is below the car.
Answer with the option's letter only.
```

二元：`Is the dog to the left of the car? Answer yes or no.` / `Is the dog to the right of the car? Answer yes or no.`

**coco_pairs**（选项 left/right/above/below；示例 target=bed，reference=chair，真关系 below）

```
Look at the image. Where is the bed relative to the chair?
A. The bed is to the left of the chair.
B. The bed is to the right of the chair.
C. The bed is above the chair.
D. The bed is below the chair.
Answer with the option's letter only.
```

二元：`Is the bed below the chair? Answer yes or no.` / `Is the bed above the chair? Answer yes or no.`

**gqa_spatial_two**（选项 left/right/front/behind；示例 target=man，reference=horse，真关系 behind）

```
Look at the image. Where is the man relative to the horse?
A. The man is to the left of the horse.
B. The man is to the right of the horse.
C. The man is in front of the horse.
D. The man is behind the horse.
Answer with the option's letter only.
```

二元：`Is the man behind the horse? Answer yes or no.` / `Is the man in front of the horse? Answer yes or no.`

**coco_spatial_one**（选项 left/right/top/bottom；示例 target=bus，真关系 top）

```
Look at the image. Where is the bus in the image?
A. The bus is on the left side of the image.
B. The bus is on the right side of the image.
C. The bus is at the top of the image.
D. The bus is at the bottom of the image.
Answer with the option's letter only.
```

二元：`Is the bus at the top of the image? Answer yes or no.` / `Is the bus at the bottom of the image? Answer yes or no.`

**gqa_spatial_one**（选项 left/right/top/bottom；示例 target=tree，真关系 left）

```
Look at the image. Where is the tree in the image?
A. The tree is on the left side of the image.
B. The tree is on the right side of the image.
C. The tree is at the top of the image.
D. The tree is at the bottom of the image.
Answer with the option's letter only.
```

二元：`Is the tree on the left side of the image? Answer yes or no.` / `Is the tree on the right side of the image? Answer yes or no.`

## 二、解析规则

- **四选一**：`parse_mcq_e0`。先用冒烟时的 `parse_mcq`，它本身不改：去掉 `<think>…</think>`；开头的 `<answer>` / `<|begin_of_box|>` 包装取里面的内容；开头是 A–D（可带括号、句点）；或者文中有 “answer/option (is|:) X”。都不行时，再找文中任意位置的 `<|begin_of_box|>X<|end_of_box|>`，对应 GLM 先推理、最后给出框起来的字母。
- **二元**：`parse_yes_no`。去掉 `<think>…</think>`，取 `<answer>…</answer>` 或 `<|begin_of_box|>…<|end_of_box|>` 里的内容；然后**只认第一个独立的 yes / no 词**，不区分大小写。“know”“nothing”“not”“Yesterday” 不算。解析不出返回 None，计入分母。已知的弱点：“yes or no? no” 会被判成 yes（只认第一个词）。测试里写明了这一条。
- **答案获取方式的顺序**（`ANSWER_MODE_ORDER`、`choose_answer_mode`；dev 解析检查每个数据集 50 个样本，解析率 ≥ 0.95 就停在当前一步）：
  - 默认：text16（生成 16 个 token 后解析）→ text64（生成 64 个 token 后解析）→ letter_logit（取第一步四个字母 token 的 logit 最大者；二元问法用 Yes/No 两个 token）。
  - GLM-4.6V-Flash：从 text64 开始 → letter_logit。
  - 所有模型都照样保存四个字母（及 Yes/No）的第一步 logit。
  - 不过线的模型只能按这个顺序往下走，不改模板措辞。
- 测试：`tests/test_prompts_e0.py`（新）加上 `tests/test_parsing_and_data.py`，共 40 项通过。

## 三、复用图像前缀 KV cache 的可行性与节省（只读代码 + 分词器 / 处理器，不实现）

脚本 `scripts/e0_prefix_reuse_estimate.py` → `manifests/e0_prefix_reuse_estimate.json`。

- **可行性**：16 个模型的对话模板里，图像都在问题文字之前。Qwen 系 / VST / RoboBrain / GLM 是 `<|vision_start|>…<|vision_end|>问题`，InternVL / SenseNova 是 `<img>…</img>\n问题`，LLaVA-1.5 是 `USER: <image>\n问题`。用同一张图配两个不同问题实际分词，两条序列的公共前缀都覆盖了全部视觉 token，在最后一个视觉 token 之后 1–3 个 token 才分开。所以同一张图的 4 次四选一加 2 次二元，可以共用一次视觉编码和图像前缀的 prefill。
- **估算方法**：每条关系问法的秒数（冒烟）≈ 前缀 + 解码。每个 token 的解码秒数用 bbox_calib 与冒烟的耗时差、平均新 token 数之差来估。前缀秒数 = 关系秒数 − 新 token 数 × 每 token 秒数（问题文字的 prefill 忽略不计，所以偏乐观）。一个样本 6 次问法共用一次前缀，可省 5/6 × 前缀。

| 模型 | 输入 token / 最后一个视觉 token 下标 / 两问公共前缀 | 前缀覆盖全部视觉 token | 关系每条秒数中前缀占比 | 关系部分 GPU 小时（全量 / 省钱） | 可省（全量 / 省钱） |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 1325 / 1244 / 1246 | 是 | 0.93 | 4.63 / 0.84 | 3.58 / 0.65 |
| Qwen3-VL-8B-Instruct | 1056 / 975 / 977 | 是 | 0.81 | 3.26 / 0.55 | 2.21 / 0.37 |
| InternVL3-8B | 341 / 259 / 262 | 是 | 0.63 | 1.31 / 0.21 | 0.69 / 0.11 |
| InternVL3_5-8B | 341 / 259 / 262 | 是 | 0.46 | 1.33 / 0.21 | 0.51 / 0.08 |
| SenseNova-SI-1.2-InternVL3-8B | 341 / 259 / 262 | 是 | 0.28 | 3.79 / 0.61 | 0.88 / 0.14 |
| SenseNova-SI-1.3-InternVL3-8B | 341 / 259 / 262 | 是 | 0.23 | 3.14 / 0.55 | 0.59 / 0.10 |
| GLM-4.6V-Flash | 1316 / 1234 / 1236 | 是 | 0.54 | 5.83 / 1.02 | 2.62 / 0.46 |
| VST-7B-RL | 1325 / 1244 / 1246 | 是 | 0.92 | 3.05 / 0.69 | 2.33 / 0.53 |
| RoboBrain2.0-7B | 1325 / 1244 / 1246 | 是 | 0.71 | 5.10 / 1.02 | 3.02 / 0.60 |
| Qwen3.5-2B | 1065 / 975 / 977 | 是 | 0.62 | 3.49 / 0.58 | 1.80 / 0.30 |
| Qwen3.5-4B | 1065 / 975 / 977 | 是 | 0.38 | 7.34 / 1.20 | 2.32 / 0.38 |
| Qwen3.5-9B | 1065 / 975 / 977 | 是 | 0.64 | 4.68 / 0.77 | 2.49 / 0.41 |
| InternVL3-2B | 341 / 259 / 262 | 是 | 0.40 | 0.83 / 0.13 | 0.28 / 0.04 |
| SenseNova-SI-1.1-InternVL3-2B | 341 / 259 / 262 | 是 | 0.11 | 3.47 / 0.55 | 0.31 / 0.05 |
| LLaVA-1.5-7B | 683 / 580 / 583 | 是 | 0.62 | 1.19 / 0.19 | 0.62 / 0.10 |
| LLaVA-1.5-13B | 683 / 580 / 583 | 是 | 0.69 | 1.83 / 0.29 | 1.05 / 0.17 |

- **合计**：关系部分全量 54.3 GPU 小时，可省约 25.3（47%）；省钱档 9.4，可省约 4.5。主跑和默认实现那一遍各适用一次，全量档最多省约 50 GPU 小时（173 → 约 123），省钱档约 9（44 → 约 35）。框的两次提问（target、reference）也可以共用同一前缀，这部分没有估。
- **风险与限制**：
  1. bf16 下，“先算前缀再接问题”与“整段一起算”数值不同，输出不一定与不复用时逐 token 相同。干预实验的基线必须用同一种做法。
  2. 默认实现那一遍如果也复用前缀，就不再是“默认的推理方式”。建议默认实现那一遍不复用，只给主跑用。这样全量档约省 25，省钱档约省 4.5。
  3. Qwen3.5 系 3/4 的层是 Gated DeltaNet，缓存里有递归状态。复制状态在原理上可行，但 transformers 对这类缓存的复制没有验证过。
  4. InternVL 远程代码的 generate 走 inputs_embeds，需要自己写解码循环。Qwen 系的 mRoPE 位置要正确接上，同第 2 轮 teacher-forced 读出时的做法。
  5. 估算只基于冒烟的 20 条关系问法；SenseNova、RoboBrain、Qwen3.5-4B 的关系回答较长，前缀占比低，省得少。GLM 改成 text64 后的耗时没算进去。

## 四、过程中的一个问题（已处理）

第一次运行估算脚本时，SenseNova 系的分词器按 `auto_map` 从 HF Hub 取远程代码，我的脚本**没有像 adapter 那样传 `code_revision`**，transformers 提示“A new version of configuration_internvl_chat.py was downloaded”。查了缓存：Hub 上 `main` 当前就指向配置里固定的 commit（81c0e4c5 / 57bfba82 / 2081462c），snapshots 里没有新 blob，加载的代码与固定版本相同。缓存里只新写了 3 个 `refs/main` 文件，没有删除或改动任何已有文件。已在脚本里补上 `code_revision`，重跑结果逐字节相同，也没有再下载。

## 五、提交

- `src/prompts.py`、`tests/test_prompts_e0.py`、`scripts/e0_prefix_reuse_estimate.py`、`manifests/e0_prefix_reuse_estimate.json`、本报告。

## 六、需要决定的事

1. 模板措辞（第一节）：planner 预审后转 Xiwei。
2. 前缀复用要不要实现。建议只用于主跑、默认实现那一遍不用；要实现的话，先在 dev 上对比复用与不复用的逐 token 一致率。
