# Round 7：COCO / GQA / COCO-Pairs 改为二选一；yes/no 解析修补；字母 logit 规则；重算估算（只用 CPU）

状态：**完成**（2026-11-26）。没有跑 GPU 作业，也没有读任何模型的新输出。

## 一句话结论

COCO-spatial、GQA-spatial、COCO-Pairs 改为同轴二选一，What's Up A/B 仍是四选一。重算后三档约为：全量 140、中档 62、省钱 42.5 GPU 小时，原来是 173 / 69 / 44，仍都超过 24 小时。前缀复用不实现。

## 一、改动（`src/prompts.py`、`tests/test_prompts_e0.py`）

1. **二选一**：`e0_mcq(dataset, target, reference, rotation, relation)`。COCO/GQA/COCO-Pairs 只给真关系所在轴的两个选项（A/B），两种顺序各问一次，“轮换后准确率”取两次的平均。选项列表按轴的固定顺序排（left/right、above/below、top/bottom、front/behind），不按真关系排，免得真答案总在 A。What's Up A/B 仍是四选一，rotation=0 时与原 `relation_mcq` 逐字相同。选择题次数由 `n_choices(dataset)` 给出（4 或 2）。关系不在选项集合里就报错：这些样本应已剔除。
2. **parse_yes_no**：去掉 think 和包装后，先删掉字面短语 “yes or no”（不区分大小写，中间允许任意空白），再认第一个 yes/no 词。“yes or no? no” 现在判为 no；“YES OR NO” 判为解析失败；“Answer yes or no. Yes.” 判为 yes。
3. **letter_logit 的 token**：
   - `letter_logit_answer`：每个字母取 “A”、“ A” 两种变体里 logit 最大的，再在字母之间取最大。
   - `yes_no_logit_answer`：比较 {Yes, yes, “ Yes”, “ yes”} 的最大 logit 与 {No, no, “ No”, “ no”} 的最大 logit。
   - `variant_token_ids`：变体在分词器里恰好编码成一个 token 才算存在，否则跳过。某个字母一个变体都没有时，返回 None。
   - 注意：Llama 系（LLaVA-1.5）的分词器会把 “A” 编成带前缀空格的 “▁A”，“ A” 则编成两个 token 而被跳过。这条规则照写，没有为个别分词器特殊处理。
4. **测试**：用手写字符串和一个手写的假分词器，`tests/test_prompts_e0.py` 加 `tests/test_parsing_and_data.py` 共 48 项通过。
5. 物体名照原样用。GQA 的 “black trees”“green trees” 这类复数带来的语法瑕疵（如 “Where is the black trees …”）不修，照实记录。

## 二、模板全文（每类一个示例）

**whatsup_a**（四选一；示例 target=mug，reference=table，真关系 on）

rotation=0（选项顺序 left/right/on/under）：
```
Look at the image. Where is the mug relative to the table?
A. The mug is to the left of the table.
B. The mug is to the right of the table.
C. The mug is on the table.
D. The mug is under the table.
Answer with the option's letter only.
```

rotation=1（选项顺序 right/on/under/left）：
```
Look at the image. Where is the mug relative to the table?
A. The mug is to the right of the table.
B. The mug is on the table.
C. The mug is under the table.
D. The mug is to the left of the table.
Answer with the option's letter only.
```

rotation=2、3 依次再左移一位。

二元：`Is the mug on the table? Answer yes or no.` / `Is the mug under the table? Answer yes or no.`

**whatsup_b**（四选一；示例 target=cup，reference=bowl，真关系 front）

rotation=0（选项顺序 left/right/front/behind）：
```
Look at the image. Where is the cup relative to the bowl?
A. The cup is to the left of the bowl.
B. The cup is to the right of the bowl.
C. The cup is in front of the bowl.
D. The cup is behind the bowl.
Answer with the option's letter only.
```

rotation=1（选项顺序 right/front/behind/left）：
```
Look at the image. Where is the cup relative to the bowl?
A. The cup is to the right of the bowl.
B. The cup is in front of the bowl.
C. The cup is behind the bowl.
D. The cup is to the left of the bowl.
Answer with the option's letter only.
```

rotation=2、3 依次再左移一位。

二元：`Is the cup in front of the bowl? Answer yes or no.` / `Is the cup behind the bowl? Answer yes or no.`

**coco_spatial_two**（二选一（同轴）；示例 target=dining table，reference=refrigerator，真关系 left）

rotation=0（选项顺序 left/right）：
```
Look at the image. Where is the dining table relative to the refrigerator?
A. The dining table is to the left of the refrigerator.
B. The dining table is to the right of the refrigerator.
Answer with the option's letter only.
```

rotation=1（选项顺序 right/left）：
```
Look at the image. Where is the dining table relative to the refrigerator?
A. The dining table is to the right of the refrigerator.
B. The dining table is to the left of the refrigerator.
Answer with the option's letter only.
```

二元：`Is the dining table to the left of the refrigerator? Answer yes or no.` / `Is the dining table to the right of the refrigerator? Answer yes or no.`

**coco_pairs**（二选一（同轴）；示例 target=bed，reference=chair，真关系 below）

rotation=0（选项顺序 above/below）：
```
Look at the image. Where is the bed relative to the chair?
A. The bed is above the chair.
B. The bed is below the chair.
Answer with the option's letter only.
```

rotation=1（选项顺序 below/above）：
```
Look at the image. Where is the bed relative to the chair?
A. The bed is below the chair.
B. The bed is above the chair.
Answer with the option's letter only.
```

二元：`Is the bed below the chair? Answer yes or no.` / `Is the bed above the chair? Answer yes or no.`

**gqa_spatial_two**（二选一（同轴）；示例 target=cart，reference=green trees，真关系 front）

rotation=0（选项顺序 front/behind）：
```
Look at the image. Where is the cart relative to the green trees?
A. The cart is in front of the green trees.
B. The cart is behind the green trees.
Answer with the option's letter only.
```

rotation=1（选项顺序 behind/front）：
```
Look at the image. Where is the cart relative to the green trees?
A. The cart is behind the green trees.
B. The cart is in front of the green trees.
Answer with the option's letter only.
```

二元：`Is the cart in front of the green trees? Answer yes or no.` / `Is the cart behind the green trees? Answer yes or no.`

**coco_spatial_one**（二选一（同轴）；示例 target=dining table，真关系 bottom）

rotation=0（选项顺序 top/bottom）：
```
Look at the image. Where is the dining table in the image?
A. The dining table is at the top of the image.
B. The dining table is at the bottom of the image.
Answer with the option's letter only.
```

rotation=1（选项顺序 bottom/top）：
```
Look at the image. Where is the dining table in the image?
A. The dining table is at the bottom of the image.
B. The dining table is at the top of the image.
Answer with the option's letter only.
```

二元：`Is the dining table at the bottom of the image? Answer yes or no.` / `Is the dining table at the top of the image? Answer yes or no.`

**gqa_spatial_one**（二选一（同轴）；示例 target=purse，真关系 right）

rotation=0（选项顺序 left/right）：
```
Look at the image. Where is the purse in the image?
A. The purse is on the left side of the image.
B. The purse is on the right side of the image.
Answer with the option's letter only.
```

rotation=1（选项顺序 right/left）：
```
Look at the image. Where is the purse in the image?
A. The purse is on the right side of the image.
B. The purse is on the left side of the image.
Answer with the option's letter only.
```

二元：`Is the purse on the right side of the image? Answer yes or no.` / `Is the purse on the left side of the image? Answer yes or no.`

## 三、新的估算（`scripts/e0_estimate.py` → `manifests/e0_estimate.json`）

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

全量主跑的份额（16 个模型合计，不含加载，单位 GPU 小时）：What's Up A 5.3、B 5.2、COCO one 12.6、COCO two 3.4、GQA one 7.0、GQA two 2.5、COCO-Pairs 26.2。

| 档 | 内容 | GPU 小时 | 墙钟（2 个作业同时跑，不含排队） |
|---|---|---|---|
| 全量 | 主跑 62.9 + 默认实现全量 62.9 + dev 筛选 11.2 + dev 解析检查 2.8 | **≈ 140**（原 173） | ≈ 70 小时 |
| 中档 | 主跑去掉 COCO-Pairs 36.7 + 默认实现只跑 What's Up 11.3 + dev 筛选 11.2 + 解析检查 2.8 | **≈ 62**（原 69） | ≈ 31 小时 |
| 省钱 | 主跑 What's Up A/B、COCO two、GQA two 17.2 + 默认实现只跑 What's Up 11.3 + dev 筛选 11.2 + 解析检查 2.8 | **≈ 42.5**（原 44） | ≈ 21 小时 |

逐模型的数在 `docs/E0_PLAN.md` 第 5 节。合成集每 1,000 张图约 25.6、深度集每 100 条约 2.6 GPU 小时，没有变化。

## 四、文档

- `docs/E0_PLAN.md`：2.1 节写明二选一与改动原因；2.2 节换上新条数表；2.3 节的状态表改为“事先写死，planner 已预审”，并写清答案获取顺序和 letter_logit 规则；第 3 节写明**前缀复用不实现**（原因是实现风险，而且 bf16 下复用与否输出不逐 token 相同）；第 5 节换上新估算；第 6 节加了一条提醒：80% 这条线是按四选一写的，二选一的随机水平是 50%，同一条线在那里更容易达到。
- 前缀复用的参考估算已按新条数重跑（`manifests/e0_prefix_reuse_estimate.json`）：关系部分全量 38.0 GPU 小时，可省约 17.7；省钱档 8.1，可省约 3.9。不实现。

## 五、需要决定的事（转 Xiwei）

1. 用哪一档（约 140 / 62 / 42.5 GPU 小时）。
2. 入选用 What's Up dev 还是 test 判定。
3. 模板全文（第二节）的审批。
