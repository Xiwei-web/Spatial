# F7：SpatialRGPT-Bench

> **更新（2026-11-25 夜）**：真值框用全部 mask 像素的外接框；mask 里的零星碎点会撑大外接框（192 个物体里 19 个去碎点前后 IoU < 0.95），是否去碎点待定。联系表：`outputs/data_review/spatialrgpt_box/`。

> **更新（2026-11-25 晚）**：用户决定：两个区域同名的 14 条深度样本跳过；只回答 “No.” 一类的样本保留、标记（`extra["relation_inferred_from_bare_no"]`）并单独报告；真值框用 mask 外接框（自带框在 `extra["provided_bbox_*"]`）；改写规则仍等用户看过 20 条对照后再全量。
>
> 跳过同名后可用 **96** 条（front 52 / behind 44）：答案明确的 67 条（front 35 / behind 32），靠推断的 29 条（front 17 / behind 12；原来 34 条里有 5 条因同名被跳过）。
>
> **这些样本也有垂直位置捷径**（`manifests/spatialrgpt_depth_cues.json`）：规则“target 的 mask 下边缘更低 = front”在 96 条上 88.5%，在答案明确的 67 条上 92.5%（多数类 52.2%）；面积比几乎没有信息。

日期 2026-11-25。代码：`src/data/spatialrgpt.py`（加载、front/behind 映射、改写规则、统计）、`scripts/spatialrgpt_rewrite_preview.py`（20 条预览）、`tests/test_gqa_spatialrgpt.py`。统计数字来自 `python -m src.data.spatialrgpt`（写到 `manifests/spatialrgpt_stats.json`）。没有跑任何模型，没有改写全量数据。

**使用限制：SpatialRGPT-Bench 完整数据仅用于评测，不得再分发；HF 卡片未声明许可。**（卡片 README 的元数据里没有 `license` 字段，HF API 的 `cardData` 也没有；数据不进仓库。）

## 下载

| 项 | 内容 |
|---|---|
| 来源 | HF dataset `a8cheng/SpatialRGPT-Bench`，固定 commit `01b38d2c7aa805223518c061f2a7602ad8f491ef`，匿名下载，未设门槛（`gated: false`） |
| 方式 | `bash scripts/download_data.sh spatialrgpt`（`huggingface_hub.snapshot_download(..., local_dir=data/depth/spatialrgpt_bench)`；新目标，不并入 `all`） |
| 文件数 / 体积 | **1,415 个文件，2,906,318,103 B（2.71 GiB）**：`images/` 1,406 个 PNG 共 1,066,306,595 B；`data/` 6 个 parquet 共 1,835,271,866 B；`SpatialRGPT-Bench_v1.json` 4,736,691 B；`README.md` 532 B；`.gitattributes` 2,419 B（不含 huggingface_hub 自己写的 `.cache/` 元数据） |
| 校验 | `data/depth/spatialrgpt_bench/SHA256SUMS`（1,415 行）；该清单自身的 sha256 = `28ca4b8bda1fb14763bae78a0a31cce9e99faac762a00f8f1a51eb283e73ec15`。`SpatialRGPT-Bench_v1.json` 的 sha256 = `2da6f65096994727f98bf243c850c209d7dea3b9755ff0f18f63d20f346997e6`，与之前 `data/depth_candidates/` 里那份逐字节相同（旧目录没动）。 |

### train-* 与 val-* parquet：逐字节相同

| 分片 | train sha256 | val sha256 |
|---|---|---|
| 00000-of-00003（297,448,559 B） | `17488e88…c959d081` | `17488e88…c959d081` |
| 00001-of-00003（310,667,068 B） | `cc9333a4…f8812100e0` | `cc9333a4…f8812100e0` |
| 00002-of-00003（309,520,306 B） | `80fe15d1…c9ebd31e38` | `80fe15d1…c9ebd31e38` |

三对分片 sha256 两两相同，即 `train-*` 是 `val-*` 的完整重复（多占 0.92 GB）。卡片 README 的配置只声明了 `val` 一个 split（`path: data/val-*`，1,406 条），`train-*` 没有被引用。

### parquet / JSON / images 三者的关系（全量核对）

- parquet（val 三片）共 1,406 行，列为 `id, image_info, qa_info, conversations, text_q, bbox, rle, image`；逐行与 JSON 对比，`text_q / qa_info / conversations / bbox / rle / image_info` **1,406/1,406 相同**。
- parquet 的 `image.bytes` 与 `images/<id>.png` 的 sha256 **1,406/1,406 相同**。
- `images/` 正好 1,406 个文件，JSON 的 `file_path` 全是 `images/<id>.png`，**全部 1,406 条都有图，没有多余文件**。1,406 个 PNG 里只有 688 个不同的 sha256（同一张原图被多条题目复用，与 `image_info.file_path` 的 688 个不同取值一致）。

所以加载器只用 JSON + `images/` 文件夹，**返回图像路径，不从 parquet 取 PIL**；parquet 用不到（留着是为了快照完整）。

## 加载器

`src.data.spatialrgpt.load_records()` 返回 1,406 条 `Record`：`id`、`image_path`（本地 `images/<id>.png`）、`image_source`（`image_info.dataset`）、`image_wh`、`qa_type`、`category`、`classes`（`qa_info.class` 原样）、`n_regions`、`question_mask`（对话写法，区域是 `<mask>`，已去掉开头的 `<image>\n `；原串在 `extra['conversation_q_raw']`）、`question_region`（`text_q`，区域是 “Region [k]”）、`answer`、`bboxes`（每个区域一个 xyxy）、`rles`（每个区域一个 COCO 压缩 RLE，`size = [h, w]`）、`extra['source_file_path']`（原数据集里的相对路径）。

核对过的事实：

- `<mask>` 写法与 `text_q` 的对应：把 `text_q` 里的 “Region [k]” 换成 `<mask>` 再加上 `<image>\n ` 前缀，与对话里的问题 **1,406/1,406 完全相同**；`text_q` 里区域总是按 [0]、[1] 的顺序出现，所以第 k 个 `<mask>` 就是 Region [k]。
- 每条都是一问一答（`conversations` 长度 2）。
- 框数 = mask 数 = 问题引用的区域数 = `<mask>` 个数：**1,406/1,406**（两个区域 1,140 条，一个区域 266 条）。带框 1,406/1,406（100%），带 mask 1,406/1,406（100%），共 2,546 个区域。
- 图像实际尺寸 = `image_info` 的宽高：1,406/1,406；RLE 的 `size` = 图像 (h, w)：2,546/2,546；框都在图像范围内：2,546/2,546；没有空 mask。框是 `[x1, y1, x2, y2]` 原图像素。
- **框不是 mask 的紧框**：框与 “mask 的外接框” 的 IoU 均值 0.78、中位数 0.80，2,546 个里 2,139 个 IoU < 0.9、77 个 < 0.5；在 1 像素内重合的只有 16 个。mask 的像素平均 98.9% 落在框内。按来源：hypersim 0.84、ARKitScenes 0.80、SUNRGBD 0.78、KITTI 0.76、nuScenes 0.68。看起来框比 mask 松（我猜是 3D 框的投影，但数据集里没有写，未证实）。用哪一个当定位真值需要你定。
- `qa_info.class[k]` 对应 Region [k]：数据集里没有文字说明。我抽了 12 条画框目检（6 条双区域、6 条 “单区域但 class 有 2 个名字”），12 条都符合 “class[k] ↔ 第 k 个框，单区域题对应 class[0]”。这只是 12 条的抽查。

## 各类别数量

定性 657 条 + 定量 749 条 = 1,406 条。

| 类别 | 条数 |
|---|---|
| above_choice | 29 |
| above_predicate | 37 |
| behind_predicate | 55 |
| below_choice | 19 |
| below_predicate | 35 |
| big_predicate | 61 |
| direction | 107 |
| distance_data | 148 |
| front_predicate | 55 |
| height_data | 133 |
| horizontal_distance_data | 122 |
| left_choice | 28 |
| left_predicate | 25 |
| right_choice | 24 |
| right_predicate | 28 |
| short_choice | 33 |
| short_predicate | 22 |
| small_predicate | 45 |
| tall_choice | 31 |
| tall_predicate | 26 |
| thin_predicate | 48 |
| vertical_distance_data | 106 |
| wide_predicate | 56 |
| width_data | 133 |

图像来源（全部）：SUNRGBD 594、hypersim 384、nuScenes 189、ARKitScenes 140、KITTI_object 99。

## front / behind（深度关系）子集 → 统一 `Sample`

`src.data.spatialrgpt.depth_samples()` 返回 **110 条**（`front_predicate` 55 + `behind_predicate` 55），`dataset = "spatialrgpt_depth"`。它没有注册进 `src.data.DATASETS`（没有划分文件；要不要注册、怎么划分等你定）。

### target / reference / label 的推导

两个类别的问题模板共 15 种，全部是 “Region [0] 是否在 Region [1] 的前面 / 后面”，主语都是 Region [0]：

- `front_predicate`（8 种）：直接问“在…前面”，或用 come / lie / positioned 等动词，或问“离相机 / 观察者更近”“距离更小”。
- `behind_predicate`（7 种）：与上面对称（“在…后面”“更远”“距离更大”）。模板原文不在仓库里转录（SpatialRGPT 只允许评测、不得再分发）；格式示意（手写）：`Is Region [0] nearer to us than Region [1]?`。

因此：

- `target` = `qa_info.class[0]`（Region [0]），`reference` = `qa_info.class[1]`（Region [1]）；`bbox_target / mask_target` = 第 0 个框 / RLE，`bbox_reference / mask_reference` = 第 1 个。
- 标准答案是 yes/no（`extra['gold_yes_no']`）：以 “Yes / That is True” 开头为 yes，以 “No / That is False” 开头为 no；110 条全部能认出（所有 `*_predicate` 类别 493 条也全部能认出）。
- `relation`（Region [0] 相对 Region [1] 的真实关系）取决于答案，依据记在 `extra['relation_source']`：
  - **yes** → 就是该类别问的关系（`answer_yes`）；
  - **no，且答案说了相反关系**（例如说明它在另一个物体前面 / 后面，或更近 / 更远；手写示意：“No, it sits in front of the other one.”）→ 取反（`answer_no_explicit_opposite`）；
  - **no，但答案只有一个否定词、没有说明**（“No.” 这类）→ 也取反，但标成 `answer_no_implied_opposite`。这一步是推断：同一批模板里凡是展开回答的 no 都给了相反关系，所以基准是按深度二分的；但严格地说 “不在后面” 不等于 “在前面”（可能深度相近）。**如果你不接受这个推断，过滤掉这 34 条即可。**
- `caption_true / caption_options` 用区域写法（“Region [0] is behind Region [1]”及其反面），没有用物体名。
- 注意 “closer to camera / less distant” 这类模板问的是离相机的远近，和 “in front of” 被当成同一个关系处理（数据集自己把它们放在同一类别里）。

### 数量

| 类别 | 标准答案 | 真实关系 | 依据 | 条数 |
|---|---|---|---|---|
| behind_predicate | yes | behind | answer_yes | 22 |
| behind_predicate | no | front | answer_no_explicit_opposite | 12 |
| behind_predicate | no | front | answer_no_implied_opposite | 21 |
| front_predicate | yes | front | answer_yes | 30 |
| front_predicate | no | behind | answer_no_explicit_opposite | 12 |
| front_predicate | no | behind | answer_no_implied_opposite | 13 |

- 按真实关系：**front 63、behind 47**（其中不依赖推断的：front 42、behind 34，共 76 条）。
- 按标准答案：yes 52、no 58。
- 带两个框：110/110；带两个 mask：110/110。
- 图像来源：hypersim 41、SUNRGBD 33、nuScenes 21、ARKitScenes 8、KITTI_object 7。
- **两个区域类名相同的有 14 条**（front_predicate 9、behind_predicate 5）：这些样本用名字指代会有歧义。

## 改写规则（设计稿，待你确认；没有对全量数据执行）

目的：把 “Region [k]” / `<mask>` 换成物体名，让不接受区域输入的普通 VLM 也能读题。实现是 `region_names()` + `substitute()` + `rewrite_record()`。

1. **名称来源**：区域 k 用 `qa_info.class[k]`，原样使用（不改拼写，如 `soundsystem`）。
2. **冠词**：一律用定冠词 **“the + 类名”**（题目指的是图里特定的那个物体）。这样 a/an 的问题不会出现；将来若有模板需要不定冠词，用 `indefinite_article()`（按首字母元音判断，带少量例外；例如 “an oven”、“an air conditioner”）。
3. **多词类名**（night stand、traffic cone、potted plant、floor mat、air conditioner、coffee maker、kitchen pan、toilet paper、fire extinguisher、fume hood）：整体原样放进去，“the night stand”。
4. **句首大写**：占位符在句首（开头，或 `. ? !` 之后）时首字母大写，“Region [0] is …” → “The lamp is …”。
5. **`<mask>` 写法**：第 k 个 `<mask>` ↔ 区域 k（上面已核对 1,406/1,406），所以两种写法改写结果相同；改写后的题目只保留一份，不带 `<image>` 前缀。
6. **一个区域 vs 两个区域**：单区域题（`height_data` / `width_data`，266 条）只换 Region [0]，用 `class[0]`。其中 **168 条的 `class` 列了 2 个名字**，第 2 个名字不对应任何区域，丢弃（依据只有上面 12 条目检里的 6 条）。
7. **两个区域类名相同**（共 123 条，其中 front/behind 14 条）：
   - 默认 `same_class="skip"`：**不改写**，状态 `ambiguous_same_class`，这些题只能用框 / mask 指代。
   - 可选 `same_class="position"`：改成 “the chair on the left / the chair on the right”，条件是 (a) 该类别不是左右类（`left_* / right_* / horizontal_distance_data / direction`，否则会泄露或干扰答案），(b) 两个框中心的水平距离 ≥ 图宽的 10%。满足条件的有 46 条（front/behind 14 条里的 10 条），其余 77 条仍不改写（57 条是左右类，20 条水平距离不足）。注意 “on the left” 是两个区域之间的相对位置，图里如果还有第三个同类物体仍然有歧义；而且这给题目加进了数据集原本没有的位置信息。**是否启用需要你决定。**
8. **复数类名**（固定名单：books、shelves、blinds、drawers、clothes、tissues、toys、plates、utensils、shoes、electronics；不靠 “以 s 结尾” 判断，bus / glass 是单数）：紧挨着占位符的 is / does / has 改成 are / do / have（“Is Region [0] bigger…” → “Are the books bigger…”；“Region [0] is …” → “The books are …”）；占位符前面是 of / from / than / and 等词时它不是主语，后面的动词不动（“The height of the books is 3 feet.”）；所有格 “Region [0]'s” → “the books'”。涉及 232 条。
9. **答案文本**：答案里的 “Region [k]” 用同一套名字替换（按下标，不按出现顺序——答案里常只提 Region [1]）。代词（it / They）不改。**建议评测时不要拿改写后的答案文本做字符串比对**，而是用规范化标签：`*_predicate` 用 yes/no（`parse_yes_no`，493/493 可解析）；`*_choice` 用被选中的区域下标；定量题用数值 + 单位。改写后的答案只用于人工阅读或 few-shot 示例。

### 20 条预览

由 `python scripts/spatialrgpt_rewrite_preview.py` 打印（固定种子，重复运行输出相同）。分层：front_predicate 4 条、behind_predicate 4 条（各含 1 条同类名）；height_data 1 条（class 有 2 个名字）、width_data 1 条；双区域定量题 3 个类别各 1 条；`*_choice` 3 个类别各 1 条；其它 `*_predicate` 4 个类别各 1 条。Q = `text_q`，M = `<mask>` 写法，A = 答案。

> **2026-11-26 更正（Xiwei）**：原先这里是 20 条真实样本的改写前后对照，含 SpatialRGPT 的题目和答案原文。SpatialRGPT 只允许评测、不得再分发，所以换成下面的**手写格式示例**（不是数据集原文）。真实的 20 条对照可以在集群上运行 `python scripts/spatialrgpt_rewrite_preview.py` 重新打印；96 条深度样本的全量改写在 `<data>/depth/spatialrgpt_derived/`，不进仓库。

| # | 情形 | 改写前（手写示例） | 改写后 | 状态 |
|---|---|---|---|---|
| 1 | 两个区域，单数 | Q: Is Region [0] nearer to us than Region [1]?<br>A: No, Region [0] sits behind Region [1]. | Q: Is the lamp nearer to us than the sofa?<br>A: No, the lamp sits behind the sofa. | ok |
| 2 | 复数类名作主语 | Q: Does Region [0] stand in front of Region [1]?<br>A: Yes. | Q: Do the shelves stand in front of the table?<br>A: Yes. | ok；复数，紧邻动词改复数 |
| 3 | 同类名 | Q: Is Region [0] nearer to us than Region [1]?（两个都是 car） | 默认（skip）：不改写<br>可选（position）：Q: Is the car on the left nearer to us than the car on the right? | ambiguous_same_class → position: ok_position |
| 4 | 单区域、句首大写 | Q: Roughly how wide is Region [0]?<br>A: Region [0] spans about 20 inches. | Q: Roughly how wide is the box?<br>A: The box spans about 20 inches. | ok |
| 5 | 答案只提 Region [1] | Q: Standing at Region [0], where is Region [1]?<br>A: Region [1] is around 2 o'clock. | Q: Standing at the bag, where is the chair?<br>A: The chair is around 2 o'clock. | ok |
| 6 | 答案含代词 | Q: How far apart horizontally are Region [0] and Region [1]?<br>A: It is about 2 meters. | Q: How far apart horizontally are the picture and the chair?<br>A: It is about 2 meters. | ok；代词不改 |

### 规则处理不了的情况

| 情况 | 影响条数 | 现在的行为 |
|---|---|---|
| 两个区域类名相同，默认规则不改写 | 123（front/behind 14）；启用 position 后仍有 77（front/behind 4） | 状态 `ambiguous_same_class`，无改写文本。按类别：above_choice 2、above_predicate 2、behind_predicate 5、below_predicate 1、big_predicate 3、direction 19、distance_data 24、front_predicate 9、horizontal_distance_data 19、left_choice 5、left_predicate 6、right_choice 5、right_predicate 3、short_choice 3、short_predicate 1、small_predicate 6、tall_choice 1、tall_predicate 2、thin_predicate 1、vertical_distance_data 5、wide_predicate 1 |
| 类名不同但实际近义，名字仍分不清（如第 20 条 person / pedestrian） | person + pedestrian 组合 6 条；其它近义组合（car / van / truck、picture / painting 等）没有统计 | 照常改写，未标记 |
| 图里有不在题目里的同类物体（“the chair” 指哪一把） | **无法统计**：数据集只给了被提问区域的标注，没有整图的物体清单 | 照常改写，未标记。这是把区域题改成名称题的根本局限 |
| 单区域题的 `class` 有 2 个名字，假定区域对应 class[0] | 168 | 用 class[0]，备注里标出；依据是 6 条目检 |
| 复数类名作主语，而答案用单数代词 “it”（“Are the blinds positioned behind the bed?” — “Yes, it is.”，见第 7 条） | 20（含复数类名且答案有 it 的共 24） | 代词不改，答案在语法上不一致 |
| 问句用 “Who” 指物体（“Who is higher up, the television or the chair?”） | 57 | 原样保留 |
| 答案不提区域（“No.”、“5.5 feet”） | 双区域题里 201 条 | 不需要改；正常 |
| 名单之外的新类名 | 当前数据 0 条（88 个类名都看过） | 按单数处理 |
| 类名本身不准或很笼统（如 stationery、electronics、目检里把投影幕标成 monitor） | 没有统计 | 原样使用 |

## 需要你决定的

1. 改写规则是否按上面执行；同类名的 123 条用 skip 还是 position。
2. front/behind 里 34 条 “只回答 No” 的样本，是否接受取反得到的关系。
3. 定位真值用框（偏松）还是 mask。
4. `spatialrgpt_depth` 是否注册进 `DATASETS`，以及 dev / test 怎么划分（110 条，两个关系 63 / 47）。
