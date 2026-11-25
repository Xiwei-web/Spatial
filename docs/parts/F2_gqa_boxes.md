# F2：GQA-spatial 的框（来自 GQA scene graph）

> **更新（2026-11-25 夜）**：划分已改为按图像（`splits/gqa_images_v2.json`），25 张因 coco_id 跟随 COCO 统一划分；旧文件作废、读取报错。

> **更新（2026-11-25 晚）**：29 处同义词按用户决定不补，保持 `box_status` 标记为无框。

日期 2026-11-25。代码：`src/data/gqa_boxes.py`（匹配规则）、`scripts/build_gqa_boxes.py`（生成 + 统计 + 拼图）、`src/data/loaders.py`（接入）、`tests/test_gqa_spatialrgpt.py`。没有跑任何模型。

## 下载

| 项 | 内容 |
|---|---|
| 官方页面 | https://cs.stanford.edu/people/dorarad/gqa/download.html ，“Download Scene Graphs” 按钮的链接就是 https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip ，页面标注 “ver 1.1 / 42.7MB” |
| 体积 | HTTP HEAD `Content-Length: 44824862`（44.8 MB，远小于 5 GB 上限）；`Last-Modified: Sun, 03 Feb 2019` |
| sha256（zip） | `59f6a3f6ac5227bac6cc615508e542dd546e309c7dbbdb666de05e42d7c51989` |
| 存放 | 压缩包 `data/_archives/gqa_sceneGraphs.zip`；解到独立新目录 `data/gqa/sceneGraphs/`（`train_sceneGraphs.json` 317,399,800 B、`val_sceneGraphs.json` 44,830,665 B、`readme.txt`）；逐文件 sha256 在 `data/gqa/SHA256SUMS.sceneGraphs` |
| 许可 | 官方站点（index / about / download 三页）**没有文字形式的许可声明**。下载页页脚只有一个 Creative Commons 图标，原文是 `<a ... href="https://creativecommons.org/licenses/by/4.0/"><img src="img/cc.png" alt="" title=""/></a>`，即链接到 CC BY 4.0。压缩包里的 `readme.txt` 也没有许可条款。 |
| 下载命令 | `bash scripts/download_data.sh gqa_sg`（新目标，不并入 `all`） |

`data/_archives/SHA256SUMS` 是之前就有的文件，按规则没有改动，所以新压缩包的 sha256 只记在这里和 `manifests/gqa_spatial_boxes.json` 的 `meta` 里。

## 匹配规则

对每个样本，取 caption 解析出的 target / reference 短语，在该图像的 scene graph 物体里找候选：

1. **R1 exact**：物体 `name` 与短语完全相同（两边只做小写、去首尾空格）。
2. **R2 attr+name**：只有 R1 一个候选都没有时才用。把短语切成 “前缀词 + 后缀”，后缀等于物体 `name`，且前缀每个词都在该物体的 `attributes` 里（“green trees” → name `trees` 且有属性 `green`；“wood coffee table” → name `coffee table` 且有属性 `wood`）。所有切分位置的候选取并集。
3. 恰好 1 个候选 → `unique`，给框；0 个 → `none`；多于 1 个 → `multiple`；图像不在 scene graph 里 → `image-not-in-scene-graphs`。后三种都不给框，样本保留。
4. **不用 left/right/top/bottom 标签在多个候选里挑。**
5. 没有加单复数和同义词归一化：实测 R1 + R2 之后剩下的 `none`，用单复数变形能救回的是 **0** 条（剩下的都是同义词或截断，见下），所以不加。

框：scene graph 的 `x, y, w, h` 换成 `[x1, y1, x2, y2]`，原图像素。有 32 个唯一匹配的框比实际图像多出 1 像素（例如 x2 = 501 而图宽 500），已裁到图像范围内，原值保留在 `bbox_raw`。

## 结果

全部 1,208 张图都在 `val_sceneGraphs.json` 里（`train` 文件没用到，`image-not-in-scene-graphs` = 0）。

| 子集 | 对象 | 样本数 | unique | 其中 R1 exact | 其中 R2 attr+name | none | multiple | 唯一匹配率 |
|---|---|---|---|---|---|---|---|---|
| gqa_spatial_one | target | 1,160 | 1,121 | 1,107 | 14 | 24 | 15 | 96.6% |
| gqa_spatial_two | target | 291 | 290 | 231 | 59 | 1 | 0 | 99.7% |
| gqa_spatial_two | reference | 291 | 287 | 283 | 4 | 4 | 0 | 98.6% |
| gqa_spatial_two | target 和 reference 都唯一 | 291 | 286 | — | — | — | — | 98.3% |

两个子集合计：1407 个样本所需的框全部到位，44 个不全（one 39 个，two 5 个）。

### 失败原因

| 原因 | 条数 | 涉及的短语（条数） |
|---|---|---|
| `none`：scene graph 用的是另一个名字（GQA 问题里的同义词/别名） | 29（one 24，two 5） | one：tennis racket 8、mobile phone 5、flower bouquet 3、meter 2、table cloth 2、bed sheet 1、palm 1、remote 1、surf board 1；two：target dish washer 1，reference palm 1、remote 1、surf board 1、tennis racket 1 |
| `multiple`：同名物体不止一个（都是 R1 exact） | 15（全在 one） | chair 2、computer 2、palm tree 2（各 5 个候选）、boat 1（3 个候选）、car 1、hot dog 1、person 1、remote control 1、spoon 1、surfboard 1、tomato 1、umbrellas 1 |
| `image-not-in-scene-graphs` | 0 | — |

`none` 的例子里 scene graph 实际的名字是 `cell phone`（对 “mobile phone”）、`parking meter`（对 “meter”）、`racket`（对 “tennis racket”）、`bouquet`、`sheet` 等。这些要靠同义词表才能对上，没有做。

### one-object 的位置一致性（只统计，没有用来挑框）

- 唯一匹配且关系是 left/right/top/bottom 的 1,119 个样本里，框中心落在 caption 所说一侧的有 **1,119 个（100%）**（另外 2 个唯一匹配的关系是 “behind”，不适用）。
- `multiple` 的 15 个样本：如果用标签去挑，“恰好只有一个候选在所说一侧” 的有 **0 个**（15 个样本的全部候选都在所说的那一侧），也就是说位置标签一个也消不了歧。

### 靠属性才唯一的情况

R2 命中且同名物体不止一个（靠属性区分）的有 4 条：one `00171_2364335` “white refrigerator”（同名 2 个）、one `00408_498126` “blue towel”（同名 3 个）、two `00081_2364335` “white refrigerator”、two `00286_2414948` “silk blanket”（同名 2 个）。这些给了框，记录里有 `n_same_name`，想更保守可以按它过滤。注意：另一个同名物体没标该属性不代表它没有这个属性。

### 图像尺寸核对

把 scene graph 里的 `width / height` 和实际文件（`data/whatsup/vg_images/<id>.jpg`）逐张对比，范围是全部 1208 张涉及的图：1207 张一致，1 张不一致——`498187`：文件 1280×960，scene graph 写 1024×768。检查该图的 12 个物体：框坐标最大到 x = 1278、y = 771，超出了 1024×768，把框画到 1280×960 的文件上位置正确（火车的框落在火车上）。结论：这张图的框本身就在我们文件的像素空间里，只是 scene graph 的 `width/height` 字段不对，**没有做缩放**。只有样本 one `00533_498187` 用到它。

## 输出与接入

- `data/gqa/gqa_spatial_boxes.json`（651,349 B，sha256 `8134ce36fdd8451bc4805079a6f3d22504f0af70ceb36643a6635e22107abd0a`），仓库副本 `manifests/gqa_spatial_boxes.json`（内容相同）。结构：`samples[<数据集名>][<sample_id>] = {image_id, relation, target: {name, status, rule, n_candidates, bbox, object_id, candidate_boxes, [bbox_raw], [n_same_name]}, reference: 同上或 null, [side_consistent], [n_candidates_on_stated_side]}`。按数据集名分两层是因为 one / two 的 sample_id 会重号。重跑两次文件逐字节相同。
- `manifests/gqa_spatial_boxes_stats.json`：本文所有数字和例子的来源。
- 加载器：`load_dataset('gqa_spatial_one' / 'gqa_spatial_two')` 现在带 `bbox_target` / `bbox_reference`（唯一匹配时），并且每个样本都有 `extra['box_status'] = {'target': 状态, 'reference': 状态或 None}`；没匹配上的样本保留，框为 `None`。加载器优先读仓库副本，其次读数据目录；两处都没有时状态为 `boxes-file-missing`，其它行为不变。`Sample` 字段没变，mask 仍为 `None`（scene graph 没有 mask）。实测：one 1,160 条里 1,121 条有 target 框；two 291 条里 290 条有 target 框、287 条有 reference 框。
- 划分文件（`splits/gqa_spatial_*.json`）没有动；有框子集在 dev / test 里各有多少没有统计。

## 例子

10 个随机的已匹配样本（种子固定）：

| 子集 | sample_id | 关系 | 匹配 |
|---|---|---|---|
| one | `00512_2369996` | left | target `television`：unique（exact），候选 1，框 [20, 96, 129, 212] |
| one | `01060_2396707` | bottom | target `salad`：unique（exact），候选 1，框 [206, 380, 305, 464] |
| two | `00044_2372090` | right | target `toilet`：unique（exact），候选 1，框 [219, 364, 333, 500]；reference `cabinet`：unique（exact），候选 1，框 [127, 337, 221, 453] |
| one | `00056_2318945` | right | target `bus`：unique（exact），候选 1，框 [342, 104, 453, 156] |
| one | `00329_2334386` | left | target `napkin`：unique（exact），候选 1，框 [0, 231, 91, 309] |
| one | `00208_2391756` | bottom | target `banana`：unique（exact），候选 1，框 [257, 237, 353, 309] |
| one | `00150_2366645` | left | target `car`：unique（exact），候选 1，框 [3, 107, 129, 172] |
| one | `00676_2352191` | bottom | target `pizza`：unique（exact），候选 1，框 [72, 370, 269, 418] |
| one | `01107_2361429` | right | target `screen`：unique（exact），候选 1，框 [294, 42, 450, 166] |
| one | `00591_2411015` | top | target `fence`：unique（exact），候选 1，框 [181, 13, 301, 101] |

10 个随机的未匹配样本：

| 子集 | sample_id | 关系 | 匹配 |
|---|---|---|---|
| one | `00864_2411367` | top | target `car`：multiple（exact），候选 2 |
| one | `00230_2413136` | right | target `remote control`：multiple（exact），候选 2 |
| one | `00801_2413136` | right | target `remote`：none，候选 0 |
| two | `00185_4709` | behind | target `building`：unique（exact），候选 1，框 [0, 46, 349, 157]；reference `palm`：none，候选 0 |
| one | `00873_2329965` | left | target `flower bouquet`：none，候选 0 |
| one | `00229_2388377` | left | target `table cloth`：none，候选 0 |
| one | `00576_2333049` | left | target `surf board`：none，候选 0 |
| one | `01016_2398329` | top | target `tennis racket`：none，候选 0 |
| one | `00843_2407703` | bottom | target `surfboard`：multiple（exact），候选 2 |
| one | `00684_2327809` | top | target `chair`：multiple（exact），候选 2 |

## 拼图

`<outputs>/sanity/gqa_boxes.png`（即 `/nfs-stor/xiwei.liu/interpretability/outputs/sanity/gqa_boxes.png`）：20 个随机已匹配样本，红框 = target，蓝框 = reference。我看了一遍，20 个里框都落在所说的物体上；这只是 20 个样本的目检，不是对全部 1,407 个的验证。

## 没做的 / 需要你决定的

- 29 条 `none` 要不要用一张人工同义词表（约 10 个短语）救回来。
- 15 条 `multiple` 没有可靠的自动办法（位置标签消不了歧）。
- scene graph 的框是众包标注，松紧不一；没有评估框的质量。
