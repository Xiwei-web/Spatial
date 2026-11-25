# F3 COCO-Pairs：构建规则与分布

本文件由 `scripts/build_coco_pairs.py` 自动生成，所有数字都是脚本实际算出来的；没有任何模型推理、没有准确率。
扫描图像数：5000（COCO val2017 全部图像，未下采样）。种子 20261125。

- 数据：`<data>/coco_pairs/pairs.jsonl`（每行一条有向 pair；sha256 `fd67adbfeb259c27390e37ddc679a2f01bddc24e7e92a828e22ab77e231b5a84`，5124 行，7479319 字节）
- 划分：`splits/coco_pairs_v2.json`（只存图像 id；继承 COCO-spatial 与 COCO-Pairs 的统一图像划分 `splits/coco_images_v2.json`）
- 统计：`manifests/coco_pairs_summary.json`
- 加载：`from src.data.coco_pairs import load_coco_pairs; load_coco_pairs(split=None|'dev'|'test')`

## 1. 规则（含需要解释的地方）

候选物体（依次施加）：

1. `iscrowd == 0`；
2. 类别唯一：该物体的类别在这张图的**全部**标注里恰好出现 1 次（计数时把 `iscrowd=1` 的标注也算进去，所以“1 个 person + 1 块 person crowd”不算唯一）；
3. 面积：COCO 标注的 `area` 字段 / (width×height) ∈ [1%, 40%]（闭区间）。用的是 COCO `area` 字段而不是解码像素数，两者的差异见下面“面积核对”；
4. 图像短边 ≥ 300 px。

配对与标签：

- 配对：同一张图里所有无序候选对；每对保留两个方向（两条记录，twin_id 互指）。`pair_id = <12 位 image_id>_<target ann_id>_<reference ann_id>`。
- 质心：解码后的二值 mask 的像素下标均值 + 0.5（像素中心，与 COCO bbox 连续坐标一致）。
- 关系：|dx|>|dy| → left/right（target 质心 x 更小 → left）；否则 above/below（target 质心 y 更小 → above）。
- 平局：|dx|==|dy| 归 above/below；质心完全重合时 ann_id 较小的一方记为 above（保证 twin 相反），并置 centroid_tie=true。实际出现次数：质心完全重合 0 对；|dx|==|dy|（6 位小数下）0 对。
- `norm_gap`：质心欧氏距离 / 两个 bbox 沿关系轴长度的均值（水平轴用宽，竖直轴用高）。“质心距离”按字面取欧氏距离；另存 `norm_gap_axis`（质心差在关系轴上的分量 / 同一分母（补充量））。本文分箱默认用 `norm_gap`。
- `bbox_iou`：真值框 = mask 外接框（[x1,y1,x2,y2]，用户 2026-11-25）；COCO 自带框及按它算的量保留在 *_detbox 字段 后计算；`mask_iou`：pycocotools 对两个实例 mask 的 RLE 求 IoU。
- `containment_target_in_reference` / `containment_reference_in_target`：bbox 交集面积 / 自身 bbox 面积；两个方向都记。
- `size_ratio`：COCO area(target) / COCO area(reference)。
- `n_other_instances`：该图全部标注数（含 crowd）- 2；n_other_instances_noncrowd 只数 iscrowd=0。
- `proj_overlap`：两个 bbox 在关系轴上的投影区间重叠长度 > 0（端点相接不算）；另存 `proj_overlap_frac` = 重叠长度 / 较短的那个投影长度。重叠表示关系本身有歧义。
- **没有过滤任何难例**：重叠、包含、投影重叠、极近的 pair 全部保留，只打标记。
- `caption_true` / `caption_options` 不在 jsonl 里，由加载器按 COCO-spatial 两物体子集的模板生成（`A photo of a X to the left of a Y` / `A photo of a X above a Y`；第二个选项是相反关系）。

## 2. 候选物体漏斗

| 步骤 | 剩余标注数 |
|---|---|
| 全部标注（含 crowd） | 36781 |
| a) iscrowd=0 | 36335 |
| b) 类别在图中唯一 | 8548 |
| c) 面积占比 1%–40% | 5050 |
| d) 图像短边 ≥ 300（= 候选物体） | 4966 |

| 量 | 数值 |
|---|---|
| 候选物体数 | 4966 |
| 至少 1 个候选的图像数 | 3139 |
| ≥2 个候选的图像数（= 有 pair 的图像） | 1274 |
| 进入 pair 的物体数 | 3101 |
| 无向 pair 数 | 2562 |
| 有向 pair 数（jsonl 行数） | 5124 |
| 候选物体的不同类别数 | 80 |
| 进入 pair 的不同类别数 | 80 |

每张图的候选物体个数：

| 候选数 | 图像数 |
|---|---|
| 0 | 1861 |
| 1 | 1865 |
| 2 | 870 |
| 3 | 285 |
| 4 | 92 |
| 5 | 24 |
| 6 | 3 |

### 面积核对（COCO `area` 对比 pycocotools 解码像素数）

对**全部** 4966 个候选物体都解码了 mask（不是抽样）。相对差 = (解码像素数 − COCO area) / COCO area。

| 统计量 | 数值 |
|---|---|
| 相对差 最小 / p5 / 中位 / p95 / 最大 | -0.4428 / -0.0022 / 0.0001 / 0.0024 / 0.0169 |
| abs(相对差) 均值 | 0.001172 |
| abs(相对差) 最大 | 0.4428 |
| abs(相对差) > 1% 的物体数 | 17 |
| abs(相对差) > 5% 的物体数 | 4 |
| 若改用解码面积，会落到 1%–40% 之外的候选数 | 4 |

筛选阈值和 `size_ratio` 用 COCO `area`；`mask_iou` 与质心用解码后的 mask；每个物体同时记录 `area` 与 `area_decoded`。

## 3. 关系分布（有向 pair）

| 关系 | 有向 pair 数 | 占比 | 其中投影重叠 | 投影重叠占比 |
|---|---|---|---|---|
| left | 1506 | 29.4% | 738 | 49.0% |
| right | 1506 | 29.4% | 738 | 49.0% |
| above | 1056 | 20.6% | 708 | 67.0% |
| below | 1056 | 20.6% | 708 | 67.0% |

无向 pair 按关系轴：horizontal 1506，vertical 1056。因为两个方向都保留，left 与 right、above 与 below 的数量必然相等。

## 4. 归一化间距分箱

| norm_gap 区间 | 有向 | 无向 | 占比 | 其中投影重叠（有向） | 按 norm_gap_axis 分箱（有向） |
|---|---|---|---|---|---|
| <0.5 | 778 | 389 | 15.2% | 778 | 938 |
| 0.5-1 | 1658 | 829 | 32.4% | 1628 | 1788 |
| 1-1.5 | 1136 | 568 | 22.2% | 480 | 1050 |
| 1.5-2 | 596 | 298 | 11.6% | 6 | 538 |
| 2-3 | 548 | 274 | 10.7% | 0 | 470 |
| 3-5 | 320 | 160 | 6.2% | 0 | 278 |
| >=5 | 88 | 44 | 1.7% | 0 | 62 |

区间左闭右开（例如 `0.5-1` 表示 0.5 ≤ gap < 1）。

## 5. 拥挤度分箱（图中其他实例数）

| 其他实例数 | 有向（全部标注） | 无向 | 占比 | 有向（只数非 crowd） | 图像数（全部标注） |
|---|---|---|---|---|---|
| 0 | 632 | 316 | 12.3% | 632 | 316 |
| 1-2 | 1200 | 600 | 23.4% | 1200 | 315 |
| 3-5 | 1258 | 629 | 24.6% | 1258 | 238 |
| 6-10 | 1154 | 577 | 22.5% | 1154 | 209 |
| 11-20 | 724 | 362 | 14.1% | 740 | 156 |
| >20 | 156 | 78 | 3.0% | 140 | 40 |

## 6. 关系 × 间距 交叉表（有向 pair）

| 关系 \ norm_gap | <0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-5 | >=5 | 合计 |
|---|---|---|---|---|---|---|---|---|
| left | 183 | 435 | 346 | 196 | 178 | 129 | 39 | 1506 |
| right | 183 | 435 | 346 | 196 | 178 | 129 | 39 | 1506 |
| above | 206 | 394 | 222 | 102 | 96 | 31 | 5 | 1056 |
| below | 206 | 394 | 222 | 102 | 96 | 31 | 5 | 1056 |
| 合计 | 778 | 1658 | 1136 | 596 | 548 | 320 | 88 | 5124 |

### 关系 × 拥挤度（有向 pair）

| 关系 \ 其他实例数 | 0 | 1-2 | 3-5 | 6-10 | 11-20 | >20 | 合计 |
|---|---|---|---|---|---|---|---|
| left | 129 | 352 | 378 | 353 | 237 | 57 | 1506 |
| right | 129 | 352 | 378 | 353 | 237 | 57 | 1506 |
| above | 187 | 248 | 251 | 224 | 125 | 21 | 1056 |
| below | 187 | 248 | 251 | 224 | 125 | 21 | 1056 |
| 合计 | 632 | 1200 | 1258 | 1154 | 724 | 156 | 5124 |

### 间距 × 拥挤度（有向 pair）

| norm_gap \ 其他实例数 | 0 | 1-2 | 3-5 | 6-10 | 11-20 | >20 | 合计 |
|---|---|---|---|---|---|---|---|
| <0.5 | 150 | 220 | 218 | 132 | 50 | 8 | 778 |
| 0.5-1 | 326 | 456 | 374 | 282 | 176 | 44 | 1658 |
| 1-1.5 | 104 | 290 | 298 | 264 | 150 | 30 | 1136 |
| 1.5-2 | 26 | 106 | 166 | 160 | 114 | 24 | 596 |
| 2-3 | 18 | 74 | 112 | 184 | 136 | 24 | 548 |
| 3-5 | 8 | 42 | 70 | 104 | 72 | 24 | 320 |
| >=5 | 0 | 12 | 20 | 28 | 26 | 2 | 88 |

## 7. 歧义与重叠

| 条件 | 有向 pair 数 | 无向 pair 数 | 占比 |
|---|---|---|---|
| 关系轴上投影重叠（proj_overlap） | 2892 | 1446 | 56.4% |
| proj_overlap_frac ≥ 0.5 | 1758 | 879 | 34.3% |
| proj_overlap_frac = 1（一个投影完全落在另一个里） | 808 | 404 | 15.8% |
| bbox IoU > 0 | 2796 | 1398 | 54.6% |
| bbox IoU ≥ 0.5 | 182 | 91 | 3.6% |
| mask IoU > 0 | 2070 | 1035 | 40.4% |
| target bbox ≥90% 落在 reference bbox 内（有向量） | 384 | - | 7.5% |
| target bbox 100% 落在 reference bbox 内（有向量） | 236 | - | 4.6% |

### 连续量的分位数（对称量按无向 pair 统计）

| 量 | 最小 | p5 | p25 | 中位 | p75 | p95 | 最大 | 均值 |
|---|---|---|---|---|---|---|---|---|
| norm_gap | 0.001984 | 0.2224 | 0.694 | 1.047 | 1.695 | 3.528 | 8.52 | 1.358 |
| norm_gap_axis | 0.001816 | 0.2006 | 0.6227 | 0.9515 | 1.54 | 3.292 | 8.332 | 1.245 |
| centroid_dist | 0.6447 | 53.57 | 140.9 | 211 | 292.2 | 433 | 627.7 | 222.5 |
| bbox_iou | 0 | 0 | 0 | 0.01256 | 0.1111 | 0.4299 | 0.9801 | 0.09097 |
| mask_iou | 0 | 0 | 0 | 0 | 0.002975 | 0.1113 | 0.9686 | 0.02485 |
| proj_overlap_frac | 0 | 0 | 0 | 0.1254 | 0.75 | 1 | 1 | 0.3488 |
| n_other_instances | 0 | 0 | 1 | 4 | 8 | 19 | 38 | 5.866 |
| n_other_instances_noncrowd | 0 | 0 | 1 | 4 | 8 | 18 | 36 | 5.827 |
| size_ratio（有向） | 0.02707 | 0.09482 | 0.3548 | 1 | 2.818 | 10.55 | 36.94 | 2.563 |
| containment_target_in_reference（有向） | 0 | 0 | 0 | 0.02469 | 0.2772 | 0.9922 | 1 | 0.1966 |

## 8. 类别

进入 pair 的不同类别数：80（COCO 共 80 类）；不同的无序类别组合数：675。

出现最多的 20 个类别（按进入 pair 的物体数排序）：

| 类别 | 进入 pair 的物体数 | 作为 target 的有向 pair 数 | 候选物体数 |
|---|---|---|---|
| person | 519 | 747 | 794 |
| dining table | 190 | 366 | 259 |
| chair | 143 | 270 | 168 |
| tv | 107 | 201 | 126 |
| cup | 94 | 182 | 109 |
| cat | 90 | 153 | 144 |
| laptop | 90 | 171 | 113 |
| couch | 83 | 149 | 118 |
| bowl | 80 | 165 | 92 |
| dog | 71 | 110 | 110 |
| sink | 68 | 101 | 96 |
| bed | 64 | 104 | 88 |
| refrigerator | 64 | 114 | 76 |
| bench | 57 | 77 | 104 |
| oven | 57 | 106 | 64 |
| toilet | 56 | 73 | 116 |
| bottle | 55 | 111 | 67 |
| potted plant | 51 | 101 | 61 |
| umbrella | 48 | 71 | 82 |
| keyboard | 47 | 89 | 57 |

出现最多的 20 个类别组合（无向）：

| 类别组合 | 无向 pair 数 |
|---|---|
| dining table + person | 43 |
| person + tennis racket | 40 |
| person + surfboard | 36 |
| sink + toilet | 35 |
| chair + person | 33 |
| person + skateboard | 30 |
| chair + dining table | 29 |
| couch + tv | 26 |
| laptop + person | 26 |
| oven + refrigerator | 26 |
| bed + person | 25 |
| person + tv | 24 |
| bench + person | 23 |
| chair + tv | 23 |
| cup + dining table | 23 |
| person + umbrella | 23 |
| cake + dining table | 22 |
| couch + dining table | 22 |
| chair + couch | 21 |
| couch + person | 20 |

## 9. 与 COCO-spatial 的图像重叠

COCO-spatial 用到的不同图像：one 1451 张，two 295 张，并集 1451 张。

|  | COCO-Pairs 图像数 | 有向 pair 数 |
|---|---|---|
| 全部 | 1274 | 5124 |
| 也在 coco_spatial_one | 834 | 3916 |
| 也在 coco_spatial_two | 284 | 1896 |
| 在 one 或 two | 834 | 3916 |
| 两者都不在 | 440 | 1208 |

每条记录带 `in_coco_spatial_one` / `in_coco_spatial_two` 两个标记；没有因此剔除任何图像。

## 10. dev / test 划分

按**图像**划分，与 COCO-spatial 统一分配（种子 20261125，`scripts/make_joint_coco_split.py`，`splits/coco_images_v2.json`）。层 = 属于哪些数据集（one / two / pairs）| 主导关系轴（H / V / T）| 候选物体数（2 / 3 / 4+）；下表也列出只属于 COCO-spatial 的层。

| 层（主导轴/候选数） | 图像数 | dev | test |
|---|---|---|---|
| -|-|pairs|H|n2 | 179 | 89 | 90 |
| -|-|pairs|H|n3 | 34 | 17 | 17 |
| -|-|pairs|H|n4+ | 7 | 4 | 3 |
| -|-|pairs|T|n4+ | 1 | 0 | 1 |
| -|-|pairs|V|n2 | 198 | 99 | 99 |
| -|-|pairs|V|n3 | 19 | 10 | 9 |
| -|-|pairs|V|n4+ | 2 | 1 | 1 |
| one|-|-|-|- | 606 | 303 | 303 |
| one|-|pairs|H|n2 | 208 | 104 | 104 |
| one|-|pairs|H|n3 | 87 | 43 | 44 |
| one|-|pairs|H|n4+ | 20 | 10 | 10 |
| one|-|pairs|T|n4+ | 4 | 2 | 2 |
| one|-|pairs|V|n2 | 175 | 88 | 87 |
| one|-|pairs|V|n3 | 45 | 22 | 23 |
| one|-|pairs|V|n4+ | 11 | 6 | 5 |
| one|two|-|-|- | 11 | 5 | 6 |
| one|two|pairs|H|n2 | 82 | 41 | 41 |
| one|two|pairs|H|n3 | 73 | 37 | 36 |
| one|two|pairs|H|n4+ | 45 | 22 | 23 |
| one|two|pairs|T|n4+ | 12 | 6 | 6 |
| one|two|pairs|V|n2 | 28 | 14 | 14 |
| one|two|pairs|V|n3 | 27 | 14 | 13 |
| one|two|pairs|V|n4+ | 17 | 8 | 9 |

|  | dev | test |
|---|---|---|
| 图像数 | 637 | 637 |
| 无向 pair | 1275 | 1287 |
| 有向 pair | 2550 | 2574 |
| 关系 left | 757 | 749 |
| 关系 right | 757 | 749 |
| 关系 above | 518 | 538 |
| 关系 below | 518 | 538 |
| norm_gap <0.5 | 384 | 394 |
| norm_gap 0.5-1 | 852 | 806 |
| norm_gap 1-1.5 | 552 | 584 |
| norm_gap 1.5-2 | 306 | 290 |
| norm_gap 2-3 | 252 | 296 |
| norm_gap 3-5 | 162 | 158 |
| norm_gap >=5 | 42 | 46 |
| 其他实例数 0 | 302 | 330 |
| 其他实例数 1-2 | 614 | 586 |
| 其他实例数 3-5 | 646 | 612 |
| 其他实例数 6-10 | 600 | 554 |
| 其他实例数 11-20 | 328 | 396 |
| 其他实例数 >20 | 60 | 96 |
| 投影重叠 | 1474 | 1418 |
| bbox IoU > 0 | 1424 | 1372 |
| mask IoU > 0 | 1058 | 1012 |
| 也在 coco_spatial_one | 1952 | 1964 |
| 也在 coco_spatial_two | 946 | 950 |

（除前两行外，单位都是有向 pair。）

COCO-Pairs 与 COCO-spatial 的划分是按图像统一分配的（`splits/coco_images_v2.json`），同一张图在两个数据集里同侧。
本文所有框几何量按 mask 外接框计算；按 COCO 自带框的旧值在记录的 `*_detbox` 字段，变化逐条列在 `manifests/coco_pairs_geometry_change.csv`。
