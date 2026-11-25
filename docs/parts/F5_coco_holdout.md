# F5 COCO holdout 图像池

由 `scripts/build_coco_holdout.py` 自动生成。输出 `splits/coco_holdout.json`（排序后的图像 id 与文件名）。

用途：只用来算与评测无关的统计量（例如视觉 embedding 的均值）。这些图像既不在 COCO-spatial（one / two），
也不在任何一条 COCO-Pairs 记录里（dev、test 都排除）。种子 20261125；
抽样方式：把剩余图像 id 排序后用 `random.Random("20261125:coco_holdout").sample` 抽取，再排序。
排除时所依据的 pairs.jsonl 的 sha256：`1b99821b4ce4ed5deb509189b505a540daaddc0cb42354036ed129d9924d5c58`。

| 项 | 图像数 |
|---|---|
| COCO val2017 全部图像 | 5000 |
| 被 coco_spatial_one 排除 | 1451 |
| 被 coco_spatial_two 排除 | 295 |
| 被 COCO-spatial（one ∪ two）排除 | 1451 |
| 被 COCO-Pairs（dev ∪ test）排除 | 1274 |
| 其中只被 COCO-Pairs 排除 | 440 |
| 其中只被 COCO-spatial 排除 | 617 |
| 其中两者都排除 | 834 |
| 排除合计（并集） | 1891 |
| 剩余可用 | 3109 |
| 选入 holdout | 1000 |

各来源之间有重叠，所以分项之和大于并集。
