# F4 参数化合成场景生成器（试跑 trial_v0）

> **更新（2026-11-25 晚）**：用户决定：抠图库只用 COCO；暂不扩量。抠图划分改为继承所在图像（`splits/synth_cutouts_v2.json`，dev 653 / test 680），同一张图的抠图不再跨侧；旧的 `splits/synth_cutouts.json` 和库内嵌的 split 字段作废。trial_v0 用的是旧 dev 抠图，其中 326 个现在属于 test，所以 **trial_v0 只能用于检查生成器，不能用于任何评测**。按图像继承后 `book` 类在 dev 没有抠图（10 个都在 test），其余 62 类两侧都有。

状态：切图库已建好（**只含 COCO**，What's Up 自动标注在构建时还没产出），500 个基础场景 + 各 2 张孪生图 = 1,500 张图已生成。
没有生成正式全量集，没有跑任何模型。所有数字来自 `manifests/synth_cutouts.json` 和 `manifests/synth_trial_v0.json`
（Slurm 作业 435245，cscc-cpu-p，8 核，节点 cn-14，已正常结束）。

## 1. 文件

| 文件 | 内容 |
|---|---|
| `configs/synth/trial_v0.yaml` | 全部设置：因子水平、抖动、摆放、遮挡下限、易混类别组、切图筛选阈值 |
| `src/synth/cutouts.py` | 切图筛选 / 裁剪 / 缩放 / 截断 / 划分 / 读取 |
| `src/synth/backgrounds.py` | 程序化背景（抽参数与画图分开） |
| `src/synth/generator.py` | 设计表 + 场景与孪生图生成（文件头有全部精确定义） |
| `scripts/build_cutout_library.py` | 建切图库 |
| `scripts/generate_synth.py` | 生成场景、统计、确定性检查、写 manifest |
| `scripts/synth_contact_sheet.py` | 人工检查用联系表 |
| `tests/test_synth.py` | 18 个单元测试（内存假切图库，不依赖数据；作业里 18 个全过） |
| `splits/synth_cutouts.json` | 切图的 dev / test 划分（进 Git） |
| `manifests/synth_cutouts.json`、`manifests/synth_trial_v0.json` | 统计 + sha256 + 路径键（进 Git） |

不进仓库的输出（路径都经 `src.paths`）：

- `<data>/synth/cutouts/`：`library.jsonl`（sha256 `b545938e…4367`，0.86 MB）+ `png/` 1,333 个 RGBA 小图（共 95 MB）
- `<data>/synth/trial_v0/`：`images/` 1,500 张 PNG、`metadata.jsonl`（7.4 MB，sha256 `aaaf51da…37a1`）、`timing.json`
- `<outputs>/synth_review/trial_v0/`：`sheet_000.jpg` … `sheet_007.jpg` + `index.json`

重现：

```bash
scripts/submit.sh cpu --time=02:00:00 --name=f4_synth bash -c 'source scripts/env.sh tf5 && \
  python scripts/build_cutout_library.py [--force] && \
  python scripts/generate_synth.py --config configs/synth/trial_v0.yaml --workers 8 --verify 20 && \
  python scripts/synth_contact_sheet.py --n-sheets 8'
python scripts/generate_synth.py --only-index 123 --out-dir <临时目录>     # 单独重生成一个场景
```

## 2. 切图库

### 2.1 实际纳入的来源

| 来源 | 是否纳入 | 说明 |
|---|---|---|
| `coco_val2017` | 是 | 人工多边形 mask |
| `whatsup_auto_v1` | **否** | 构建时 `<data>/whatsup_annotations/auto_v1/annotations.jsonl` 不存在（标注作业当时还在跑，目录里只有 `raw.jsonl`）。代码路径已实现并用合成记录做了单元测试；该文件产生后跑 `build_cutout_library.py --force` 即可纳入。**注意重建会改变划分和之后生成的场景**，trial_v0 的图要跟着重生成。这份标注是自动标注、未经人工核对，metadata 里一律标 `source: whatsup_auto_v1`，`human_verified: false`。 |

`splits/coco_holdout.json` 在构建时**已存在，排除已生效**（1,000 张图，sha256 `d78618e6…1f4b`，涉及 7,391 个实例）。
`splits/coco_pairs.json` 也已存在（sha256 `f581a8af…f5c3`），用于记录归属。如果这两个文件之后被重做，切图库需要重建。

### 2.2 筛选规则（按顺序判，记第一条不满足的）

| 规则 | 阈值 | 淘汰实例数 |
|---|---|---|
| iscrowd=1 | — | 331 |
| mask 面积 | ≥ 9,216 px（= 96×96）且 ≥ 图像面积的 1% | 21,698 |
| mask 的框离图像四边 | ≥ 2 px | 2,780 |
| 最大连通块（8 连通）占 mask 面积 | ≥ 95%；入库时只保留最大连通块 | 296 |
| 最大连通块面积 / 它的紧致框面积 | ≥ 0.40 | 477 |
| 同图任一其他实例（含 crowd）的 mask 落在本物体框内的面积 / 框面积 | ≤ 0.10 | 1,543 |
| **通过** | | **2,265**（非 holdout 图里共 29,390 个实例） |

最后两条是“没有被严重遮挡”的近似：COCO 没有遮挡标注，只能从形状和邻居猜。被没有标注的东西挡住的物体查不出来。

之后：每个类别按 `sha256(seed:cutout_id)` 排序取前 30 个（2,265 → 1,349）；不足 4 个的类别整类丢掉
（fork 1、handbag 1、knife 2、remote 1、snowboard 2、spoon 1、sports ball 2、tennis racket 1、toaster 2、wine glass 3，共 16 个）；
小图长边缩到 ≤ 256 px。

### 2.3 统计

| | 数量 |
|---|---|
| 切图 | 1,333（全部来自 COCO，取自 1,119 张图） |
| 类别 | 63（要求 ≥ 40） |
| dev / test | 656 / 677（按 cutout id 划分、按类别分层、种子 20261125，奇数时多的 1 个归 test；两侧无共用切图） |
| 来源图在 COCO-spatial one / two | 659 / 177 个切图 |
| 来源图在 COCO-Pairs | 472 个切图 |

每类 dev/test 个数：
airplane 4/4、apple 5/5、backpack 2/2、banana 5/5、bear 14/14、bed 11/12、bench 13/14、bicycle 8/8、bird 8/9、boat 12/13、book 5/5、bottle 4/5、
bowl 15/15、broccoli 7/8、bus 15/15、cake 15/15、car 15/15、carrot 5/5、cat 9/9、cell phone 4/4、chair 15/15、clock 15/15、couch 15/15、cow 15/15、
cup 12/12、dining table 15/15、dog 15/15、donut 15/15、elephant 15/15、fire hydrant 13/14、frisbee 2/2、giraffe 6/7、horse 15/15、hot dog 4/5、
keyboard 9/10、kite 5/6、laptop 15/15、microwave 3/4、motorcycle 12/12、mouse 2/2、orange 8/9、oven 15/15、parking meter 10/10、person 15/15、
pizza 15/15、potted plant 7/7、refrigerator 15/15、sandwich 14/15、sheep 10/11、sink 12/13、skateboard 2/2、stop sign 9/10、suitcase 11/12、
surfboard 5/6、teddy bear 11/12、toilet 15/15、traffic light 6/6、train 15/15、truck 15/15、tv 15/15、umbrella 13/13、vase 9/10、zebra 15/15。

## 3. 场景定义

- 画布 896×896。坐标：像素 (i, j) 覆盖 [i, i+1)×[j, j+1)，框 `[x1, y1, x2, y2)`，质心 = 下标均值 + 0.5（同 `src/data/coco_pairs.py`）。
- **mask** = alpha ≥ 128。**完整 mask** = 物体自己的全部 mask；**可见 mask** = 完整 mask 去掉被上层主物体完整 mask 盖住的部分。两种都以 RLE 存。框直接由 mask 算。
- **大小** = 缩放后完整 mask 紧致框的长边：8% / 12% / 18% × 896 → 72 / 108 / 161 px。两个主物体同一档；identity 孪生图的新 target 也是这一档；干扰物各自随机一档。
  实测 3,000 个主物体里 2,965 个恰好等于要求值，35 个少 1 px（缩放后边缘 alpha 落到阈值以下；实际值记在 `long_side_px`）。
- **间距** = 两个完整 mask 质心沿关系轴的坐标差的绝对值。物体只能放在整数像素上而质心带小数，所以不可能严格相等；
  做法是先放 reference，再把 target 放到离目标质心最近的整数位置，理论误差 ≤ 0.5 px。实测最大 0.499 px，平均 0.255 px（要求值和实际值都记）。
- **抖动**：垂直于关系轴的整数偏移，均匀取自 [−J, J]，J = floor(min(0.25×间距, 32))。上限随间距缩小，是为了小间距时关系轴仍然明确（间距 8 px 时 J = 2）。
- **整体位置**：在“三张图里主物体都完整落在画布内（留 4 px 边）”的所有整数平移中均匀抽；可行范围和抽到的值都记。
- **干扰物**：0 / 2 / 4 / 6 个，框与主物体在三张图里的所有位置、以及彼此之间相隔 ≥ 8 px。
  场景内所有物体（含 identity 孪生图的新 target）类别互不相同，且不来自同一个“易混类别组”（配置里列出，如 cup / mug / glass）。
- **背景**：solid / linear_gradient / radial_gradient / noise（n×n 随机网格双三次放大），参数全部写进 metadata，只凭参数可重画。
- **遮挡**：`occluded_fraction` = |下层完整 ∩ 上层完整| / |下层完整|；`iou_complete` = 两个完整 mask 的 IoU；
  两个可见 mask 按定义不相交，`iou_visible` 恒为 0。谁在上层由设计表指定，跟物体身份走（位置互换后不变）。
  下层物体露出不足 5% 时整个场景重采样（500 个里有 5 个重采样过 1 次，原因都是这一条；次数记在 `seed.attempt`）。
- **孪生图**：`position_swap` 把位移向量取反（互换后的质心离对方原质心 ≤ 1 px），关系标签取反，背景 / 干扰物 / 上下层不变；
  `identity_swap` 位置不变，target 换成另一类别的切图。三张图的 id 在 `twins` 字段互相引用。
- **关系标签**由实际质心算：left / right（horizontal）、above / below（vertical）。
- **随机数**：`scene_seed = sha256("<global_seed>:<index>")` 前 8 字节，第 a 次尝试用 `default_rng([scene_seed, a])`。场景只依赖（配置、切图库、序号）。
- **metadata**：每张图一行 JSON（1,500 行），含设计参数、种子、切图 id、两物体的框 / 质心 / 完整与可见 mask、z 顺序、遮挡量、干扰物、背景参数、PNG 与像素的 sha256、切图库 sha256。

## 4. 试跑设计表（500 个基础场景，只用 dev 侧切图）

间距 11 × 大小 3 × 轴 2 × 干扰物数 4 = 264 个格子。序号 k = index mod 264 按互素步长走遍所有格子，所以前 264 个场景每格恰好 1 个，
后 236 个再覆盖其中 236 格（每格 1–2 个场景，264 格全部覆盖）。背景 / 上层物体 / 方向用固定种子在层内均衡分配。

| 因子 | 各水平的基础场景数 |
|---|---|
| 间距 (px) | 8: 46、16: 46、24: 46、32: 46、48: 46、64: 45、96: 45、128: 45、192: 45、256: 45、384: 45 |
| 大小 | 8%: 167、12%: 167、18%: 166 |
| 轴 | horizontal 252、vertical 248 |
| 干扰物数 | 0 / 2 / 4 / 6 各 125 |
| 背景 | solid / linear_gradient / radial_gradient / noise 各 125 |
| 上层物体 | target 250、reference 250 |
| 关系标签（base） | left 126、right 126、above 124、below 124 |

每个间距档位内：每种大小 15–16 个，每个轴 22–24 个，每种干扰物数 11–12 个，每种背景 11–12 个，上层 22–23 / 22–23（完整交叉表在 manifest 的 `design_cross_by_distance`）。
63 个 dev 类别都当过主物体。

为什么只用 dev 切图：试跑是用来检查和调整流程的，属于“选择类操作”。正式集建议 dev 场景只用 dev 切图、test 场景只用 test 切图（配置里 `cutout_split`）。

## 5. 各间距档位的遮挡（base 图）

| 间距 (px) | 场景数 | 有遮挡的比例 | 被遮挡比例中位数（全部场景） | 中位数（只算有遮挡的） | 最大值 | iou_complete 中位数 |
|---|---|---|---|---|---|---|
| 8 | 46 | 100% | 0.689 | 0.689 | 0.930 | 0.467 |
| 16 | 46 | 100% | 0.606 | 0.606 | 0.934 | 0.354 |
| 24 | 46 | 100% | 0.566 | 0.566 | 0.906 | 0.310 |
| 32 | 46 | 100% | 0.426 | 0.426 | 0.800 | 0.232 |
| 48 | 46 | 89.1% | 0.240 | 0.256 | 0.593 | 0.133 |
| 64 | 45 | 75.6% | 0.066 | 0.206 | 0.574 | 0.032 |
| 96 | 45 | 37.8% | 0 | 0.097 | 0.437 | 0 |
| 128 | 45 | 8.9% | 0 | 0.089 | 0.131 | 0 |
| 192 / 256 / 384 | 45 / 45 / 45 | 0% | 0 | — | 0 | 0 |

遮挡与大小强相关（例如间距 96：8% 档 0/15 有遮挡，12% 档 20%，18% 档 93%），按间距×大小的表在 manifest 的 `occlusion_by_distance_and_size_base`。
三张图合并统计的结果相近（manifest 的 `all_variants`）。

## 6. 确定性

- 单元测试：同一场景用全新的库对象、不同顺序生成两次，PNG 逐字节相同、metadata 相同。
- 试跑检查：在主进程里重生成 20 个等间隔场景（序号 0, 26, …, 499，共 60 张图），与 8 进程写到磁盘的 PNG 和 metadata 行逐字节比较，**全部一致**。
- 逐字节一致的前提是同一个环境（tf5 锁文件：Pillow 13.0.0、numpy 2.4.6）。换 zlib / Pillow 版本后 PNG 字节可能不同，所以 metadata 里另存了 `pixels_sha256`（像素内容的哈希）。没有做跨环境验证。

## 7. 实测耗时与体积，及全量外推

硬件：Slurm cscc-cpu-p，8 核 32 G，8 个进程（节点 cn-14）。一个场景 = 3 张图。

| 实测（500 场景 / 1,500 张图） | |
|---|---|
| 墙钟总时间（含进程池启动、读切图） | 23.2 s → 0.046 s / 场景 |
| 单核耗时 / 场景（生成 + 写盘） | 平均 0.361 s，中位数 0.332 s，p95 0.718 s |
| PNG 字节 / 张 | 平均 145,763，中位数 140,682，最小 13,564，最大 402,354 |
| PNG 总量 | 218.6 MB |
| metadata | 7.42 MB（4,945 字节 / 张） |
| 切图库构建（一次性） | 46 s |

线性外推到正式全量 5,000 个基础场景（15,000 张图），同样的 8 核：

| | 外推值 |
|---|---|
| 墙钟时间 | 约 232 s（≈ 4 分钟）；合 1,804 核·秒 |
| PNG | 约 2.19 GB |
| metadata | 约 74 MB（超过 10 MB，只能放 nfs-stor，仓库里存 sha256） |

外推假设全量的因子配比与试跑相同；纳入 What's Up 切图后单张体积可能略有变化。

## 8. 联系表

`<outputs>/synth_review/trial_v0/sheet_000.jpg … sheet_007.jpg`（即 `/nfs-stor/xiwei.liu/interpretability/outputs/synth_review/trial_v0/`），
每页 8 个场景，每行 base | position_swap | identity_swap，红框 target、蓝框 reference、灰框干扰物，虚线是被遮挡物体的可见框，行首是参数。
场景按间距档位轮流排，**前 2 页（sheet_000、sheet_001）覆盖全部 11 个间距档位**；8 页共 64 个场景。`index.json` 列出每页的场景和间距。
我自己看了前两页，框、标签、孪生关系与图一致；没有做系统的人工检查。

## 9. 我做的设计选择

1. “间距”定义为沿关系轴的质心差，抖动另算；欧氏距离也记了（`euclidean_px`）。
2. 抖动上限随间距缩放（见上）。
3. 上下层跟物体身份走，不跟位置走；设计表里均衡。
4. 下层物体至少露出 5%，否则重采样。阈值是我定的，小间距时仍有被遮住 90% 以上的场景，下游可按 `occluded_fraction` 再筛。
5. 干扰物的非重叠用框判（间隔 ≥ 8 px），比用 mask 判更严。
6. 场景内类别互不相同，并加了一份手写的易混类别组（主要为之后纳入 What's Up 的 mug / cup、can / soda can 等准备）。分组是我凭常识写的，需要过目。
7. 类别均匀抽样（先抽类别再抽切图），所以切图少的类别（如 mouse dev 只有 2 个）里的单个切图出现得更频繁。
8. 合成时用软 alpha 混合（边缘抗锯齿），而 mask 用 alpha ≥ 128 的硬阈值；两者在边缘 1 px 内不完全一致。
9. 每张图一行 metadata（而不是每个基础场景一行）。
10. 划分按 cutout id（任务要求），所以同一张 COCO 图的两个切图可能分在两侧。

## 10. 已知局限

- 不真实：物体悬浮在程序化背景上，没有阴影和接触面，光照、视角、相对尺度互不一致（大象和橙子一样大）。背景颜色随机，有些很刺眼。
- 切图边缘：COCO 多边形不贴边，会带一圈原背景或切掉一点物体；没有做腐蚀或羽化。
- 切图质量没有人工检查：有的 COCO 实例本身被未标注的东西挡住、类别难认或标注有误（联系表里能看到个别难认的切图）。“未被严重遮挡”只是启发式。
- 放大：最小的切图长边约 96 px，18% 档需要放大到 161 px（最多约 1.7 倍），会发虚。
- 类别偏大物体：面积阈值淘汰了 fork / knife / spoon 等小物体类别；每类切图数不均（2–15 个 / 侧）。
- 大间距时 target 的绝对位置与标签不可避免地相关（间距 384 时两物体必然分处两侧）；单张图上无法消除，靠 position_swap 孪生图成对抵消。
- 切图来源图与评测集重叠：dev+test 共 659 个切图来自 COCO-spatial 用到的图、472 个来自 COCO-Pairs 的图（已记录，未排除——任务只要求排除 holdout）。
- What's Up 来源纳入后，同一个实物会出现在多张照片里，按 cutout id 划分无法把同一实物隔在一侧。

## 11. 待定问题

1. What's Up 自动标注产出后重建切图库（并重生成 trial_v0），还是 trial_v0 保持纯 COCO、等正式版再纳入？
2. 是否要把来自 COCO-spatial / COCO-Pairs 图的切图排除出去，或让切图划分跟随来源图的 dev / test？
3. 5% 的最小可见比例、易混类别组、0.10 的邻居重叠阈值是否合适？
4. 正式集的 dev / test 场景配比，以及是否需要给每个格子固定的重复数（5,000 / 264 ≈ 18.9）。
