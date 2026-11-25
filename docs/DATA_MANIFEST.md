# 数据清单（DATA_MANIFEST）

数据根目录：`configs/paths.yaml` 的 `data`（`/nfs-stor/xiwei.liu/interpretability/data`）。下载脚本 `scripts/download_data.sh`，核对脚本 `scripts/data_manifest.py`（结果在 `manifests/data_summary.json`）。下载日期 2026-11-25。原始压缩包保留在 `data/_archives/`。

每个目录下的 `SHA256SUMS` 是逐文件 sha256 清单；下表给出该清单文件自身的 sha256 以便核对。

## 已下载

| 数据 | 官方来源 | 版本 | 许可 | 文件数 | 体积 | SHA256SUMS 位置（其 sha256 前 16 位） |
|---|---|---|---|---|---|---|
| What's Up（Subset A/B、COCO-spatial、GQA-spatial 的标注，A/B 图像，GQA-spatial 所需图像） | https://github.com/amitakamath/whatsup_vlms ，文件在其 `dataset_zoo/aro_datasets.py` 指向的 Google Drive（README 给出的文件夹：https://drive.google.com/drive/u/3/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ ） | 仓库 HEAD `7c1f2550`（2024-02-28）；Drive 文件无版本号，压缩包内文件时间为 2024-02-06 | 代码仓库 MIT；数据本身没有单独的许可声明（图像分别来自作者自拍、COCO、Visual Genome/GQA） | 5,635 | 850 MB | `data/whatsup/SHA256SUMS`（`bc9a308d7549c6c2`） |
| COCO val2017 图像 + `instances_val2017.json` | http://images.cocodataset.org/zips/val2017.zip ， http://images.cocodataset.org/annotations/annotations_trainval2017.zip （只解出 `instances_val2017.json`） | 2017 | 标注 CC BY 4.0；图像遵循 Flickr 使用条款（见 cocodataset.org/#termsofuse，本次未逐字核对条款页） | 5,001 | 835 MB | `data/coco/SHA256SUMS`（`b1fde39e9da05417`） |
| 深度关系候选集的说明文件与标注元数据（**没有下图像**） | HF datasets：`a8cheng/SpatialRGPT-Bench@01b38d2c`、`ccvl/3DSRBench@5c90e422`、`nyu-visionx/CV-Bench@bc284db5` | 固定到左列 commit | 见下文 | 7 | 6.9 MB | `data/depth_candidates/SHA256SUMS`（`14ac1216ac940603`） |
| 原始压缩包 | 同上 | — | — | 5 | 1.9 GB | `data/_archives/SHA256SUMS`（`9aaecee3875b06cd`） |

### What's Up 各子集数量（核对结果：全部通过）

| 加载器名 | 标注文件 | 样本数 | 不同图像数 | 缺图 | 关系分布 | 自带框/mask |
|---|---|---|---|---|---|---|
| `whatsup_a`（Subset A） | `controlled_images_dataset.json` | 412 | 412 | 0 | left 103 / right 103 / on 103 / under 103 | 无 |
| `whatsup_b`（Subset B） | `controlled_clevr_dataset.json` | 408 | 408 | 0 | left 102 / right 102 / front 102 / behind 102（**front / behind 与垂直位置完全混淆**） | 无 |
| `coco_spatial_one` | `coco_qa_one_obj.json` | 2,247 | 1,451 | 0 | bottom 744 / left 576 / right 564 / top 363 | 见下 |
| `coco_spatial_two` | `coco_qa_two_obj.json` | 440 | 295 | 0 | left 150 / right 129 / above 86 / below 75 | 见下 |
| `gqa_spatial_one` | `vg_qa_one_obj.json` | 1,160 | 1,043 | 0 | left 392 / right 376 / bottom 198 / top 192 / behind 2 | 无 |
| `gqa_spatial_two` | `vg_qa_two_obj.json` | 291 | 258 | 0 | right 137 / left 127 / front 19 / behind 5 / top 3 | 无 |

- **任务书要求的核对**：Subset A 的 left/right = 103 + 103 = **206**（符合）；Subset B = **408**（符合）。
- **What's Up B 的 front / behind 与垂直位置完全混淆**（v3 上“target 质心更低 = front”204 / 204 成立），答对不能说明用了深度信息；前后关系不能只依赖 SpatialRGPT（它也有较强的垂直位置捷径）。
- 真值框统一为 mask 外接框（用户 2026-11-25）：COCO-spatial / COCO-Pairs 由 COCO 实例 mask 算出（与 COCO 自带框 IoU 最低 0.88）；What's Up 为 v3 的 `box_gt_xyxy`；SpatialRGPT 由自带 RLE 算出。
- **更正与剔除**（更正表 `annotations/whatsup/corrections.yaml`；原始数据未改动）：`pillow_left_of_chair` 与 `pillow_right_of_chair` 两张**剔除**。原因：原始数据左右与标签相反——文件名和官方标注 JSON 分别为 left / right，图像中枕头却分别在椅子的右边 / 左边。已核对：本地文件与官方压缩包内同名文件 sha256 相同，加载器给出的标签与官方 JSON 一致，所以不是下载、解压或加载造成的。两张都在 Subset A 的 test 里；划分文件不变，`load_split` 读取时剔除，因此 **Subset A 实际可用 410 张（dev 204 / test 206），其中 left/right 204 张**（原始发布 206 张，核对数量时仍以 206 为准）。用户 2026-11-25 决定剔除、不改标签。
- Subset A 图像目录里有 421 个文件，比标注多 9 个（`armchair.jpeg`、`chair.jpeg`、`table.jpeg`、`chair_*_table.jpeg`×3、`chair2_*_table.jpeg`×3）；它们不在官方标注 JSON 里，加载器不使用。
- GQA-spatial 图像目录有 4,797 张，标注实际引用 1,043 + 258（有重叠）张；这是官方打包的子集，没有下载完整 GQA/VG。
- COCO-spatial 的框和 mask **不是 What's Up 发布的**：加载器用 COCO 实例标注按类别名匹配，仅当该类别在图中恰有一个非 crowd 实例时才给出。实测 one-object 2,247/2,247、two-object 440/440（target 与 reference 都有）都能唯一匹配。
- GQA-spatial 没有框：需要 GQA scene graph（未下载）。如需要请告知。
- 数据特点（非错误）：GQA 的 caption 有 “on the behind”“to the top of” 这类少量稀有关系（2～5 个样本），分层划分后每侧只有 1～3 个。

### 划分

**COCO-spatial、COCO-Pairs、合成抠图**：2026-11-25 起按图像统一划分（`splits/coco_images_v2.json`，脚本 `scripts/make_joint_coco_split.py`），各数据集的 `*_v2.json` 继承它。旧的 `coco_spatial_one/two.json`、`coco_pairs.json`、`synth_cutouts.json` 保留但标记 `deprecated`，读取会报错。GQA-spatial 也改为按图像划分（`splits/gqa_images_v2.json`，有 coco_id 的跟随 COCO；VG 图像元数据 `image_data.json.zip`，1.8 MB，来自 homes.cs.washington.edu/~ranjay/visualgenome，sha256 `b87a9491…`）。下面一段说的是 What's Up 的划分。


`splits/<数据集>.json`，由 `scripts/make_splits.py` 生成：种子 20261125，按关系类别分层，对半（某类为奇数时多出的 1 个归 test）。文件已提交，脚本默认不覆盖已有划分。

| 数据集 | dev | test |
|---|---|---|
| whatsup_a | 204 | 208 |
| whatsup_b | 204 | 204 |
| coco_spatial_one（v2，按图像） | 1,122 | 1,125 |
| coco_spatial_two（v2，按图像） | 208 | 232 |
| coco_pairs（v2，按图像；有向对） | 2,550 | 2,574 |
| gqa_spatial_one（v2，按图像） | 581 | 579 |
| gqa_spatial_two（v2，按图像） | 139 | 152 |

冒烟测试用的 20 张：`splits/smoke20.json`（A、B 的 dev 各 10 张）。

### 抽样可视化（请肉眼检查）

`/nfs-stor/xiwei.liu/interpretability/outputs/sanity/` 下每个数据集一张图（5 个样本，带标签；COCO 带框）：
`whatsup_a.png`、`whatsup_b.png`、`coco_spatial_one.png`、`coco_spatial_two.png`、`gqa_spatial_one.png`、`gqa_spatial_two.png`。

## GSR-Bench 的 What's Up 框 / mask / 深度标注：未找到公开发布

- 论文（arXiv 2406.13246v2）正文只有一句 “All the code and data will be publicly available.”，HTML 全文里没有 GitHub / HF / Drive 链接。
- GitHub 仓库搜索 “GSR-Bench”：命中的 `KLMmotion/GSR-bench`、`Gento-Teleoperation-Apex/GSR-bench` 是不相关的项目（描述为 “GSRbench Code Implementation”，无 What's Up 内容）；第一作者 GitHub（NavidRajabi）公开仓库里没有 GSR-Bench。
- HF datasets 搜索 “gsr-bench” 无结果。
- 结论：截至 2026-11-25 没找到官方发布。检索不保证穷尽（例如作者个人主页、邮件索取未尝试）。
- HF 上有第三方的 What's Up 派生数据集（如 `ServiceNow/whatsup_all`、`Likithp/whatsup-*`），未核实其内容与许可，未下载。
- 本机已有可用于自行生成标注的工具权重（只读，属于其它项目）：`/nfs-stor/xiwei.liu/tool/checkpoints/` 下有 groundingdino（4.8 G）、sam3（6.5 G）、depth_anything_3（6.8 G）。

## 深度关系候选集（只下了元数据，等你三选一）

| | SpatialRGPT-Bench | 3DSRBench | CV-Bench |
|---|---|---|---|
| HF 仓库 | `a8cheng/SpatialRGPT-Bench` | `ccvl/3DSRBench` | `nyu-visionx/CV-Bench` |
| 许可（HF 卡片元数据） | **未声明**（cardData 无 license） | cc-by-4.0 | apache-2.0 |
| 完整体积 | 2.91 GB（1,415 个文件；parquet train 与 val 各约 0.92 GB，文件大小逐一相同） | 1.52 GB（其中两个内嵌图像的 tsv 共 1.2 GB；csv 标注 0.3 MB + `coco_images.zip` 303 MB） | 0.41 GB |
| 样本总数 | 1,406（定性 657 + 定量 749） | 2,100（12 种关系各 175） | 2,638（2D 1,438 + 3D 1,200） |
| 带框 | **是**，每条都有 `bbox`（每个被提问区域一个框） | 否（csv 里只有物体名称文本） | 仅 3D 部分 1,200 条有 `bbox` |
| 带 mask | **是**，每条都有 `rle` | 否 | 否 |
| 前/后（深度）关系样本 | `behind_predicate` 55 + `front_predicate` 55 = **110**（另有距离类定量题，如 `distance_data` 148） | `location / closer to camera` **175**；另有 `orientation / in front of` 175（指物体朝向意义上的“前”，不是相机深度）、`multi_object / closer to` 175（物体间距离） | 3D `Depth`（哪个离相机更近）**600**；`Distance` 600 |
| 图像来源 | SUNRGBD 594、Hypersim 384、nuScenes 189、ARKitScenes 140、KITTI 99 | COCO（train2017 2,064 张、val2017 36 张，以 URL 引用） | 3D 部分全部来自 Omni3D；2D 部分 COCO / ADE20K |
| 需要注意 | 问题用 `<mask>` / “Region [0]” 指代物体，是为区域输入模型设计的；用于普通 VLM 需要自己把区域换成名称或画框 | 无框无 mask，需自行标注；图像绝大多数在 COCO train2017，不在已下载的 val2017 里 | 3D 题的图像上**已经画了红/蓝框**来指代物体（题干写 “highlighted by a red box”），图像不是干净原图 |

## 合成场景背景图：未下载，候选来源

| 来源 | 官方页面 | 许可（页面原文） | 规模 |
|---|---|---|---|
| Poly Haven HDRI | https://polyhaven.com/license | “Our assets are all licensed as CC0 … You can use our assets for any purpose, including commercial work.” | 总数未确认 |
| BG-20k | https://github.com/JizhiziLi/GFM | “20,000 high-resolution background images excluded salient objects”；“Agreement (MIT License)” | 2 万张；下载体积页面未写 |
| DTD 纹理 | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | “made available to the computer vision community for research purposes” | 5,640 张，625 MB |
| Places365 | https://github.com/CSAILVision/places365 | “The copyright of all the images belongs to the image owners”（CC BY 只覆盖模型） | train 约 180 万张；val 体积未确认 |

ADE20K（需注册，条款页当时打不开）和 SUN397（页面 404）未能核实。

## ImageNet val：未下载；集群上没有官方公共副本，有其他用户的可读副本

- 没有找到管理员维护的公共数据集目录（`/apps`、`/datasets`、`/data`、`/share*` 均无）。
- `/nfs-stor/mingming.gong/imagenet1k/val/`：他人目录，全局可读，按 1000 个 synset 子目录组织，抽查 `n01440764` 下 50 张（与 ILSVRC2012 val 每类 50 张一致）。未做完整性校验，也不清楚其来源与是否会被移走。
- `/nfs-stor/guangyi.chen/imagenet.tar.gz`（166 GB，他人文件，可读；内容未查看）。
- 使用他人目录前建议先征得对方同意；ImageNet 本身需要在 image-net.org 接受条款。
