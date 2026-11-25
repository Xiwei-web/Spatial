# Round 1：合成数据 trial_v1、划分统一、深度数据检查、fp32_dec

状态：**完成**（2026-11-26 01:13 作业 435836 结束）。
路径约定：`<data>` = `/nfs-stor/xiwei.liu/interpretability/data`，`<outputs>` = `…/outputs`，`<logs>` = `…/logs`；其余为仓库内路径。没有计算任何模型正确率。

## 一句话结论

合成数据 trial_v1、按图像的统一划分、几何与真值框都已完成；两个候选深度数据集（SpatialRGPT、CV-Bench）都有很强的垂直位置捷径，CV-Bench 与捷径相反的只有 88 条（< 150），按要求停下，没有构造平衡子集。fp32_dec：InternVL3-8B 在 float32 下两种注意力实现几乎完全相同（期望坐标最大差 0.0064 / 1000），bf16 下期望坐标中位数差 0.13、最大 5.0；7B 模型 float32 + eager 单卡 40 GB 放得下但很紧。

## 一、深度数据（第 1–3 项）

命令：`python scripts/analysis_spatialrgpt_depth_cues.py`、`python scripts/analysis_cvbench_depth_cues.py`（CPU）。

| 数据 | 条数 | “下边缘更低 = 更近 / 在前” | 与捷径相反 |
|---|---|---|---|
| What's Up B（质心规则） | 204 | 100% | 0 |
| SpatialRGPT，跳过同名后全部 | 96 | 88.5% | 11 |
| SpatialRGPT，答案明确 | 67 | 92.5% | 5 |
| SpatialRGPT，靠 “No.” 推断 | 29 | 79.3% | 6 |
| CV-Bench 3D Depth，全部 | 600 | 85.3% | 88 |
| – Hypersim / SUN RGB-D / nuScenes | 各 200 | 80.0% / 82.0% / 94.0% | 40 / 36 / 12 |

- CV-Bench：HF 卡片许可 apache-2.0（图像来自 Omni3D；Omni3D 代码库为 CC BY-NC 4.0，各源数据集许可未逐一核实）；两个物体的 2D 框都在元数据里，没有 mask；图像上已画红 / 蓝框；508 张来源图。
- 第 3 项：88 < 150，没有构造平衡子集。结果文件：`manifests/spatialrgpt_depth_cues.json`、`manifests/cvbench_depth_cues.json`。

## 二、划分（第 4–5 项）

命令：`python scripts/make_joint_coco_split.py`、`python scripts/make_gqa_split.py`、`python scripts/check_split_consistency.py`。

- GQA-spatial 1,208 张图：dev 600 / test 608。样本：one-object 581 / 579，two-object 139 / 152。
- 597 张有 coco_id，其中因 coco_id 被强制分侧 **25 张**（21 张跟随 COCO-spatial / COCO-Pairs 并集，4 张跟随抠图来源图；dev 9、test 16）。其余随机：无 coco_id 611 张，coco_id 不在统一划分里 572 张。
- 跨数据集一致性：4,376 个图像键，1,269 个出现在多个数据集，**跨侧 0**；各数据集 dev ∪ test 等于全部样本。已写成测试。
- 旧文件保留并标记 `deprecated`，读取报错。
- **留出池有 1 张与 gqa_spatial_one 重叠**（COCO 126226 = VG 2406339），没有改动。

## 三、几何与真值框（第 6–7 项）

命令：`python scripts/build_coco_pairs.py`、`python scripts/coco_pairs_geometry_change.py`、`python scripts/spatialrgpt_box_review.py`。

- COCO-Pairs 5,124 个有向对按 mask 外接框重算：关系标签变化 **0**，间距档变化 **14**（2-3→3-5 4 个，0.5-1→1-1.5 6 个，1-1.5→1.5-2 4 个），投影重叠与否变化 16。归一化间距最大变化 0.070。逐条：`manifests/coco_pairs_geometry_change.csv`。旧值在 `*_detbox` 字段，旧文件 `<data>/coco_pairs/pairs_detbox_20261125.jsonl` 保留。
- SpatialRGPT 低 IoU 联系表：`<outputs>/data_review/spatialrgpt_box/lowest20_depth_used.jpg`（实际会用的 192 个物体）、`lowest20_all_regions.jpg`（全集 2,546 个区域）。
- 我看了两张：低 IoU 多数是自带框比 mask 松。另发现 **mask 有零星碎点会撑大外接框**：192 个物体里 57 个有面积 < 1% 的碎块，去掉后 32 个的框会变，19 个 IoU < 0.95，6 个 < 0.8。真值框目前按全部 mask 像素取，没有去碎点。

## 四、合成数据 trial_v1（第 8–13 项）

命令（Slurm）：435883 `build_synth_library_v1.py`；435902 `generate_synth_v1.py --workers 8`。日志 `<logs>/synth_lib_v1_435883.out`、`<logs>/synth_trial_v1_435902.out`。

- 抠图库：1,333 → 962 个通过全部过滤 → 两侧各 ≥ 3 的类别 53 / 63 → 最终 **923** 个（dev 456 / test 467）。单项不通过：长边 32、放大 170、宽高比 89、CLIP 187。
- 排除的类别（dev / test 剩余数）：airplane 1/1、backpack 2/1、book 0/7、bottle 2/4、frisbee 2/2、microwave 2/2、mouse 1/2、potted plant 0/5、skateboard 0/2、surfboard 2/1。
- CLIP 用 `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`（MIT，下载 1.71 GB）；没有用 OpenAI CLIP，因为 LLaVA-1.5 的视觉塔就是它。
- 场景：500 + 各 2 张孪生 = 1,500 张，dev 250 / test 250；重生成 20 个场景 60 张逐字节相同；30 秒，0.33 GB。
- 各档场景数：gap 0 / 4 / 8 / 16 / 32 各 46，gap 64 / 128 / 256 各 45，遮挡三档各 45。
- 身份互换质心偏差：中位数 0.41 px，最大 0.70 px（要求 ≤ 2）；面积比 0.985–1.014。放大倍数最大 1.496；离边最近 16 px。
- 遮挡比例（base）：<10% 档 0.006–0.097；10–30% 档 0.117–0.293；30–50% 档 0.313–0.490。
- 每类抠图数、每类过滤前后数量：`manifests/synth_library_v1.json`、`manifests/synth_trial_v1.json`。
- 联系表：`<outputs>/synth_review/trial_v1/sheet_000…007.jpg`、`<outputs>/synth_review/library_v1/sheet_00…04.jpg`。我看了 trial_v1 前两张和 library_v1 第一张。

两个局限：
1. **身份互换孪生的边缘间距不受控**：质心和面积对齐后形状不同，例如 gap0 档实际间距中位数 13.5 px、最大 69 px。质心对齐与边缘间距不能同时保证，按要求以质心为准，实际值在元数据里。
2. **CLIP 过滤不严**：概率大多饱和在 1.00，残缺的抠图也能通过（被枝条遮住的熊、只剩床面的床）；有左右镜像的近重复。

## 五、GLM 重跑（第 15 项的一半）

Slurm 435862，日志 `<logs>/glm_rerun.out`、`<logs>/smoke/GLM-4.6V-Flash.log`。

- 现在 1148×840、1,230 个视觉 token（30×41），输入像素与 Qwen2.5-VL 逐字节相同。
- 冒烟：框 40/40 可解析；关系 13/20 可解析（仍不过，原因是不按要求只答字母）；其余六项通过。
- 映射验证重合率 0.42（最低 0.40）；sdpa 对 eager：框 29/40、关系 18/20 逐 token 一致，选项第一名 20/20 一致，选项 logit 最大差 0.281。
- 覆盖统计已按新分辨率重算（Slurm 435918）。旧输出留档 `<outputs>/archive/glm_default_resolution_20261125/`。

## 六、fp32_dec（第 14–15 项）

作业 435836：COMPLETED，exit 0，10 分 59 秒，gpu-55（不是坏节点，没有重新提交），commit eb9db24。日志 `<logs>/fp32_decision.log`；结果 `<outputs>/diagnostics/teacher_forced_InternVL3-8B_{bf16,fp32}.json`、`teacher_forced_compare_InternVL3-8B.json`、`bench_attn_*.json`。完整表和说明在 `docs/parts/bbox_impl_diff.md` 第 4–6 节。

命令：`bash scripts/slurm/fp32_decision.sh`（作业内 6 步全部 rc=0）；`python scripts/compare_teacher_forced.py InternVL3-8B`（CPU；我在汇总表里加了均值和 n 两列，没有改计算）。

### 6.1 InternVL3-8B：teacher-forced（40 条 × 4 个坐标 = 160）

- fp32 确认：参数 dtype 只有 float32（bf16 运行只有 bfloat16）；调用计数 eager 轮 19,040 / 0，sdpa 轮 0 / 19,040，两种精度都如此。出错 0、OOM 0、NaN 0。
- 表注：**“真值”是检测框 `box_det_xyxy`，不是默认真值框 `box_gt_xyxy`**；这里只比较实现差异。期望坐标单位 0–1000。lm_head 用模型自带的；float32 lm_head 的行见 `bbox_impl_diff.md`。

| 读出量 | lm_head | 比较 | n | 均值 | 中位数 | 95% 分位 | 最大 |
|---|---|---|---|---|---|---|---|
| 期望坐标（精确枚举，0–1000 单位） | 模型自带 | bf16 eager vs bf16 sdpa | 160 | 0.449 | 0.126 | 1.97 | 5.04 |
| 期望坐标（精确枚举，0–1000 单位） | 模型自带 | fp32 eager vs fp32 sdpa | 160 | 0.000149 | 4.55e-05 | 0.000597 | 0.00643 |
| 期望坐标（精确枚举，0–1000 单位） | 模型自带 | bf16 eager vs fp32 eager | 160 | 0.79 | 0.168 | 4.39 | 14 |
| 期望坐标（精确枚举，0–1000 单位） | 模型自带 | bf16 sdpa vs fp32 sdpa | 160 | 0.816 | 0.199 | 3.68 | 13.7 |
| 期望坐标（逐位条件，0–1000 单位） | 模型自带 | bf16 eager vs bf16 sdpa | 160 | 1.1 | 0.234 | 4.01 | 30 |
| 期望坐标（逐位条件，0–1000 单位） | 模型自带 | fp32 eager vs fp32 sdpa | 160 | 0.000197 | 6.1e-05 | 0.000525 | 0.00671 |
| 期望坐标（逐位条件，0–1000 单位） | 模型自带 | bf16 eager vs fp32 eager | 160 | 1.3 | 0.302 | 6.34 | 15.7 |
| 期望坐标（逐位条件，0–1000 单位） | 模型自带 | bf16 sdpa vs fp32 sdpa | 160 | 1.25 | 0.325 | 4.94 | 16.2 |
| 真值数字串对数概率（nat，每个坐标） | 模型自带 | bf16 eager vs bf16 sdpa | 160 | 0.118 | 0.0836 | 0.359 | 0.786 |
| 真值数字串对数概率（nat，每个坐标） | 模型自带 | fp32 eager vs fp32 sdpa | 160 | 2.45e-05 | 1.66e-05 | 6.28e-05 | 0.000203 |
| 真值数字串对数概率（nat，每个坐标） | 模型自带 | bf16 eager vs fp32 eager | 160 | 0.116 | 0.0991 | 0.331 | 1 |
| 真值数字串对数概率（nat，每个坐标） | 模型自带 | bf16 sdpa vs fp32 sdpa | 160 | 0.129 | 0.0962 | 0.389 | 0.649 |

- 互检：同一个量用两条计算路径，fp32 最大差 1.6e-4 nat；bf16 中位数 0.06–0.11、最大 0.42–0.81 nat，与 bf16 两种实现之间的差异同一量级（bf16 的数值底噪）。
- 逐 token 一致率（这个作业不产生，引自 435763 / 435451）：bf16 eager 对 sdpa 34/40，fp32 eager 对 sdpa 40/40，eager 的 bf16 对 fp32 31/40，sdpa 的 bf16 对 fp32 29/40；关系提示 bf16 20/20，fp32 未测。
- 每条耗时（含全部读出）：bf16 eager 1.30 秒、sdpa 1.11 秒（1 卡）；fp32 eager 3.26 秒、sdpa 3.31 秒（2 卡）。峰值显存：bf16 19.2 GB（1 卡）；fp32 15.93 + 18.85 GB（2 卡）；fp32 单卡未测。

### 6.1.1 |e_tree 差| 最大的 10 个坐标（bf16 对 fp32，eager）

脚本 `scripts/teacher_forced_top_diffs.py`（CPU，只读已有 JSON），结果 `outputs/diagnostics/teacher_forced_top10_InternVL3-8B.json`。e_tree 单位 0–1000，lm_head 用模型自带；“真值”是检测框。

| # | 样本 / 角色 | 物体 | 坐标 | 真值（检测框） | bf16 eager | bf16 sdpa | fp32 eager | fp32 sdpa | 差（bf16 − fp32，eager，绝对值） | fp32 对真值的对数概率 | 是否第 1 节的 pan / pillow 样本 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sunglasses_under_table / reference | table | y2 | 937 | 706.0 | 705.7 | 692.0 | 692.0 | 14.04 | -4.95 | 否 |
| 2 | pillow_on_table / target | pillow | y2 | 438 | 446.0 | 445.9 | 455.9 | 455.9 | 9.96 | -5.26 | 否 |
| 3 | bowl_in-front_of_headphones / reference | headphones | y2 | 539 | 532.7 | 536.0 | 539.7 | 539.7 | 7.04 | -4.59 | 否 |
| 4 | glasses_right_of_armchair / target | glasses | x1 | 722 | 394.1 | 396.8 | 400.9 | 400.9 | 6.77 | -9.36 | 否 |
| 5 | knife_left_of_chair / target | knife | x1 | 250 | 200.1 | 197.4 | 206.1 | 206.1 | 5.99 | -4.65 | 否 |
| 6 | mug_right_of_table / target | mug | x1 | 758 | 639.4 | 643.7 | 645.0 | 645.0 | 5.62 | -4.90 | 否 |
| 7 | ball-of-yarn_on_chair / target | ball of yarn | y1 | 535 | 361.9 | 361.7 | 357.3 | 357.3 | 4.65 | -6.60 | 否 |
| 8 | knife_left_of_chair / target | knife | y1 | 757 | 728.9 | 727.9 | 724.5 | 724.5 | 4.45 | -3.54 | 否 |
| 9 | shoes_under_chair / target | shoes | y1 | 762 | 732.8 | 731.5 | 728.4 | 728.4 | 4.39 | -5.26 | 否 |
| 10 | pillow_on_table / target | pillow | y1 | 313 | 320.9 | 320.9 | 325.1 | 325.1 | 4.28 | -7.10 | 否 |

- pan_left_of_chair / target：真值 [223, 772, 385, 873]；e_tree bf16 eager [166.4, 762.7, 356.6, 883.1]、bf16 sdpa [161.4, 763.0, 358.2, 883.3]、fp32 eager [168.2, 762.7, 359.2, 883.0]、fp32 sdpa [168.2, 762.7, 359.2, 883.0]；bf16 对 fp32（eager）四个坐标的差 [1.7, 0.0, 2.6, 0.1]，其中最大者在 160 个坐标里排第 16。
- pillow_left_of_armchair / target：真值 [47, 765, 332, 932]；e_tree bf16 eager [53.6, 730.3, 265.0, 942.1]、bf16 sdpa [53.6, 730.2, 264.9, 941.7]、fp32 eager [53.5, 727.5, 265.5, 941.4]、fp32 sdpa [53.5, 727.5, 265.5, 941.4]；bf16 对 fp32（eager）四个坐标的差 [0.1, 2.8, 0.4, 0.7]，其中最大者在 160 个坐标里排第 15。

- **前 10 个都不属于第 1 节的 pan / pillow 两条**（`pillow_on_table` 与 `pillow_left_of_armchair` 是不同样本）；那两条样本在 teacher-forced 下的差只有 2.6 和 2.8，排第 16 和 15。所以自由生成里的大差异不对应 teacher-forced 读出里的大差异。
- 这 10 个坐标 fp32 给真值的对数概率都很低（−3.5 到 −9.4），期望值常远离真值（如 glasses x1 真值 722，期望约 400）：分布宽、多峰，期望对概率的小变化敏感。fp32 的 eager 与 sdpa 在这 10 个上相同（精度到 0.1）。

### 6.2 float32 + eager 显存与耗时（每种 5 条，A100 40 GB）

参数 dtype 只有 float32，注意力权重能取出；该脚本没有做调用计数；bf16、sdpa、teacher-forced 未测。bf16 eager 对 sdpa 逐 token 一致率（已有）：Qwen2.5-VL-7B 框 23/40、关系 20/20；InternVL3_5-8B 框 24/40、关系 20/20。

| 模型 | 卡数 | 视觉 token | 操作 | 峰值 allocated（GB，逐卡） | 峰值 reserved（GB） | 每条秒数 | 成功 / 出错或 OOM |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 1 | 1230 | 前向，不取注意力 | 34.08 | 34.32 | 1.83 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 1 | 1230 | 前向，全部层注意力留在 GPU | 37.22 | 38.74 | 1.83 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 1 | 1230 | 前向，逐层注意力搬到 CPU | 34.08 | 34.33 | 3.98 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 1 | 1230 | 框提示贪心生成 | 34.06 | 34.31 | 2.89 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 2 | 1230 | 前向，不取注意力 | 19.03 + 15.80 = 34.83 | 19.27 + 17.14 | 1.84 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 2 | 1230 | 前向，全部层注意力留在 GPU | 22.15 + 18.32 = 40.47 | 22.92 + 19.49 | 1.87 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 2 | 1230 | 前向，逐层注意力搬到 CPU | 19.03 + 15.80 = 34.83 | 19.27 + 17.14 | 4.08 | 5 / 0 |
| Qwen2.5-VL-7B-Instruct | 2 | 1230 | 框提示贪心生成 | 19.01 + 15.63 = 34.64 | 19.25 + 15.79 | 3.23 | 5 / 0 |
| InternVL3_5-8B | 1 | 256 | 前向，不取注意力 | 31.99 | 32.36 | 0.41 | 5 / 0 |
| InternVL3_5-8B | 1 | 256 | 前向，全部层注意力留在 GPU | 32.44 | 32.83 | 0.41 | 5 / 0 |
| InternVL3_5-8B | 1 | 256 | 前向，逐层注意力搬到 CPU | 31.99 | 32.34 | 0.68 | 5 / 0 |
| InternVL3_5-8B | 1 | 256 | 框提示贪心生成 | 31.97 | 32.05 | 1.49 | 5 / 0 |
| InternVL3_5-8B | 2 | 256 | 前向，不取注意力 | 15.13 + 17.02 = 32.15 | 15.16 + 17.32 | 0.38 | 5 / 0 |
| InternVL3_5-8B | 2 | 256 | 前向，全部层注意力留在 GPU | 15.24 + 17.27 = 32.51 | 15.30 + 17.60 | 0.38 | 5 / 0 |
| InternVL3_5-8B | 2 | 256 | 前向，逐层注意力搬到 CPU | 15.13 + 17.02 = 32.15 | 15.21 + 17.32 | 0.68 | 5 / 0 |
| InternVL3_5-8B | 2 | 256 | 框提示贪心生成 | 15.12 + 16.96 = 32.08 | 15.16 + 16.98 | 1.82 | 5 / 0 |

- **更正我之前的说法**：“7B–8B 模型 float32 + eager 单卡放不下”不对。Qwen2.5-VL-7B 单卡能跑，但全部注意力留在 GPU 时 reserved 38.74 / 39.49 GB，没有余量加干预 hook；建议逐层搬到 CPU 或用 2 卡。
- 其余 6 个第一组模型的 fp32 组合都未测（清单见 `bbox_impl_diff.md` 第 6 节）。

## 报错记录

- `build_coco_pairs.py` 第一次重建在写报告时 `KeyError: 'n_images'`（统一划分的分层表字段名不同），已修；重跑后 pairs.jsonl 逐字节相同。
- 435886（coco_pairs_gt）在排队时由我取消，改为在登录节点低优先级直接运行，耗时 11 秒。

## 需要决定的事

1. 深度检验的数据来源（三个候选都有捷径）。
2. 留出池里与 GQA 重叠的 1 张图是否剔除（修改数据集，需 Xiwei 确认）。
3. SpatialRGPT 真值框是否去碎点（同上）。
4. 合成抠图库是否需要比 CLIP 更严的筛选；身份互换孪生的边缘间距不受控是否可接受；自然背景用哪个来源。
5. SpatialRGPT 改写规则的 20 条对照（`docs/parts/F7_spatialrgpt.md`）等 Xiwei 确认。
6. 框预测精度（bf16 / fp32，自由生成 / teacher-forced 读出）：数据见第六节。只有 InternVL3-8B 测过 fp32 的实现一致性。若选 fp32，其它模型是否要补同样的检查；Qwen2.5-VL 系单卡没有余量，默认 2 卡。
