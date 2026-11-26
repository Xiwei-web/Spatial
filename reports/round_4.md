# Round 4：第 3 轮标定的四项决定与收尾（只用 dev，不碰 test）

状态：**完成**（2026-11-26 05:50）。以下都是 dev 上标定坐标约定用的数，不是论文结果，不用于模型间比较。

## 一句话结论

四项决定已写进配置和代码（`50e67be`）。VST 加 P2 后，What's Up 上的 3D 框输出从 205 条降到 73 条，P2 的总平均 IoU 最高（0.603）。但按原规则 VST **仍判“未定”**：COCO 上 P0 更好（0.813 对 0.788，差 +0.025 [+0.002, +0.051]），所以 VST 的提示词在配置里仍写 pending，等 planner 裁定。

## 一、决定与改动（提交 `50e67be`，先提交、后跑作业）

规则（`src/bbox_calib.py`）没改。

| 模型 | 写入 configs/bbox_conventions.yaml `calibrated` 块 | 依据 |
|---|---|---|
| Qwen3-VL-8B-Instruct | P1，norm1000_xyxy | **规则判未定，planner 裁定**：两个子集的第一名约定都是 norm1000_xyxy，分歧只在提示词；P1 让 3D 框输出从 111 条降到 0 条；总差 +0.174 [+0.135, +0.215]；COCO 上 P0 只高 0.003，区间 [−0.001, +0.009] 含 0（本轮补算） |
| RoboBrain2.0-7B | P0，abs_model_xyxy；家族条目的 convention 也由 abs_orig_xyxy 改为 abs_model_xyxy（注释指向 manifest 与本报告） | **规则判未定，planner 裁定**：What's Up 上两种约定差得很开（0.588 对 0.162，差 −0.427 [−0.442, −0.412]）；COCO 上原图像素略好，见第二节；基座 Qwen2.5-VL 用输入像素 |
| SenseNova-SI-1.1 | P1，norm1_xyxy | 规则选出 |
| SenseNova-SI-1.2 / 1.3 | P0，range_norm1_else_norm1000 | 规则选出；同一个框里混用两种尺度的输出照样按规则换算，算作模型错误（planner） |
| VST-7B-RL | convention abs_model_xyxy；prompt_id **pending** | 见第三节 |
| 其余 10 个 | 按第 3 轮规则选出的（提示词, 约定），都与家族条目一致 | 规则选出 |

- `src/prompts.py` 的 `to_original_xyxy` 支持 `range_norm1_else_norm1000`（逐条按数值范围换算）。
- `scripts/bbox_calib_run.py` 增加 P2，只给 VST：P0 + " Output only a 2D box with four numbers. Do not output 3D boxes."。**P2 是看到第 3 轮结果后才加的。**
- 测试：`tests/test_bbox_calib.py` 新增 calibrated 块与提示词一致、range 换算两项；连同 `tests/test_parsing_and_data.py` 共 16 项通过。

## 二、CPU 补算（`scripts/bbox_calib_subset_ci.py` → `manifests/bbox_calib_round4_extra.json`；同一 bootstrap 函数与种子）

| 模型 / 子集 | 比较 | 平均 IoU | 差 [95% 区间] |
|---|---|---|---|
| Qwen3-VL-8B / COCO（100 图 200 条） | P0 − P1（都是 norm1000_xyxy） | 0.831 对 0.828 | +0.003 [−0.001, +0.009] |
| RoboBrain2.0-7B / COCO | 原图像素 − 模型输入像素（P0） | 0.699 对 0.694 | **+0.006 [+0.002, +0.009]**，不含 0 |
| RoboBrain2.0-7B / What's Up（200 图 400 条） | 同上 | 0.162 对 0.588 | −0.427 [−0.442, −0.412] |
| VST-7B-RL / COCO | P0 − P2（都是 model_xyxy） | 0.813 对 0.788 | **+0.025 [+0.002, +0.051]**，不含 0 |
| VST-7B-RL / What's Up | 同上 | 0.314 对 0.510 | −0.196 [−0.227, −0.165] |

- RoboBrain 在 COCO 上原图像素**确实略好，而且区间不含 0**，幅度 0.006。COCO 图的模型输入尺寸与原图尺寸之比（每张图一次，100 张）：宽 0.971–1.027（中位数 1.006），高 0.962–1.019（中位数 0.992），偏离 1 最多 3.8%，没有一张恰好为 1。所以两种约定在 COCO 上的坐标差不到 4%，但不是完全等价。What's Up 的比例固定为宽 0.897、高 0.875。
- planner 的裁定理由“COCO 上几乎等价”在幅度上成立，在统计上不成立；请据此确认是否维持裁定。

## 三、VST-7B-RL 补测 P2（作业 436482）

- `srun -p cscc-gpu-p --exclude=gpu-05,gpu-07,gpu-49 --gres=gpu:1 -c 8 --mem=64G -t 60 -J bbox_cal6 bash scripts/slurm/bbox_calib.sh tf5:VST-7B-RL`。gpu-04，05:30–05:42，生成 695.5 秒（0.2 GPU 小时），断点续跑只跑了 P2 的 600 条。日志 `<logs>/bbox_cal6.log`。
- 记录里的 git_commit 为 `50e67be…`，没有 `-dirty`（见第四节）。出错 0；7 条达到 96 个新 token 上限。

| 提示词 | 9 个数的 3D 框输出（全部在 What's Up） | 解析失败率 | 平均 IoU（model_xyxy） | What's Up / COCO |
|---|---|---|---|---|
| P0 | 205 | 0.342 | 0.480 | 0.314 / 0.813 |
| P1 | 284 | 0.475 | 0.410 | 0.209 / 0.812 |
| P2 | **73** | 0.135 | **0.603** | 0.510 / 0.788 |

- 3D 输出明显减少（205 → 73），但没有消失。P2 其余 8 条解析失败：6 个数的列表（如 `[0.01, 0.19, 1.91, 0.57, 0.73, 1.04]`），以及“图里没有 pan”一类回答。
- 按原规则在 P0/P1/P2 × 10 种约定里重选：第一名 (P2, model_xyxy) 0.603，IoU@0.5 0.710；第二名 (P0, model_xyxy) 0.480，差 +0.122 [+0.097, +0.148]；越界 1.0%。**What's Up 第一名是 (P2, model_xyxy)，COCO 第一名是 (P0, model_xyxy) → 未定。** 约定在三个提示词、两个子集上都是 model_xyxy，分歧只在提示词。按指令没有再加别的提示词。
- 结果文件 `manifests/bbox_calib_VST-7B-RL.json`（已按 P0/P1/P2 重新生成）。

## 四、运行时的 “-dirty”

- 第 3 轮的 13,200 条记录都是 `daec068…-dirty`。查文件修改时间：daec068 提交于 02:01:45，所有已跟踪文件最后一次修改是 02:01:35（就是 daec068 里提交的那些），02:02–05:20 运行期间没有任何已跟踪文件被修改。src/、scripts/、configs/ 都没有未提交的改动。
- `-dirty` 的来源是 `scripts/smoke_test.py` 的 `git_commit()` 用 `git status --porcelain`，把**未跟踪文件**也算了进去：`CLAUDE.md`（Xiwei 尚未决定是否提交，一直未跟踪），以及后来生成的未跟踪 manifest。
- 已改为 `--untracked-files=no`，只看已跟踪文件（`50e67be`）。本轮 VST 记录是干净的 `50e67be`。以后提交作业前先 commit。注意：2026-11-26 之前的冒烟等记录里的 `-dirty` 也可能只是未跟踪文件造成的。

## 五、文档

- `docs/DATA_REPORT.md` 第 8 节第 7 条、`docs/parts/bbox_impl_diff.md` 末尾加了标定结论与配置改动的说明。

## 六、需要决定的事

1. VST-7B-RL：规则判“未定”（COCO 上 P0 好 0.025，What's Up 上 P2 好 0.196；约定都是 model_xyxy）。请裁定用哪个提示词；配置里现在写 pending。
2. RoboBrain2.0-7B：COCO 上原图像素好 0.006，区间不含 0。是否维持 model_xyxy 的裁定。
