# 实验前准备报告（PREP_REPORT）

日期：2026-11-25。范围：环境、模型清点与补齐、数据集、冒烟测试。**本阶段没有跑任何正式实验，本报告不含任何可写进论文的数字。** 冒烟测试只回答“管线能否跑通、内部状态能否拿到”，没有计算任何正确率。

**更新（2026-11-25 晚）**：第 3.3 节的冒烟表是修复 eager 注意力问题之后的重跑结果；第 4–6 节标注了各事项的后续。数据构建、1.3 / 1.4 的排查结论、两种注意力实现的对比、What's Up 标注 v3 定稿见 `docs/DATA_REPORT.md`，两份报告不一致时以 DATA_REPORT 为准。

配套文档：`docs/MODEL_INVENTORY.md`（模型逐项清单）、`docs/DATA_MANIFEST.md`（数据清单）、`docs/SMOKE_RESULTS.md`（冒烟结果表，脚本生成）。

---

## 1. 集群与现状（阶段 A）

### 1.1 集群

| 项 | 结果 |
|---|---|
| 调度系统 | Slurm 25.05.6，集群名 `mbz-ciai`，登录节点 `ciai-login-1` |
| 分区 | `long`（默认，30 节点）、`cscc-gpu-p`（13 节点）、`cscc-cpu-p`（12 节点，无 GPU） |
| GPU | 全部是 NVIDIA A100-SXM4-**40GB**，每节点 4 张；驱动 570.195.03，驱动支持的 CUDA 12.8。**没有 80GB 卡** |
| 节点 | 256 线程、约 250 GB 内存 |
| 本账号可用的 QoS | `cscc-gpu-qos`（批处理：单作业 48 h，每用户最多 4 卡、同时 2 个作业）、`gpu-debug-qos`（**单作业 3 h**，每用户最多 4 卡、2 个作业；`srun`/`salloc` 交互作业会被自动路由到它）、`cscc-cpu-qos`（72 h，同时 3 个作业） |
| `long` 分区 | **本账号不可用**：`sbatch -p long` 返回 `Invalid qos specification`（该分区只允许 `gpu-12` / `gpu-debug-qos`，而账号 `cscc-users` 没有 `gpu-12`） |
| 排队情况（当天实测） | `sbatch` 到 `cscc-gpu-p` 的预计开始时间约 **1 天多之后**；走 debug QoS 的 `srun` 1 卡作业数分钟内能启动，3 卡作业调度器预估要等十几个小时；CPU 分区排队约 1 小时 |
| 计算节点联网 | **能**。在 `gpu-49` 上访问 huggingface.co 和 github.com 均返回 200；登录节点同样能联网（hf-mirror.com 也通） |
| 其它 | `sacctmgr`、`sacct` 在登录节点不可用（连不上 slurmdbd），以上限额来自 `scontrol show assoc_mgr`；申请 1 卡的作业里 `nvidia-smi` 能看到节点上全部 4 张卡（未验证是否有 cgroup 隔离，代码里只用 `CUDA_VISIBLE_DEVICES` 给的卡）。**部分节点缺卡**：gpu-05 出现过 CUDA 报错，gpu-07 只认得到 3 张卡（分到第 4 张时 CUDA 不可用），gpu-49 在一次探测里也只看到 3 张 |

对后续实验的直接影响：27B 以上的模型 bf16 权重 55～72 GB，单卡放不下，必须 2～4 张 40GB 卡切分；而多卡作业排队明显更久。

### 1.2 配额与空间

| 文件系统 | 总量 | 已用 | 剩余 | 用户配额 |
|---|---|---|---|---|
| `/home`（`ciai-nfs-c:/c/home`） | 53 T | 44 T（84%） | 8.8 T | `quota -s` 没有返回任何条目，**查不到每用户配额** |
| `/nfs-stor`（`nfs-home-ib:/c/data`） | 581 T | 546 T（94%） | 36 T | 同上，查不到 |
| `/scratch`（各节点本地盘） | 3.6 T | — | 登录节点剩 1.2 T | 节点本地，不跨节点 |

### 1.3 软件

- Lmod 可用。相关 module：`anaconda3`、`nvidia/cuda/11.0～12.0`（默认 12.0）、`gcc-7/8`、`cmake-3.20.0` 等。本项目没有使用任何 module（PyTorch 自带 CUDA 12.8 运行库）。
- conda：`~/miniconda3`（conda 24.11.3，base Python 3.12.7）。已有环境 `actspace`、`actspace-vllm`、`cambrian-s`、`llama3_inf`、`mmcv`、`pySpatial`、`spatialcraft`（transformers 5.5.4 / torch 2.10.0+cu128）等，均未改动。
- uv：系统里没有；已在本项目自己的环境里装了 uv 0.13.10 用于锁定依赖。
- git 与 git-lfs 可用。全局 git 配置里没有 user.name / user.email，我只在本仓库的本地配置里设了提交身份（`Xiwei Liu` + 你的邮箱），没有动全局配置。
- 环境变量里没有 `HF_TOKEN`；`~/.cache/huggingface/` 下也没有 token 文件。本次所有下载都是匿名完成的。

### 1.4 已有权重的来源（`/nfs-stor/xiwei.liu/l/model`，只读，未写入任何文件）

- `MANIFEST.md`（更新于 2026-11-03，11-05 追加）记录了 11 个模型的 HF 仓库、commit、字节数、许可（全部 apache-2.0、非 gated、匿名下载），并说明做过文件大小/哈希校验和“只加载 config + processor”的冒烟。
- **有 6 个目录不在 `MANIFEST.md` 里**：Qwen3.5-2B、SenseNova-SI-1.1-InternVL3-2B、MiniCPM-V-4_5、GLM-4.6V-Flash、RoboBrain2.0-7B、LLaVA-OneVision-7B（2026-11-13 下载）。它们的仓库与 commit 只出现在 `_logs/hf_download_*.out` 里，日志显示各自“核对通过”。RoboBrain 第一次下载失败、第二次成功。
- `README.md` 是更早的 SpatialCraft 说明，里面的路径写的是 `/l/users/xiwei.liu/model/...`，该路径现在不存在（`/l` 为空目录）。
- `MANIFEST.md` 注明 Cambrian-S-7B 是 float32（32 GB）且需要官方代码才能加载；Qwen3.8-27B 与 Qwen3.6-27B 的张量名称/形状完全相同（不代表数值相同）。

### 1.5 旧项目残留搜索结果：**没有找到任何旧项目文件**

在 `/home/xiwei.liu` 和 `/nfs-stor/xiwei.liu` 下只读搜索（跳过了 conda 环境、`site-packages`、`.git`、IDE 服务端目录、两个权重目录、ARKitScenes、ScanNetpp）：

| 搜索 | 命中 | 判断 |
|---|---|---|
| 文件名含 `whatsup` / `whats_up` / `mediation` / `knockout` / `controlled_images` / `controlled_clevr` / `gsr*bench` | **0**（只有本次新下载的 What's Up 文件） | 无残留 |
| 目录名含 `whatsup` / `mediation` / `knockout` / `ablation` / `interp` / `grounding` | **0**（除本项目两个新目录） | 无残留 |
| 文件内容含 `whatsup` / `what's up` / `controlled_images` / `causal mediation` / `attention knockout` / `activation patch`（<3 MB 的 py/md/sh/yaml/json/jsonl/csv/txt/ipynb） | 2 个，都是 `env_backups/spatialcraft-before-qwen-20260907/lib/python3.11/uu.py`（Python 标准库，误命中） | 无关 |
| 文件名含 `ablation` | 约 25 个，全部属于 ActSpace-Bench / ActSpace-release / spatialcraft（如 `ActSpace-Bench/src/actspace_eval/providers/ablations.py`、`/nfs-stor/xiwei.liu/ActSpace-Result/api/_reports/ablations_api.csv` 85 KB，2026-11-13） | 其它项目 |
| `*bbox*.json` / `*mask*.json` / `*boxes*.json` | 6,718 个，合计约 30 MB，几乎全在 `spatialcraftLog/`、`spatialcraftRuns/`、`l/spatialcraftLog/` 的运行产物里（如 `artifact_bounding_boxes_*.json`，0.1～0.9 KB，2026-09） | 其它项目的工具调用产物，不是 What's Up 标注 |
| `*.ipynb` | 50 个，全在第三方仓库里（`tool/repos/sam3`、`GroundingDINO`、`EasyOCR`、`Depth-Anything-3`、`ActSpace-Result/_code/cambrian-s/lmms-eval`） | 无关 |

完整命中清单（路径、大小、修改时间，6,878 行）在会话临时目录里，没有放进仓库；需要的话我可以存到 nfs-stor。结论：**旧的 What's Up 框和 mask 标注不在这两处，需要重新生成。**

搜索中顺带看到、可能有用的已有资源（都没有动）：
- `/nfs-stor/xiwei.liu/tool/checkpoints/`：`sam3`（6.5 G）、`groundingdino`（4.8 G）、`depth_anything_3`（6.8 G）、`moge2`、`orient_anything`、`easyocr`；对应代码在 `tool/repos/`。
- `/nfs-stor/xiwei.liu/ActSpace-Result/_code/cambrian-s/`：Cambrian-S 官方代码的一份拷贝；另有 conda 环境 `cambrian-s`。

### 1.6 两个项目目录的原状

`/home/xiwei.liu/Interpretability` 和 `/nfs-stor/xiwei.liu/interpretability` 在开始时都是**空目录**（分别建于当天 14:23 和 14:24），不是 Git 仓库。

---

## 2. 环境（阶段 B）

- 仓库：`/home/xiwei.liu/Interpretability`，分支 `main`，按任务书的目录结构建立，另加了 `requirements/`（锁文件）、`splits/`（划分）、`manifests/`（数据核对摘要）。仓库当前约 1.5 MB，没有超过 10 MB 的文件。
- 外部路径全部在 `configs/paths.yaml`，代码通过 `src/paths.py` 读取。`src/`、`scripts/`、`tests/` 里没有写死的绝对路径（Slurm 日志目录也由 `scripts/submit.sh` 从该文件读取）。
- nfs-stor 下已建 `models/ data/ cache/ outputs/ envs/`，另加 `logs/`。
- 所有缓存（`HF_HOME`、`TORCH_HOME`、pip、uv、matplotlib）由 `scripts/env.sh` 指向 `cache/`。
- 依赖一个环境满足不了所有模型，建了两个（数据阶段另加了第三个 `tools` 环境，给 GroundingDINO / SAM 用，锁文件 `requirements/tools.lock`）：

| 环境 | 位置 | 关键版本 | 锁文件 | 用于 |
|---|---|---|---|---|
| `tf5` | `/nfs-stor/xiwei.liu/interpretability/envs/tf5`（7.7 G） | Python 3.11.18、torch 2.10.0+cu128、transformers **5.24.1**、accelerate 1.16.0 | `requirements/tf5.lock`（89 个包，带 sha256） | transformers 原生模型：Qwen2.5-VL、Qwen3-VL、Qwen3.5/3.6/3.8、RoboBrain2.0、VST、GLM-4.6V、Gemma-4、LLaVA-1.5、LLaVA-OneVision |
| `tf4` | `/nfs-stor/xiwei.liu/interpretability/envs/tf4`（7.8 G） | Python 3.11.18、torch 2.10.0+cu128、transformers **4.57.6** | `requirements/tf4.lock`（73 个包，带 sha256） | 依赖远程代码的模型：InternVL3 / InternVL3.5 / SenseNova-SI / MiniCPM-V-4_5 |

- 每个模型用哪个环境写在 `configs/adapters.yaml`，`scripts/run_smoke.sh` 按它自动切换。
- 构建方式：conda 建 Python 3.11 前缀 → `uv pip compile --generate-hashes` → `uv pip sync`（`scripts/build_env.sh`）。两个环境都在计算节点上构建，构建后确认 `torch.cuda.is_available() == True`。
- **没有验证过的假设**：我没有尝试让 InternVL / MiniCPM 的远程代码在 transformers 5.x 下运行，直接把它们放进了 4.x 环境；所以“必须两个环境”是出于稳妥而不是实测结论。
- 没有安装 `flash-linear-attention` 和 `causal-conv1d`：Qwen3.5 系的 Gated DeltaNet 层走 transformers 自带的纯 PyTorch 参考实现（日志提示“correct but much slower”）。没有安装 flash-attn（全部用 eager 注意力以便取权重）。
- Cambrian-S-7B 没有环境：它需要官方 cambrian-s 代码库，本阶段未适配（见第 5 节）。

---

## 3. 模型总表（阶段 C + E）

### 3.1 清点字段

23 个模型（17 个已有 + 6 个补充）的完整字段在 `docs/MODEL_INVENTORY.md`，每个字段的出处（文件:行号）在 `configs/models/<名称>.yaml` 的 `sources`，查不到的列在 `unknown`。清单内容来自本地 config、model card、随权重发布的代码和 transformers 源码；由子代理逐个读取填写，我对关键字段（层数、逐层混合机制、head 数）与 `config.json` 做了核对，并且这些字段在冒烟测试中被实际用到（层数对不上会直接失败）。

要点摘要：

| 家族 | 模型 | LM 层 / 维度 / 头 | 逐层序列混合机制 | 一张图的视觉 token | 单视图 | 思考模式 |
|---|---|---|---|---|---|---|
| Qwen2.5-VL 系 | Qwen2.5-VL-7B-Instruct、RoboBrain2.0-7B（空间微调，卡片未写基座）、VST-7B-RL（空间微调，基座只能从 config 推断为 Qwen2.5-VL 7B） | 28 / 3584 / 28 | 全部标准自注意力 | (H'/28)·(W'/28)，行优先 | 本来就是单视图，动态分辨率 | RoboBrain / VST 由提示词触发 |
| Qwen3-VL | Qwen3-VL-8B-Instruct | 36 / 4096 / 32 | 全部标准自注意力；**DeepStack**：ViT 第 8/16/24 块的特征在 LM 第 0/1/2 层之后加到图像 token 上（在层模块之外） | (H'/32)·(W'/32) | 单视图 | 无 |
| Qwen3.5 / 3.6 / 3.8 | 2B、4B、9B、3.6-27B、3.8-27B、3.6-35B-A3B | 24/2048/8、32/2560/16、32/4096/16、64/5120/24、64/5120/24、40/2048/16 | **每 4 层只有 1 层是标准自注意力**（层号 3,7,11,…），其余是线性注意力（Gated DeltaNet）。35B-A3B 的 MLP 是 MoE（256 专家，每 token 8 个 + 1 个共享） | (H'/32)·(W'/32) | 单视图 | 有；`enable_thinking=False` 关闭（2B 默认就是关） |
| InternVL3 系 | InternVL3-2B/8B、SenseNova-SI-1.1-2B（基座 InternVL3-2B）、SenseNova-SI-1.2/1.3-8B（基座 InternVL3-8B） | 28/1536/12、28/3584/28 | 全部标准自注意力 | 每个 448 切片 256 个，16×16 行优先 | 默认 1～12 切片 + 缩略图；只送一张 448×448 即单视图 | 无 |
| InternVL3.5 | InternVL3_5-8B | 36 / 4096 / 32 | 全部标准自注意力 | 同上 | 同上 | 有，默认关（靠 system prompt 开） |
| Gemma-4 | gemma-4-31B-it | 60 / 5376 / 32 | **50 层滑动窗口自注意力（窗口 1024）+ 10 层全局自注意力**（层 5,11,…,59）；图像 token 之间在滑动层里是双向的 | 最多 280，(H'/48)·(W'/48) | 单视图 | 有；`enable_thinking=False` |
| GLM | GLM-4.6V-Flash | 40 / 4096 / 32 | 全部标准自注意力 | (H'/28)·(W'/28) | 单视图 | 有；`enable_thinking=False` |
| MiniCPM | MiniCPM-V-4_5 | 36 / 4096 / 32 | 全部标准自注意力 | 每视图固定 64 个（resampler 的 learned query） | `max_slice_nums=1` | 有；默认关 |
| LLaVA | LLaVA-1.5-7B / 13B | 32/4096/32、40/5120/40 | 全部标准自注意力 | 固定 576（24×24） | 单视图。`llava-hf` 默认是短边缩放到 336 后中心裁剪；**按你的决定改为先用均值色补成正方形**（`pad_to_square`） | 无 |
| LLaVA-OneVision | LLaVA-OneVision-7B | 28 / 3584 / 28 | 全部标准自注意力 | 729（整图）+ anyres 切片 | **没有官方的单视图开关** | 无 |
| Cambrian-S | Cambrian-S-7B（空间微调；基座是纯文本 LLM Qwen2.5-7B-Instruct + SigLIP2） | 28 / 3584 / 28 | config 显示为标准自注意力（未经代码确认） | 查不到 | 查不到 | 无 |

SenseNova-SI 的基座来自各自 README 的 front matter（`base_model: OpenGVLab/InternVL3-2B` / `-8B`）；另一个旁证是 SenseNova-SI-1.1-2B 的 `config.json` 里的 `_commit_hash` 与我下载的 `OpenGVLab/InternVL3-2B` 的 commit `89915501…` 相同。

### 3.2 官方框提示词与坐标约定

**本地 model card 没有一个写了框的提示词或坐标格式。** 下表来自官方网页来源（GitHub cookbook / 官方文档 / 论文），每条的 URL 和原文引用在 `configs/bbox_conventions.yaml`。

| 模型 | 找到官方约定？ | 坐标约定 | 备注 |
|---|---|---|---|
| Qwen2.5-VL-7B | 是 | 缩放后图像上的绝对像素，x1,y1,x2,y2，`bbox_2d` JSON | |
| Qwen3-VL-8B | 是 | 0–1000 归一化，x1,y1,x2,y2 | |
| Qwen3.5-2B/4B/9B、Qwen3.6-27B/35B-A3B | 是，**但依据较弱** | 0–1000 归一化 | 唯一的依据是阿里云百炼 FAQ 里针对 API 模型的一句话，不是针对开源权重；提示词借用 Qwen3-VL cookbook |
| Qwen3.8-27B | **否** | 未知 | 卡片、GitHub、FAQ 都没有 |
| InternVL3-2B/8B | 是 | 0–1000 归一化，`<ref>…</ref>` → `<box>[[x1,y1,x2,y2]]</box>` | |
| InternVL3_5-8B | 是（沿用 InternVL3 的说法，没有 3.5 专门的声明） | 同上 | |
| SenseNova-SI 1.1/1.2/1.3 | **否** | 未知 | 官方只有 v1.4 的一个例子，没给输出格式 |
| RoboBrain2.0-7B | 是，**但依据较弱** | 原图绝对像素（从官方仓库某个旧 commit 的画框代码推断，没有明文） | 仓库主分支已是 RoboBrain 2.5（改为 0–1000） |
| VST-7B-RL | **否** | 未知 | |
| LLaVA-1.5 | 是 | 0–1 归一化，相对“补成正方形”的图 | 依据是仓库作者在 issue 里的回复；而 `llava-hf` 的预处理是中心裁剪而非补边，两者不一致 |
| LLaVA-OneVision | **否**（论文给了提示词，没给坐标格式） | 未知 | |
| GLM-4.6V-Flash | 是 | 0–1000 归一化，`<|begin_of_box|>…<|end_of_box|>` | |
| MiniCPM-V-4_5 | 是 | 0–1000 归一化，`<box>x1 y1 x2 y2</box>` | |
| gemma-4-31B-it | 是 | 0–1000 归一化，**y1,x1,y2,x2**，`box_2d` | |
| Cambrian-S-7B | **否** | 未知 | |

### 3.3 冒烟测试结果（八项）

八项对应任务书阶段 E：1 加载；2 框预测能解析并还原到原图；3 关系四选一能解析并存下四个选项 logit；4 每层隐藏状态；5 标准注意力层的注意力权重 + 单个 head 输出的读取与改写；6 视觉 token 起止位置与图像区域映射；7 干预正确性（无干预逐位一致、自 patch 输出不变、屏蔽后权重为 0）；8 贪心解码两次输出完全相同。

判定是严格的：第 2 项要求“有官方提示词和坐标约定，且 40 次生成全部解析出四个数并还原”；第 3 项要求 20/20；第 7 项用 `torch.equal` 判逐位一致、要求被屏蔽位置的权重精确等于 0。**“通过”不代表框或答案是对的**——没有计算正确率。

<!-- SMOKE_TABLE_BEGIN -->
| 模型 | 组 | 环境 | 1 加载 | 2 框（解析出四个数的次数） | 3 关系（解析出选项的次数） | 4 隐状态 | 5 注意力/head | 6 视觉token映射 | 7 干预 | 8 确定性 | 显存 GB | 秒/样本 | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 1 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.491 | `f7bac9e` |
| Qwen3-VL-8B-Instruct | 1 | tf5 | 通过 | **失败** 29/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 16.33 / 22.72 | 1.866 | `f7bac9e` |
| RoboBrain2.0-7B | 1 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.139 | `f7bac9e` |
| VST-7B-RL | 1 | tf5 | 通过 | **失败** 19/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.45 / 22.97 | 1.489 | `f7bac9e` |
| Qwen3.5-2B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 4.12 / 9.08 | 1.439 | `579f5aa` |
| Qwen3.5-4B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 8.46 / 13.42 | 1.97 | `4a26260` |
| Qwen3.5-9B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 17.53 / 22.49 | 1.924 | `4a26260` |
| Qwen3.6-27B | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| Qwen3.8-27B | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| Qwen3.6-35B-A3B | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| GLM-4.6V-Flash | 1 | tf5 | 通过 | 通过 40/40 | **失败** 12/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 19.17 / 34.44 | 1.803 | `f7bac9e` |
| gemma-4-31B-it | 3 | tf5 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |
| LLaVA-OneVision-7B | 3 | tf5 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | **失败** | 通过 | 14.96 / 38.59 | 1.164 | `4a26260` |
| LLaVA-1.5-7B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 13.16 / 16.25 | 0.567 | `4a26260` |
| LLaVA-1.5-13B | 2 | tf5 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 24.87 / 29.32 | 0.76 | `4a26260` |
| InternVL3-8B | 1 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.712 | `f7bac9e` |
| InternVL3-2B | 2 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 3.89 / 4.97 | 0.618 | `4a26260` |
| InternVL3_5-8B | 1 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 15.89 / 16.97 | 0.953 | `f7bac9e` |
| SenseNova-SI-1.1-InternVL3-2B | 2 | tf4 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 3.89 / 4.97 | 0.786 | `4a26260` |
| SenseNova-SI-1.2-InternVL3-8B | 1 | tf4 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.789 | `f7bac9e` |
| SenseNova-SI-1.3-InternVL3-8B | 1 | tf4 | 通过 | **失败** 40/40 | 通过 20/20 | 通过 | 通过 | 通过 | 通过 | 通过 | 14.8 / 15.88 | 0.804 | `f7bac9e` |
| MiniCPM-V-4_5 | 3 | tf4 | 通过 | 通过 40/40 | 通过 20/20 | 通过 | 通过 | **失败** | 通过 | 通过 | 17.37 / 17.83 | 0.75 | `4a26260` |
| Cambrian-S-7B | excluded | — | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | — | — | — |

## 失败项与说明

- **Qwen3-VL-8B-Instruct** / 2_bbox：官方提示词=True，约定=norm1000_xyxy，解析出四个数 29/40，还原 29/40；
- **VST-7B-RL** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 19/40，还原 0/40；
- **Qwen3.5-2B** / 非标准层：18 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.5-4B** / 非标准层：24 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.5-9B** / 非标准层：24 层 linear_attention（子模块 {'linear_attn': 'Qwen3_5GatedDeltaNet', 'mlp': 'Qwen3_5MLP', 'input_layernorm': 'Qwen3_5RMSNorm', 'post_attention_layernorm': 'Qwen3_5RMSNorm'}）。能：读取/替换该层输出的残差流（activation patching）；hook 混合模块整体的输入输出。不能：没有注意力权重矩阵，不能做位置间注意力屏蔽，也没有 softmax 注意力意义上的 head 级分析。实测自 patch 逐位一致=True。
- **Qwen3.6-27B**：未运行。未运行
- **Qwen3.8-27B**：未运行。未运行
- **Qwen3.6-35B-A3B**：未运行。未运行
- **GLM-4.6V-Flash** / 3_relation：解析出选项 12/20，四个选项 logit 齐全 20/20
- **gemma-4-31B-it**：未运行。未运行
- **LLaVA-OneVision-7B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **LLaVA-OneVision-7B** / 7_interventions：{}
- **LLaVA-OneVision-7B** / 7c_attention_knockout：OutOfMemoryError: CUDA out of memory. Tried to allocate 486.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 170.94 MiB is free. Including non-PyTorch memory, this process has 39.32 GiB memory in use. Of the allocated memory 38.56 GiB is allocated by PyTorch, and 272.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://py
- **SenseNova-SI-1.1-InternVL3-2B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **SenseNova-SI-1.2-InternVL3-8B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **SenseNova-SI-1.3-InternVL3-8B** / 2_bbox：官方提示词=False，约定=unknown，解析出四个数 40/40，还原 0/40；
- **MiniCPM-V-4_5** / 6_visual_token_mapping：视觉 token 没有固定的图像区域对应
- **Cambrian-S-7B**：未运行。用户决定不适配，已从候选中去掉（2026-11-25）；model_type=cambrian_qwen 不在 transformers 中
<!-- SMOKE_TABLE_END -->

原始输出：`/nfs-stor/xiwei.liu/interpretability/outputs/smoke/<模型>/records.jsonl`（每条带模型名、revision、提示词、配置哈希、Git commit）、`summary.json`、`token_region.png`（第 6 项的验证图）。运行日志在 `logs/smoke/<模型>.log`。

干预单元测试在 `tests/test_interventions.py`，默认用随机初始化的 4 层小模型在 CPU 上跑（全仓库目前 63 个测试，全部通过）；`INTERP_TEST_MODEL=<模型名>` 可在真实模型上跑同一套检查。冒烟脚本对每个真实模型调用的是同一组检查函数（`src/checks.py`）。

冒烟测试里做过、需要你知道的处理：
- 所有模型都用 **eager 注意力**、bf16 加载，语言模型的注意力打分和 softmax 在 float32 里计算（`patch_eager_fp32`）。第一轮没有这个修复，eager + bf16 会损坏部分模型的输出（RoboBrain 框输出可解析 15/40，修复后 40/40）；**第 7 项“逐位一致”是同一设置下自己和自己比，抓不到这类问题**，所以另做了 sdpa 对比（DATA_REPORT 1.5）。
- 解码统一为贪心，并显式覆盖了仓库自带的生成参数（RoboBrain 自带 `repetition_penalty=1.05`、`do_sample=true`，被覆盖为 1.0 / 不采样）。
- 动态分辨率的 Qwen 系 / GLM 把 `max_pixels` 限制在 1,003,520（What's Up 原图 1280×960 → 约 1,000～1,560 个视觉 token）。这是冒烟用的设置，不是对正式实验的建议。 **更正（2026-11-25）：该限制对 GLM-4.6V 没有生效**（它的处理器不接受 `max_pixels`），实际是 1288×952、1,564 个视觉 token。四个 27B 以上模型的冒烟：未做，让位给数据作业。
- InternVL 系固定为单视图（整图压到 448×448，256 个 token）。
- LLaVA-OneVision 默认 anyres 在原图上产生 7,309 个视觉 token，eager 注意力在 40 GB 卡上 OOM；改为先把输入等比缩到长边 768 再重跑；重跑后只剩注意力屏蔽一步仍因单卡显存不足失败，没有在 2 张卡上重试。
- 解析器在第一轮后补了三种模型实际输出的包裹格式（`<answer>C</answer>`、`<|begin_of_box|>C<|end_of_box|>`、`<box>x y x y</box>`）。这是让解析器认识真实输出格式，判定标准没有放宽。

---

## 4. 数据总表（阶段 D）

详见 `docs/DATA_MANIFEST.md`。

| 数据 | 数量 | 体积 | 许可 | 核对 |
|---|---|---|---|---|
| What's Up Subset A | 412 张（left/right **206**）；其中 2 张 pillow 因原始数据左右与标签相反被剔除，实际可用 410 张 | 合计 850 MB（含 GQA 图像） | 仓库 MIT；数据无单独许可声明 | 通过，无缺图 |
| What's Up Subset B（front / behind 与垂直位置完全混淆，深度检验只用 SpatialRGPT） | **408** 张 | 同上 | 同上 | 通过，无缺图 |
| COCO-spatial（one / two） | 2,247 / 440 条 | 标注很小；图像见 COCO | 同上 | 通过，引用的图像都在 val2017 里 |
| GQA-spatial（one / two） | 1,160 / 291 条；官方打包图像 4,797 张 | 含在 850 MB 内 | 同上 | 通过，无缺图 |
| COCO val2017 + 实例标注 | 5,000 张 + `instances_val2017.json` | 835 MB | 标注 CC BY 4.0 | 通过（5,000） |
| GSR-Bench 的框 / mask / 深度标注 | — | — | — | **没有找到公开发布** |
| SpatialRGPT-Bench / 3DSRBench / CV-Bench | 只下了说明和标注元数据（6.9 MB） | 完整 2.91 / 1.52 / 0.41 GB | 未声明 / CC BY 4.0 / Apache-2.0 | 已定：只下 SpatialRGPT-Bench 完整数据（仅评测、不再分发），已完成 |
| 背景图 | 未下载 | — | 见清单 | 4 个候选来源 |
| ImageNet val | 未下载 | — | — | 没有官方公共副本；有其他用户的可读副本 |

- 加载器：`src/data/loaders.py`，六个数据集返回同一个 `Sample` 结构（图像路径、target、reference、关系、框、mask）。加载器里 What's Up A/B 的 `Sample` 没有框（标注 v3 用 `src/data/whatsup_annotations.py` 单独读取）；GQA 的框已由场景图补上（DATA_REPORT 3.1）；COCO-spatial 的框和 mask 是我用 COCO 实例标注按类别名唯一匹配得到的（全部样本都能唯一匹配），不是 What's Up 发布的。
- 划分：`splits/*.json`，种子 20261125，按关系分层对半，已提交。**COCO-spatial 的划分已作废**，由按图像的统一划分取代（DATA_REPORT 第 4 节）。
- 抽样图：`/nfs-stor/xiwei.liu/interpretability/outputs/sanity/{whatsup_a,whatsup_b,coco_spatial_one,coco_spatial_two,gqa_spatial_one,gqa_spatial_two}.png`，**请肉眼检查**。我看了 `coco_spatial_two.png`，5 个样本的框与标签一致；其余几张我没有逐张看。

---

## 5. 未完成或失败的事项

第 3.3 节的“失败项与说明”逐条列出了每个失败的原因。结构性的未完成事项（括号里是后续）：

1. **Cambrian-S-7B 完全没有测试。** `model_type=cambrian_qwen` 不在 transformers 里，仓库不带代码，需要官方 cambrian-s 代码库和它自己的依赖环境。本机有一份官方代码拷贝和一个 `cambrian-s` conda 环境（属于其它项目），我没有擅自使用。（你已决定不适配，去掉。）
2. **GSR-Bench 标注没有找到**，What's Up 的框和 mask 需要自己生成（本阶段按要求没有生成）。（已生成并定稿为 v3：自动标注，经模型辅助预审和部分人工复核。）
3. **辅助模型（SAM、LaMa）没有下载**，按“先报告再下载”处理，见第 6 节。（已下载 sam2.1-hiera-large 与 big-lama。）
4. **Gemma-3 没有下载**（gated，需要你接受许可）。（你已决定不下载。）
5. **每用户磁盘配额查不到。**
6. **GQA-spatial 没有框**（需要 GQA scene graph，未下载）。（已下载并补框。）
7. 模型清单里所有标为“查不到”的字段（各 yaml 的 `unknown`），其中最重要的是多个模型的官方框约定（3.2 节）。
8. 第 6 项“视觉 token → 图像区域”的验证方式是画图供人看 + 数量与连续性检查，**没有做基于模型行为的验证**（例如遮挡某区域看哪些 token 变化）。Qwen3.5-2B 的图我看过，网格与图像对齐；其余模型的图需要你抽查。（第一组 9 个模型已补做，见 DATA_REPORT 1.4：没有发现错位的证据，但不能证明精确到单个 token。）
9. **四个 27B 以上模型的冒烟未做**（Qwen3.6-27B、Qwen3.8-27B、Qwen3.6-35B-A3B、gemma-4-31B-it）：让位给数据作业。

---

## 6. 需要你决定的事项

**你的决定（2026-11-25）**：1 不下载 Gemma-3；2 选 SpatialRGPT-Bench；3 不需要 ImageNet，改用 COCO 留出池；6 下载 SAM 2.1 和 LaMa；7 留到有真值框后在 dev 上标定；8 先补成正方形；9、11 你自己处理；10 不适配 Cambrian-S。第 5 条按你给的三组处理（`configs/adapters.yaml` 的 `group`）。**目前仍待决定的事项以 DATA_REPORT 第 8 节为准**，下面是当时的原文。

1. **Gemma-3（4b-it 或 12b-it）**：两个仓库都是 `gated: manual`，许可 `gemma`。体积 8.6 GB / 24.4 GB。需要你在 HF 上接受许可并提供 `HF_TOKEN` 环境变量后我再下。要哪一个，还是都不要？
2. **深度关系数据集三选一**（对比表在 `docs/DATA_MANIFEST.md`）：
   - SpatialRGPT-Bench：唯一自带框**和** mask 的；前/后关系 110 条；2.9 GB；**HF 卡片没有声明许可**；问题是用 “Region [0]” 指代物体的，需要改写。
   - 3DSRBench：`closer to camera` 175 条；CC BY 4.0；**没有框和 mask**；图像大多在 COCO train2017（未下载）。
   - CV-Bench：`Depth` 600 条，带框；Apache-2.0；但**图像上已经画了红/蓝框**，不是干净原图。
3. **ImageNet val 要不要**：集群没有官方公共副本。`/nfs-stor/mingming.gong/imagenet1k/val/` 是他人目录下的可读副本（1000 类，抽查每类 50 张）。要用的话，是引用它（建议先问对方）还是自己下一份（约 6.4 GB，需要你的 image-net.org 账号）？
4. **超出体积阈值的下载**：本次没有。最大单项是 LLaVA-1.5-13B 26.7 GB，新下载模型累计 95 GB、数据 3.4 GB，都在阈值内。
5. **哪些模型不适合做 head 级分析 / 注意力屏蔽**（架构原因，请决定是否保留）：
   - **Qwen3.5-2B/4B/9B、Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B**：四分之三的层是 Gated DeltaNet，没有注意力矩阵，不能做位置间屏蔽，也没有 softmax 意义上的 head；只有层 3,7,11,… 可以。残差流的读取和 patch 在所有层都可以（已在 2B/4B/9B 上实测自 patch 逐位一致）。如果论文的核心论证依赖“逐层的注意力屏蔽”，这一族只能给出不完整的图景。
   - **gemma-4-31B-it**：50/60 层是窗口 1024 的滑动注意力。图像 token 距离答案位置超过 1024 时，这些层根本看不到图像，只有 10 个全局层能直接读；另外图像 token 之间在滑动层是双向注意力。head 级分析技术上可做，但解释时必须考虑窗口。
   - **MiniCPM-V-4_5**：LM 是标准注意力，但视觉 token 是 resampler 的 64 个 learned query，**没有固定的图像区域对应**（第 6 项失败是架构决定的）。凡是需要“把物体所在位置的视觉 token 挑出来”的实验都做不了。
   - **LLaVA-OneVision-7B**：没有官方单视图开关，同一图像区域同时出现在整图视图和切片视图里，视觉 token 的替换实验需要同时处理多份。
   - **Qwen3-VL-8B**：标准注意力，但有 DeepStack（视觉特征在第 0/1/2 层后再次注入），对最前面几层做 patch 时需要考虑。
   - **Cambrian-S-7B**：未适配，无法判断。
   - **Qwen3.6-35B-A3B**：除上述外 MLP 是 MoE；72 GB 权重需要 3～4 张 40 GB 卡。
6. **辅助模型，等你确认后下载**：
   - 分割：`facebook/sam2.1-hiera-large`，1.8 GB，Apache-2.0，非 gated。（`facebook/sam3` 是 gated、许可 `other`、6.9 GB；本机 `tool/checkpoints/sam3` 已有一份 6.5 GB 的权重，属于其它项目；`facebook/sam-vit-huge` 7.7 GB、Apache-2.0。）
   - 修复：`smartywu/big-lama`（LaMa 官方 README 给出的下载地址），0.38 GB，Apache-2.0。
   - 两个都确认的话共 2.2 GB。
7. **没有官方框约定的模型怎么办**：Qwen3.8-27B、SenseNova-SI 三个、VST-7B-RL、LLaVA-OneVision、Cambrian-S。选项：(a) 沿用其基座 / 同族的约定并在论文里说明；(b) 把它们排除出框预测部分。冒烟里观察到 SenseNova-SI 在通用提示词下自发输出 0–1 小数的四元组，但这只是 20 张图上的观察，不足以当作约定。另外 Qwen3.5/3.6 和 RoboBrain2.0 的约定依据较弱（3.2 节），是否接受也请你定。
8. **LLaVA-1.5 的预处理**：`llava-hf` 版本做中心裁剪（What's Up 的 4:3 图会丢掉左右各 12.5%），而原版 LLaVA 是补成正方形。左右关系实验里物体可能被裁掉。要不要改成先补边再送入？
9. **GPU 资源**：只有 40 GB 卡，`long` 分区不可用，批处理队列要等一天以上，debug QoS 单作业 3 小时。27B 级模型的正式实验在这个条件下会很慢。是否要去申请 `gpu-12` QoS 或 80 GB 卡的访问？
10. **Cambrian-S-7B**：要不要为它单独建第三个环境（基于官方代码）？
11. **可清理项（我没有删除任何东西，仅列出）**：`/nfs-stor/xiwei.liu/l/`（568 G）与 `/nfs-stor/xiwei.liu/` 顶层的 `model`（70 G）、`qwen`（54 G）、`tool`（28 G）、`benchmark`、`cache`、`env_backups`、`spatialcraftLog` 是同名的两套目录，不是硬链接；我只比对了 `Qwen3.5-9B` 第一个权重分片的前 100 MB（相同），没有做完整比对。是否重复、删哪边由你决定。本项目自己产生的可清理项：`cache/uv`（7.7 G）和 `cache/conda_pkgs`（0.9 G）在环境建好后可删；`data/_archives/`（1.9 G）是已解压的原始压缩包。

---

## 7. 磁盘占用

| 位置 | 占用 |
|---|---|
| `/home/xiwei.liu/Interpretability`（仓库） | 1.5 MB |
| `/nfs-stor/xiwei.liu/interpretability/models`（6 个补充模型） | 89 G（95.2 GB） |
| `…/envs`（tf5 + tf4） | 16 G |
| `…/cache`（uv 7.7 G、conda_pkgs 0.9 G、其余很小） | 8.7 G |
| `…/data` | 3.4 G |
| `…/outputs`、`…/logs` | 约 10 MB |
| **本项目合计**（本表为准备阶段结束时的数字；最新合计约 131 G，见 DATA_REPORT 第 9 节） | **约 117 G** |
| 已有权重 `/nfs-stor/xiwei.liu/l/model`（未改动） | 421 G |

补充模型明细：Qwen2.5-VL-7B-Instruct 16.6 GB、Qwen3-VL-8B-Instruct 17.5 GB、InternVL3-8B 15.9 GB、InternVL3-2B 4.2 GB、LLaVA-1.5-7B 14.1 GB、LLaVA-1.5-13B 26.7 GB。全部固定到 commit（`configs/downloads/models.yaml`），下载后逐文件核对大小、LFS 文件核对 sha256，全部一致；各目录下有 `SHA256SUMS`。LLaVA-1.5 的许可标签是 `llama2`（非 gated，但不是 Apache/MIT，发布衍生物前请留意）。InternVL3 的卡片 front matter 写 apache-2.0（license_name: qwen），README 正文写 MIT，两处不一致。

---

## 8. 提醒：请配置 Git 远程仓库并推送

仓库只有本地提交，**我没有配置也没有推送任何远程**。请你自己执行：

```bash
cd /home/xiwei.liu/Interpretability
git remote add origin <你的仓库地址>
git push -u origin main
```

推送前可以确认一遍：仓库里没有任何 token 或密钥（HF token 全程没有使用过）；`configs/paths.yaml` 和 `configs/models/*.yaml` 的 `sources` 里含有本集群的绝对路径和你的用户名，如果仓库要公开请留意。
