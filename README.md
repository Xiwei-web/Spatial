# Interpretability：Grounding 与空间推理（重做）

研究 VLM 中“框预测”和“空间关系回答”是否共用同一个物体位置表示。
本仓库只放代码、配置、脚本、文档；大文件都在 `configs/paths.yaml` 指向的存储上。

当前状态：**实验前准备阶段**（环境、模型清点、数据、冒烟测试）。没有任何正式实验结果。详见 `docs/PREP_REPORT.md`。

## 目录

```
configs/paths.yaml            所有外部路径（代码里不写绝对路径）
configs/models/<name>.yaml    每个模型的事实清单（来自 config / model card / 源码，带出处）
configs/adapters.yaml         每个模型在本仓库里的加载方式、所用环境、GPU 数
configs/bbox_conventions.yaml 各家族官方框提示词与坐标约定（带来源 URL）
configs/downloads/            需要补充下载的模型（固定 commit）
requirements/*.in|*.lock      两个环境的依赖与带哈希的锁文件
splits/                       固定的 dev/test 划分 + 冒烟测试用的 20 张图
src/                          paths / data 加载器 / models 适配层 / hooks / checks / prompts
scripts/                      环境构建、下载、划分、冒烟测试、Slurm 提交
tests/                        单元测试（含干预正确性测试）
docs/                         PREP_REPORT.md / MODEL_INVENTORY.md / DATA_MANIFEST.md
```

## 常用命令

```bash
bash scripts/build_env.sh tf5            # 按锁文件建环境（在计算节点上跑）
source scripts/env.sh tf5                # 激活环境，HF 缓存等全部指向 nfs-stor
python scripts/make_splits.py            # 重新生成划分（已存在则跳过；划分文件已提交）
pytest tests/                            # CPU 上用随机小模型跑干预正确性测试
INTERP_TEST_MODEL=Qwen3.5-2B pytest tests/test_interventions.py   # 在真实模型上跑（需 GPU）
python scripts/smoke_test.py <模型名>     # 冒烟测试，输出到 outputs/smoke/<模型名>/
```

## 规则

- dev / test：所有“选择”类操作只用 dev，所有报告数字只用 test。
- **注意力实现与精度（所有正式实验必须遵守）**：
  - 干预组和它的对照基线必须用同一种注意力实现、同一种精度。
  - 每个模型另外报告一次默认实现下的无干预结果，作为参照。
  - 依据：bf16 下 sdpa 与 eager 的框预测不是逐 token 相同的（`docs/DATA_REPORT.md` 1.5 节）。
- `existing_models` 指向的目录只读。
- HF token 只从环境变量 `HF_TOKEN` 读取，不落盘。
- 不在登录节点跑 GPU 或长时间 CPU 任务。
