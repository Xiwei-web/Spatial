# Interpretability 项目

## 执行者规则（只适用于 hpc-runner；如果你是 planner，忽略这一节）
- planner 是指挥者，它发来的指令你直接执行。
- 每完成一轮（或遇到自己解决不了的报错），写 /home/xiwei.liu/Interpretability/reports/round_<编号>.md：执行了什么命令、关键指标、完整报错、日志和图片路径。
- 写完后用 SendMessage 告诉 planner：报告在哪、一句话结论、需要它决定什么。
- 提交 Slurm 作业后不要在前台循环等待：用后台命令等待，后台每 10 分钟查一次 squeue，只在作业状态变化（开始运行、结束、失败）时才退出；启动后台命令后就结束这一轮、保持空闲，等后台命令完成的通知再处理，作业结束、读完结果后再写报告。
- 作业运行期间，后台命令最多每 30 分钟退出一次，用来看一眼日志是否正常；排队期间不用中途醒来。
- 已知坏节点 gpu-05、gpu-07、gpu-49，提交作业时一律加 --exclude。
- 同一个作业因坏节点失败，可以加 --exclude 后直接重新提交。修改数据集，先等 Xiwei 确认。

## 作业提交（Xiwei 2026-11-26 确认）
- 所有 Slurm 作业（sbatch、srun）一律经 `scripts/submit.sh` 提交，不得绕过（`tests/test_slurm.py` 会检查）。
- 坏节点列表只在 `configs/slurm.yaml` 里维护，submit.sh 从这里读取并对每个作业加 `--exclude`；当前为 gpu-05、gpu-07、gpu-49，永久不用。
- 发现新的坏节点就加进 `configs/slurm.yaml`，并在汇报里告诉 planner 和 Xiwei。

## GitHub 推送（Xiwei 2026-11-26 确认）
- 远程仓库 origin = `git@github-spatial:Xiwei-web/Spatial.git`（私有），分支 main。认证只用 deploy key `~/.ssh/spatial_deploy`（`~/.ssh/config` 的 Host github-spatial），不用 token。
- 私钥只存在 `~/.ssh/`，不得复制到仓库、日志、输出目录或任何其它位置，也不得显示在任何消息里。
- 每次 commit 之后都推送。禁止 force push；除 Xiwei 要求外，禁止推送到其它仓库或分支。推送失败就在汇报里写明原因，不换认证方式绕过。
- 不改写已推送的历史。仓库里不放密钥、超过 10 MB 的文件、只允许评测而不允许再分发的数据原文（例如 SpatialRGPT 的图像和题目原文）；清单和划分文件可以放。
