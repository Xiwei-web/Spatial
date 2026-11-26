"""E1 的共用部分：合成集上的提示词（与 E0 两物体二选一的模板逐字相同）、记录字段、token 换算、test 安全阀。
分析在 src/e1_analysis.py；推理在 scripts/synth_run.py。"""
from __future__ import annotations

import subprocess

import yaml

from src import paths
from src.prompts import ANSWER_LETTER, ANSWER_YES_NO, LETTERS, OPPOSITE, PHRASE_TWO

AXIS_OPTIONS = {"horizontal": ["left", "right"], "vertical": ["above", "below"]}
# 关系 → (轴, 符号)：符号 -1 表示 target 的中心坐标小于 reference（左 / 上）。front / behind 不能由框推出
REL_AXIS = {"left": ("x", -1), "right": ("x", 1), "above": ("y", -1), "on": ("y", -1), "top": ("y", -1), "below": ("y", 1), "under": ("y", 1), "bottom": ("y", 1)}
PREREG = "docs/PREREGISTRATION.md"


def synth_cfg():
    return yaml.safe_load(open(paths.REPO_ROOT / "configs" / "synth" / "synth_v2.yaml"))


def token_key(model: str, cfg=None) -> tuple[str, int]:
    cfg = cfg or synth_cfg()
    for k, v in cfg["token_px"].items():
        if model in v["models"]:
            return k, v["px"]
    raise KeyError(f"{model} 不在 synth_v2.yaml 的 token_px 里（只有入选的 11 个模型）")


def synth_mcq(target: str, reference: str, axis: str, order: int) -> tuple[str, list[str]]:
    """同轴二选一，order 0 / 1 为两种选项顺序。文字与 src.prompts.e0_mcq 的两物体分支相同。"""
    opts = AXIS_OPTIONS[axis][order:] + AXIS_OPTIONS[axis][:order]
    lines = [f"{L}. The {target} is {PHRASE_TWO[o]} the {reference}." for L, o in zip(LETTERS, opts)]
    return f"Look at the image. Where is the {target} relative to the {reference}?\n" + "\n".join(lines) + "\n" + ANSWER_LETTER, opts


def synth_binary(target: str, reference: str, relation: str) -> str:
    return f"Is the {target} {PHRASE_TWO[relation]} the {reference}? {ANSWER_YES_NO}"


def is_test(r: dict) -> bool:
    return r.get("split") == "test" or r.get("side") == "test" or r.get("stage") == "test"


def prereg_committed(repo=None, runner=subprocess.run) -> bool:
    repo = repo or paths.REPO_ROOT
    p = runner(["git", "-C", str(repo), "cat-file", "-e", f"HEAD:{PREREG}"], capture_output=True)
    return p.returncode == 0


def require_prereg_if_test(records, repo=None, runner=subprocess.run) -> None:
    """数据里只要有 test 记录，就要求 git HEAD 里已有 docs/PREREGISTRATION.md，否则报错退出（planner 第 10 轮）。"""
    n = sum(1 for r in records if is_test(r))
    if n and not prereg_committed(repo, runner):
        raise SystemExit(f"数据里有 {n} 条 test 记录，但 git HEAD 里没有 {PREREG}：预注册提交之前不得在 test 上计算 E1 的量")


__all__ = ["AXIS_OPTIONS", "REL_AXIS", "OPPOSITE", "synth_mcq", "synth_binary", "token_key", "require_prereg_if_test", "is_test"]
