"""E0（能力筛选与行为基线）的公共部分：冻结校验、dev 解析检查的样本、逐条答案的推导、指标。CPU 部分都在这里，可单测。
口径见 docs/E0_PLAN.md（“批复与执行口径”一节）。"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess

import numpy as np
import yaml

from src import paths
from src.prompts import (E0_RELATION_OPTIONS, LETTERS, NO_VARIANTS, OPPOSITE, YES_VARIANTS, letter_variants, n_choices, parse_mcq_e0, parse_yes_no)

SEED = 20261125
BBOX_CFG = "configs/bbox_conventions.yaml"
E0_CFG = "configs/e0.yaml"
FROZEN_DIRS = ("src", "scripts", "configs")
FAMILY = {"Qwen2.5-VL-7B-Instruct": "Qwen", "Qwen3-VL-8B-Instruct": "Qwen", "VST-7B-RL": "Qwen", "RoboBrain2.0-7B": "Qwen", "Qwen3.5-2B": "Qwen", "Qwen3.5-4B": "Qwen",
          "Qwen3.5-9B": "Qwen", "InternVL3-8B": "InternVL", "InternVL3_5-8B": "InternVL", "InternVL3-2B": "InternVL", "SenseNova-SI-1.1-InternVL3-2B": "InternVL",
          "SenseNova-SI-1.2-InternVL3-8B": "InternVL", "SenseNova-SI-1.3-InternVL3-8B": "InternVL", "GLM-4.6V-Flash": "GLM",
          "LLaVA-1.5-7B": None, "LLaVA-1.5-13B": None}          # LLaVA-1.5：附录弱基线，不计入家族数
MAX_NEW = {"text16": 16, "text64": 64, "letter_logit": 16}
BOX_MAX_NEW = 96


# ---------------- 冻结 ----------------
def block_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def blocks_from_text(bbox_text: str, e0_text: str | None) -> dict:
    cal = (yaml.safe_load(bbox_text) or {}).get("calibrated")
    modes = (yaml.safe_load(e0_text) or {}).get("answer_modes") if e0_text else None
    return {"calibrated": cal, "answer_modes": modes}


def compare_frozen(current: dict, committed: dict) -> dict:
    """current / committed：blocks_from_text 的结果。不一致就报错；返回两个块的哈希。"""
    for k in ("calibrated", "answer_modes"):
        if block_hash(current[k]) != block_hash(committed[k]):
            raise RuntimeError(f"configs 里的 {k} 与已提交版本不一致：先 commit，再跑 E0")
    if not current["calibrated"]:
        raise RuntimeError("configs/bbox_conventions.yaml 没有 calibrated 块")
    return {"calibrated_sha": block_hash(current["calibrated"]), "answer_modes_sha": block_hash(current["answer_modes"])}


def _git(*a) -> str:
    return subprocess.run(["git", "-C", str(paths.REPO_ROOT), *a], capture_output=True, text=True, check=True).stdout


def check_frozen() -> dict:
    """E0 脚本启动时调用：src/ scripts/ configs/ 下没有未提交改动，且 calibrated、answer_modes 与 HEAD 一致。返回哈希与 commit。"""
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *FROZEN_DIRS).strip()
    if dirty:
        raise RuntimeError("src/ scripts/ configs/ 有未提交的改动，E0 不允许运行：\n" + dirty)
    cur = blocks_from_text(open(paths.REPO_ROOT / BBOX_CFG).read(), open(paths.REPO_ROOT / E0_CFG).read() if (paths.REPO_ROOT / E0_CFG).is_file() else None)
    try:
        e0_head = _git("show", f"HEAD:{E0_CFG}")
    except subprocess.CalledProcessError:
        e0_head = None
    com = blocks_from_text(_git("show", f"HEAD:{BBOX_CFG}"), e0_head)
    return {**compare_frozen(cur, com), "git_commit": _git("rev-parse", "HEAD").strip()}


def calibrated(name: str) -> dict:
    return yaml.safe_load(open(paths.REPO_ROOT / BBOX_CFG))["calibrated"][name]


def answer_modes(name: str) -> dict | None:
    f = paths.REPO_ROOT / E0_CFG
    if not f.is_file():
        return None
    return ((yaml.safe_load(open(f)) or {}).get("answer_modes") or {}).get(name)


# ---------------- 样本 ----------------
PARSE_CHECK_DATASETS = ["whatsup_a", "whatsup_b", "coco_spatial_one", "coco_spatial_two", "gqa_spatial_one", "gqa_spatial_two", "coco_pairs"]


def dev_samples(dataset: str):
    from src.data import load_split
    from src.data.coco_pairs import load_coco_pairs
    S = load_coco_pairs("dev") if dataset == "coco_pairs" else load_split(dataset, "dev")
    return [s for s in S if s.relation in E0_RELATION_OPTIONS[dataset]]


def parse_check_samples(dataset: str, n: int = 50):
    S = sorted(dev_samples(dataset), key=lambda s: s.sample_id)
    return sorted(random.Random(f"{SEED}:e0_parse_check:{dataset}").sample(S, n), key=lambda s: s.sample_id)


def image_key(s) -> str:
    from pathlib import Path
    if s.dataset.startswith("whatsup"):
        return f"{s.dataset}:{s.sample_id}"
    if s.dataset == "coco_pairs":
        return f"coco:{s.extra['image_id']}"
    return f"{'coco' if s.dataset.startswith('coco') else 'vg'}:{int(Path(s.image_path).stem)}"


# ---------------- 首步 logit 的变体 ----------------
def variant_logits(first_logits, tokenizer) -> dict:
    """{变体字符串: logit 或 None}；变体在分词器里不是单个 token 时为 None（之后跳过）。"""
    out = {}
    for v in [x for L in LETTERS for x in letter_variants(L)] + list(YES_VARIANTS) + list(NO_VARIANTS):
        ids = tokenizer.encode(v, add_special_tokens=False)
        out[v] = float(first_logits[ids[0]]) if len(ids) == 1 else None
    return out


def _best(vl: dict, variants) -> float | None:
    xs = [vl[v] for v in variants if vl.get(v) is not None]
    return max(xs) if xs else None


def logit_letter(vl: dict, n: int) -> str | None:
    b = {L: _best(vl, letter_variants(L)) for L in LETTERS[:n]}
    if any(v is None for v in b.values()):
        return None
    return max(LETTERS[:n], key=lambda L: b[L])


def logit_yes_no(vl: dict) -> str | None:
    y, no = _best(vl, YES_VARIANTS), _best(vl, NO_VARIANTS)
    if y is None or no is None:
        return None
    return "yes" if y >= no else "no"


def answer(rec: dict, mode: str):
    """由一条记录推出答案：四/二选一返回字母，二元返回 yes/no；解析不出返回 None。
    text16：只看前 16 个新 token 的解码；text64：完整输出（最多 64 个）；letter_logit：首步变体 logit。"""
    if rec.get("error"):
        return None
    if mode == "letter_logit":
        return logit_letter(rec["variant_logits"], len(rec["options"])) if rec["task"] == "mcq" else logit_yes_no(rec["variant_logits"])
    texts = (rec["text16"], rec["text16_raw"]) if mode == "text16" else (rec["text"], rec["text_raw"])   # 先用去掉特殊 token 的文本（与冒烟相同），再用原始文本
    for text in texts:
        if rec["task"] == "mcq":
            L = parse_mcq_e0(text)
            if L in LETTERS[:len(rec["options"])]:
                return L
        else:
            a = parse_yes_no(text)
            if a is not None:
                return a
    return None


def correct(rec: dict, ans) -> bool:
    if ans is None:
        return False
    if rec["task"] == "mcq":
        return rec["options"][LETTERS.index(ans)] == rec["relation"]
    return (ans == "yes") == (rec["asked_relation"] == rec["relation"])


# ---------------- 统计 ----------------
def bootstrap_mean(values: np.ndarray, images: list[str], n_boot: int = 10000, seed: int = SEED) -> list[float]:
    uniq = sorted(set(images))
    pos = {u: k for k, u in enumerate(uniq)}
    g = np.array([pos[i] for i in images])
    s = np.bincount(g, weights=values.astype(float), minlength=len(uniq))
    n = np.bincount(g, minlength=len(uniq)).astype(float)
    draw = np.random.default_rng(seed).integers(0, len(uniq), (n_boot, len(uniq)))
    m = s[draw].sum(1) / n[draw].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load_records(f) -> dict:
    """JSONL → {key: 最后一条}；key = (stage, dataset, sample_id, task, role, rotation, asked_relation)。"""
    last = {}
    for l in open(f):
        r = json.loads(l)
        last[record_key(r)] = r
    return last


def record_key(r: dict) -> tuple:
    return (r["stage"], r["dataset"], r["sample_id"], r["task"], r.get("role"), r.get("rotation"), r.get("asked_relation"))
