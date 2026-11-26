"""dev 解析检查 → 每个模型每种题型（mcq / binary）的答案获取方式（CPU，只读 JSONL）。按 src/prompts.py 的 ANSWER_MODE_ORDER 与 0.95 这条线自动选。
用法：python scripts/e0_parse_select.py [--write]   --write 时把结果写进 configs/e0.yaml 的 answer_modes（之后要 commit 才能跑筛选）
解析率按题型在 7 个数据集上合并计算（mcq 每模型 350 条，binary 700 条）；另输出逐数据集的解析率供参考。"""
import argparse
import json
from collections import defaultdict

import yaml

from src import e0, paths
from src.prompts import PARSE_RATE_MIN, answer_mode_order, choose_answer_mode

ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
A = ap.parse_args()
res, modes = {}, {}
for name in e0.FAMILY:
    f = paths.get("outputs") / "e0" / name / "parse_check.jsonl"
    if not f.is_file():
        print(name, "没有 parse_check 记录"); continue
    R = [r for r in e0.load_records(f).values() if r["stage"] == "parse_check"]
    assert len(R) == 50 * 7 * 3, (name, len(R))
    res[name], modes[name] = {}, {}
    for task in ("mcq", "binary"):
        rt = [r for r in R if r["task"] == task]
        rates = {m: sum(e0.answer(r, m) is not None for r in rt) / len(rt) for m in ("text16", "text64", "letter_logit")}
        per_ds = defaultdict(dict)
        for m in ("text16", "text64"):
            for ds in e0.PARSE_CHECK_DATASETS:
                x = [r for r in rt if r["dataset"] == ds]
                per_ds[ds][m] = sum(e0.answer(r, m) is not None for r in x) / len(x)
        order = answer_mode_order(name)
        chosen = choose_answer_mode(name, {m: rates[m] for m in order if m != "letter_logit"})
        if chosen == "letter_logit" and rates["letter_logit"] < PARSE_RATE_MIN:
            chosen = None                                         # 字母 / Yes-No 变体在分词器里缺失：全部方式都不到 0.95
        res[name][task] = {"n": len(rt), "parse_rate": rates, "order": order, "chosen": chosen, "per_dataset": dict(per_ds),
                           "n_errors": sum(bool(r.get("error")) for r in rt), "n_hit_64": sum(bool(r.get("hit_max_tokens")) for r in rt)}
        modes[name][task] = chosen
json.dump(res, open(paths.REPO_ROOT / "manifests" / "e0_parse_check.json", "w"), indent=1, ensure_ascii=False)
print("| 模型 | 题型 | n | text16 | text64 | letter_logit | 顺序 | 选中 |\n|" + "---|" * 8)
for m, d in res.items():
    for t, x in d.items():
        p = x["parse_rate"]
        print(f"| {m} | {t} | {x['n']} | {p['text16']:.3f} | {p['text64']:.3f} | {p['letter_logit']:.3f} | {' → '.join(x['order'])} | {x['chosen'] or '**全部不到 0.95**'} |")
if A.write:
    f = paths.REPO_ROOT / e0.E0_CFG
    cfg = yaml.safe_load(open(f)) if f.is_file() else {}
    cfg["answer_modes"] = {m: {t: v for t, v in d.items()} for m, d in modes.items() if all(d.values())}
    cfg["answer_modes_note"] = "dev 解析检查（manifests/e0_parse_check.json）按 ANSWER_MODE_ORDER 与 0.95 自动选出；全部方式都不到 0.95 的模型不列入（不跑筛选）"
    yaml.safe_dump(cfg, open(f, "w"), allow_unicode=True, sort_keys=False)
    print("written", f)
