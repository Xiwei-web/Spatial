"""把 outputs/smoke/*/summary.json 汇总成 docs/SMOKE_RESULTS.md（每个模型一行 × 八项测试）。
只报告通过/失败与资源占用，不含任何正确率。"""
import json

import yaml

from src import paths

EIGHT = ["1_load", "2_bbox", "3_relation", "4_hidden_states", "5_attention_and_heads", "6_visual_token_mapping",
         "7_interventions", "8_determinism"]
HEAD = ["1 加载", "2 框", "3 关系", "4 隐状态", "5 注意力/head", "6 视觉token映射", "7 干预", "8 确定性"]
ad = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "adapters.yaml"))
root = paths.get("outputs") / "smoke"
rows, notes = [], []
for name, spec in ad["models"].items():
    f = root / name / "summary.json"
    env = spec.get("env", ad["defaults"]["env"]) if spec.get("kind") != "unsupported" else "—"
    if not f.is_file():
        why = spec.get("reason", "未运行")
        rows.append(f"| {name} | {env} | " + " | ".join(["未跑"] * 8) + " | — | — | — |")
        notes.append(f"- **{name}**：未运行。{why}")
        continue
    s = json.load(open(f))
    t = s["tests"]
    cells = []
    for k in EIGHT:
        r = t.get(k)
        cells.append("未跑" if r is None else ("通过" if r.get("pass") else "**失败**"))
    l = t.get("1_load", {})
    rows.append(f"| {name} | {s.get('env')} | " + " | ".join(cells) +
                f" | {l.get('gpu_mem_after_load_gb', '—')} / {s.get('gpu_mem_peak_gb_overall', l.get('gpu_mem_peak_gb', '—'))} | {l.get('seconds_per_generation_mean', '—')} | `{str(s.get('git_commit'))[:7]}` |")
    for k in EIGHT + ["5a_attention_weights", "5b_head_hook", "7a_noop_hooks_bitwise", "7b_self_patch", "7c_attention_knockout"]:
        r = t.get(k)
        if r is not None and not r.get("pass"):
            why = r.get("error") or r.get("reason") or r.get("why") or ""
            if k == "2_bbox":
                why = (f"官方提示词={r.get('official_prompt')}，约定={r.get('convention')}，解析出四个数 {r.get('n_parsed_four_numbers')}/{r.get('n_generated')}，"
                       f"还原 {r.get('n_restored_to_original')}/{r.get('n_generated')}；{(r.get('errors') or [''])[0]}")
            if k == "3_relation":
                why = f"解析出选项 {r.get('n_parsed_option')}/{r.get('n_generated')}，四个选项 logit 齐全 {r.get('n_with_four_option_logits')}/{r.get('n_generated')}"
            if not why:
                why = json.dumps({a: b for a, b in r.items() if a not in ("pass", "traceback", "layers", "criterion")}, ensure_ascii=False, default=str)[:400]
            notes.append(f"- **{name}** / {k}：{str(why)[:500]}")
    ns = t.get("5c_nonstandard_layers", {})
    if ns.get("has_nonstandard"):
        notes.append(f"- **{name}** / 非标准层：{ns['n_nonstandard']} 层 {ns['type']}（子模块 {ns['children']}）。能：{ns['can']}。不能：{ns['cannot']}。"
                     f"实测自 patch 逐位一致={ns['self_patch_identical']}。")
out = ["# 冒烟测试结果（SMOKE_RESULTS）", "",
       "由 `scripts/build_smoke_table.py` 从 `outputs/smoke/<模型>/summary.json` 生成。**只表示管线是否跑通，不是评测；不含任何正确率，不得用于比较模型。**", "",
       "显存列为“加载后 / 全程峰值”GB（各卡之和，bf16，eager 注意力）；耗时为 60 次生成（40 次框 + 20 次关系）的平均秒数，含预处理。", "",
       "| 模型 | 环境 | " + " | ".join(HEAD) + " | 显存 GB | 秒/样本 | commit |", "|---|---|" + "---|" * 11] + rows
out += ["", "## 失败项与说明", ""] + (notes or ["（无）"])
(paths.REPO_ROOT / "docs" / "SMOKE_RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(rows))
