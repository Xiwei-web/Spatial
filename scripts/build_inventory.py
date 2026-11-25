"""由 configs/models/*.yaml 生成 docs/MODEL_INVENTORY.md（总表 + 每个模型的完整字段）。
清单内容全部来自 yaml（yaml 的每个字段带出处）；本脚本不添加任何事实。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from src import paths

CFG = paths.REPO_ROOT / "configs"


def short(v, n=90):
    s = "—" if v is None else " ".join(str(v).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def mixer_pattern(m):
    c = Counter(m)
    if len(c) == 1:
        return f"全部 {m[0]}（{len(m)} 层）"
    parts = []
    for t, k in c.most_common():
        idx = [i for i, x in enumerate(m) if x == t]
        parts.append(f"{t}×{k}" + (f"（层 {','.join(map(str, idx))}）" if k <= 16 else ""))
    return "；".join(parts)


def main():
    adapters = yaml.safe_load(open(CFG / "adapters.yaml"))
    files = sorted(f for f in (CFG / "models").glob("*.yaml") if not f.name.startswith("_"))
    out = ["# 模型清单（MODEL_INVENTORY）", "",
           "由 `scripts/build_inventory.py` 从 `configs/models/*.yaml` 生成，请勿手改。",
           "每个字段的出处见对应 yaml 的 `sources`；查不到的字段为 “—”，并列在 yaml 的 `unknown` 里。", "",
           "## 总表", "",
           "| 模型 | HF repo @ revision | 体积 GB | 许可 | LM 层/维度/头 | 逐层混合机制 | 视觉 token/图 | 角色（基座） | 环境 |",
           "|---|---|---|---|---|---|---|---|---|"]
    docs = []
    for f in files:
        d = yaml.safe_load(open(f))
        lm, name = d["language_model"], d["name"]
        ad = {**adapters["defaults"], **adapters["models"].get(name, {})}
        role = d.get("role") or "—"
        if d.get("base_model"):
            role += f"（{short(d['base_model'], 40)}）"
        size = f"{d['size_bytes'] / 1e9:.1f}" if d.get("size_bytes") else "—"
        out.append(f"| {name} | {d.get('hf_repo')} @ `{str(d.get('revision'))[:8]}` | {size} | {d.get('license')} | "
                   f"{lm.get('num_layers')}/{lm.get('hidden_size')}/{lm.get('num_attention_heads')} | {mixer_pattern(lm.get('layer_mixers') or [])} | "
                   f"{short((d.get('connector') or {}).get('tokens_per_image'), 60)} | {role} | {ad.get('env') if ad.get('kind') != 'unsupported' else '未适配'} |")
        docs.append((name, d))
    out += ["", "## 逐模型明细", ""]
    for name, d in docs:
        out += [f"### {name}", ""]
        for sec in ("hf_repo", "revision", "revision_source", "size_bytes", "license", "license_source", "role", "base_model"):
            out.append(f"- **{sec}**: {short(d.get(sec), 600)}")
        for sec in ("loader", "language_model", "vision_encoder", "connector", "multi_view", "bbox", "thinking"):
            out += ["", f"**{sec}**", ""]
            for k, v in (d.get(sec) or {}).items():
                if k == "layer_mixers":
                    out.append(f"- layer_mixers（逐层，0 起）: `{' '.join({'full_attention': 'F', 'linear_attention': 'L', 'sliding_attention': 'S'}.get(x, '?') for x in v)}`"
                               f"  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）")
                else:
                    out.append(f"- {k}: {short(v, 1200)}")
        if d.get("unknown"):
            out += ["", "**查不到 / 未确认**", ""] + [f"- {short(u, 500)}" for u in d["unknown"]]
        out.append("")
    (paths.REPO_ROOT / "docs" / "MODEL_INVENTORY.md").write_text("\n".join(out) + "\n")
    print("wrote docs/MODEL_INVENTORY.md", len(docs), "models")


if __name__ == "__main__":
    main()
