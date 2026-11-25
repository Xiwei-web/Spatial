"""用已保存的 JSON 结果刷新 docs/DATA_REPORT.md 里成对标记之间的表（CPU，可重复运行；不重算任何结果、不含正确率）。
用法：python scripts/build_smoke_table.py && python scripts/fill_data_report.py"""
import json

import numpy as np
import yaml

from src import paths
from src.prompts import parse_four_numbers

O = paths.get("outputs")
G1 = [n for n, s in yaml.safe_load(open(paths.REPO_ROOT / "configs" / "adapters.yaml"))["models"].items() if s.get("group") == 1]
REP = paths.REPO_ROOT / "docs" / "DATA_REPORT.md"


def smoke():
    t = (paths.REPO_ROOT / "docs" / "SMOKE_RESULTS.md").read_text()
    return t[t.index("| 模型 |"):].replace("## 失败项与说明", "**失败项与说明**")


def mapping():
    out = ["| 模型 | 网格 | 重合率（均值 / 最低） | 最大变化 token 落在与矩形相交格子内的比例 | 预测图与实测图相关系数 | 质心距离（token，均值 / 最大） | 变化最大的单个 token 在矩形内 | 翻转对照重合率（左右 / 上下 / 两者） | 矩形内外平均变化量之比 | 第 1 / 2 / 3 层后的重合率 |",
           "|" + "---|" * 10]
    for m in G1:
        d = json.load(open(O / "token_mapping_verify" / m / "result.json"))
        g, c, a = d["images"][0]["grid_hw"], d["mean_control_overlap"], d["mean_overlap_after_layer"]
        out.append(f"| {m} | {g[0]}×{g[1]} | {d['mean_overlap_hs0']:.2f} / {d['min_overlap_hs0']:.2f} | {d['mean_top_in_any_intersect_hs0']:.2f} | {d['mean_corr_pred_vs_measured']:.2f} | "
                   f"{d['mean_centroid_error_tokens']:.1f} / {d['max_centroid_error_tokens']:.1f} | {d['n_argmax_inside_rect']}/5 | {c['hflip']:.2f} / {c['vflip']:.2f} / {c['hvflip']:.2f} | "
                   f"{d['mean_change_ratio_in_vs_out']:.1f} | {a['1']:.2f} / {a['2']:.2f} / {a['3']:.2f} |")
    return "\n".join(out)


def impl():
    t1 = ["| 模型 | 框提示逐 token 一致 | 关系提示逐 token 一致 | 四个选项 logit 最大差 | 选项 logit 平均差 | 四选项第一名一致 |", "|" + "---|" * 6]
    t2 = ["| 模型 | 框提示不一致条数 | 两边都能解析 | 坐标最大差：中位数 | 坐标最大差：最大值 | 相差 ≥ 50 的条数 |", "|" + "---|" * 6]
    for m in G1:
        d = json.load(open(O / "smoke" / m / "default_vs_eager.json"))
        t1.append(f"| {m} | {d['bbox_token_identical']}/40 | {d['relation_token_identical']}/20 | {d['option_logit_max_abs_diff']:.3f} | {d['option_logit_mean_abs_diff']:.3f} | {d['option_argmax_identical']}/20 |")
        df = []
        for a, b in zip(d["runs"]["observed"], d["runs"]["reference"]):
            if a["task"] == "bbox" and a["tokens"] != b["tokens"]:
                x, y = parse_four_numbers(a["text"]), parse_four_numbers(b["text"])
                if x and y:
                    df.append(max(abs(i - j) for i, j in zip(x, y)))
        t2.append(f"| {m} | {40 - d['bbox_token_identical']} | {len(df)} | {np.median(df):g} | {max(df):g} | {sum(v >= 50 for v in df)} |")
    return "\n".join(t1) + "\n\n坐标差按各模型原始输出的数字计（单位是各自的输出约定）：\n\n" + "\n".join(t2)


def coverage():
    d = json.load(open(paths.REPO_ROOT / "manifests" / "whatsup_token_coverage.json"))
    f = lambda v: f"{v['median']:g} ({v['p5']:g}–{v['p95']:g})"
    out = [f"标注：{d['annotation']}；下表用全部 {d['n_images_with_both_masks']} 张（质检通过子集、A / B 分开的数字在 JSON 里）。格式：中位数（5%–95% 分位）。", "",
           "| 模型 | 设置 | 网格 | 视觉 token 数 | target 覆盖（有重叠） | target 覆盖（≥50% 面积） | reference 覆盖（有重叠） | reference 覆盖（≥50% 面积） | 两质心距离（token） | target 没有任何 ≥50% token 的比例 |",
           "|" + "---|" * 10]
    for m, st in d["models"].items():
        for k, v in sorted(st.items(), key=lambda kv: kv[1]["n_visual_tokens"]):
            a = v["all"]
            out.append(f"| {m} | {k} | {v['grid_hw'][0]}×{v['grid_hw'][1]} | {v['n_visual_tokens']} | {f(a['target_tokens_any_overlap'])} | {f(a['target_tokens_ge50pct'])} | "
                       f"{f(a['reference_tokens_any_overlap'])} | {f(a['reference_tokens_ge50pct'])} | {f(a['centroid_distance_tokens'])} | {a['share_target_with_zero_ge50pct_tokens']:.1%} |")
    return "\n".join(out)


t = REP.read_text()
for tag, fn in (("SMOKE_TABLE", smoke), ("MAPPING_TABLE", mapping), ("IMPL_TABLE", impl), ("COVERAGE_TABLE", coverage)):
    a, b = f"<!-- {tag}_BEGIN -->", f"<!-- {tag}_END -->"
    assert t.count(a) == 1 and t.count(b) == 1, tag
    t = t[: t.index(a) + len(a)] + "\n" + fn().strip() + "\n" + t[t.index(b):]
REP.write_text(t)
print("ok")
