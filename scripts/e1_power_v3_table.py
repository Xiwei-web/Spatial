"""功效模拟第三版的汇总表（CPU，只读 manifests/e1_power_sim_v3.json 与 manifests/e1_pilot_nuisance.json）→ 打印 markdown。
表 1：试跑讨厌参数；表 2：合成集（主集两种噪声、截尾版、重叠子集及分档）；表 3：真实数据集（二选一 / 成对二元 × 原判定 / A × H1 / H3）。
用法：python scripts/e1_power_v3_table.py > <输出>.md"""
import json

from src import e1, paths

MODELS = e1.synth_cfg()["single_object_check"]["models"]
P = json.load(open(paths.REPO_ROOT / "manifests" / "e1_power_sim_v3.json"))["scenarios"]
N = json.load(open(paths.REPO_ROOT / "manifests" / "e1_pilot_nuisance.json"))["models"]
f2 = lambda v: "—" if v is None else f"{v:.2f}"      # noqa: E731
SHORT = {"Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "SenseNova-SI-1.2-InternVL3-8B": "SN-1.2", "SenseNova-SI-1.3-InternVL3-8B": "SN-1.3"}
sh = lambda m: SHORT.get(m, m)      # noqa: E731


def g(sc, m, ag, key):
    try:
        return P[sc][m]["summary"][ag].get(key)
    except KeyError:
        return None


print("### 表 1 试跑讨厌参数（dev，200 个基础场景；只含 L 的尺度与噪声、框误差、边际正确率）\n")
print("| 模型 | 框解析失败 | 二选一 L 中位数 [1%, 99%] | |L|>10 | L<0 | m(Δ) 范围 | κ_loc 中位数（/token） | ν（顺序差） | 成对二元 L 中位数 | ν（二元） | 直线拟合 κ / ν（sd 版，不用） | 框误差 MAD 中位数（token） |")
print("|" + "---|" * 12)
for m in MODELS:
    if m not in N:
        print(f"| {sh(m)} | 缺 |" + " |" * 10); continue
    n = N[m]
    lm, lb = n["L_local"]["mcq"], n["L_local"].get("binary", {})
    q = n["L_mcq_quantiles"]
    lf = n["L_fit"]["sd"]["mcq_mean_of_orders"]
    mads = sorted(v["mad_e"] for v in n["box_error_stats"].values() if v["mad_e"] is not None)
    print(f"| {sh(m)} | {n['box_parse_fail']:.3f} | {q[3]:.1f} [{q[1]:.1f}, {q[5]:.1f}] | {n['frac_L_mcq_abs_gt10']:.2f} | {n['frac_L_mcq_negative']:.3f} | "
          f"{lm['m_quantiles'][0]:.1f}–{lm['m_quantiles'][4]:.1f} | {lm['kappa_loc_per_token_quantiles'][2]:.2f} | {lm['nu']:.2f} | {n['L_binary_quantiles'][3]:.1f} | "
          f"{f2(lb.get('nu'))} | {lf['kappa_through_origin']:.3f} / {lf['nu']:.1f} | {mads[len(mads) // 2]:.3f} |")

print("\n### 表 1b 试跑的边际正确率（按该模型的 token 档；框推 / 二选一 / 成对二元）\n")
bins = sorted({int(b) for m in N for b in N[m]["marginal_by_token_bin"]})
print("| 模型 | " + " | ".join(f"档 {b}" for b in bins) + " |\n|" + "---|" * (len(bins) + 1))
for m in MODELS:
    if m not in N:
        continue
    mb = N[m]["marginal_by_token_bin"]
    cells = []
    for b in bins:
        v = mb.get(str(b))
        cells.append("—" if v is None else f"{v['box_acc']:.2f} / {v['mcq_acc']:.2f} / {v['pair_acc']:.2f}")
    print(f"| {sh(m)} | " + " | ".join(cells) + " |")
print("\n档 −1 = 超出该模型 9 档范围（28 px 模型 > 11.3 token 的场景），正式分析不用。")

print("\n### 表 2 合成集：各方案判对的概率（100 次重复）\n")
cols = [("synth_main_so", "主集·单物体噪声 κ=3 ν=0.5"), ("synth_main_pilot", "主集·试跑参数"), ("synth_overlap_pilot", "重叠子集·试跑参数")]
print("| 模型 | 排除尺寸 | 情景 | n(test) | 主判定 H1 / H2 / H3 | A H1 / H2 / H3 | A 等价 H3 | B H1 / H3 | C H1 / H3 | 截尾池 A H1 / H3 |")
print("|" + "---|" * 10)
for m in MODELS:
    for sc, lab in cols:
        if m not in P.get(sc, {}):
            continue
        d = P[sc][m]
        wins = ""
        if sc == "synth_main_pilot" and m in P.get("synth_main_pilot_wins", {}):
            wins = f"{f2(g('synth_main_pilot_wins', m, 'H1', 'p_correct_A'))} / {f2(g('synth_main_pilot_wins', m, 'H3', 'p_correct_A'))}"
        print(f"| {sh(m)} | {d['info']['failed_sizes_excluded'] or '—'} | {lab} | {d['info']['n_test_items']} | "
              f"{f2(g(sc, m, 'H1', 'p_correct_main'))} / {f2(g(sc, m, 'H2', 'p_correct_main'))} / {f2(g(sc, m, 'H3', 'p_correct_main'))} | "
              f"{f2(g(sc, m, 'H1', 'p_correct_A'))} / {f2(g(sc, m, 'H2', 'p_correct_A'))} / {f2(g(sc, m, 'H3', 'p_correct_A'))} | {f2(g(sc, m, 'H3', 'p_correct_A_equiv'))} | "
              f"{f2(g(sc, m, 'H1', 'p_correct_B'))} / {f2(g(sc, m, 'H3', 'p_correct_B'))} | {f2(g(sc, m, 'H1', 'p_correct_C'))} / {f2(g(sc, m, 'H3', 'p_correct_C'))} | {wins} |")

print("\n### 表 2b 重叠子集分遮挡档：方案 A 判对的概率 H1 / H3（主判定 H1 / H3）\n")
tiers = sorted({t for m in P.get("synth_overlap_pilot", {}) for t in P["synth_overlap_pilot"][m]["summary"].get("by_tier", {})})
print("| 模型 | " + " | ".join(tiers) + " |\n|" + "---|" * (len(tiers) + 1))
for m in MODELS:
    if m not in P.get("synth_overlap_pilot", {}):
        continue
    bt = P["synth_overlap_pilot"][m]["summary"].get("by_tier", {})
    cells = []
    for t in tiers:
        v = bt.get(t)
        cells.append("—" if not v else f"A {f2(v['H1'].get('p_correct_A'))} / {f2(v['H3'].get('p_correct_A'))}（主 {f2(v['H1'].get('p_correct_main'))} / {f2(v['H3'].get('p_correct_main'))}）")
    print(f"| {sh(m)} | " + " | ".join(cells) + " |")

print("\n### 表 3 真实数据集：判对的概率（100 次重复；原判定 = 按 dev 三分位距离档分层的 MH OR）\n")
DS = [("whatsup_a", "WU-A"), ("whatsup_b", "WU-B"), ("coco_spatial_two", "COCO-2"), ("gqa_spatial_two", "GQA-2")]
for suffix, lab in (("", "直接回答 = 二选一（两个顺序）"), ("_pair", "直接回答 = 成对二元")):
    print(f"\n**{lab}**；每格：原判定 H1 / H3 ‖ A H1 / H3\n")
    print("| 模型 | " + " | ".join(f"{d[1]}" for d in DS) + " |\n|" + "---|" * (len(DS) + 1))
    for m in MODELS:
        cells = []
        for ds, _ in DS:
            sc = f"real_{ds}{suffix}"
            if m not in P.get(sc, {}):
                cells.append("—"); continue
            cells.append(f"{f2(g(sc, m, 'H1', 'p_correct_main_or'))} / {f2(g(sc, m, 'H3', 'p_correct_main_or'))} ‖ {f2(g(sc, m, 'H1', 'p_correct_A'))} / {f2(g(sc, m, 'H3', 'p_correct_A'))}")
        print(f"| {sh(m)} | " + " | ".join(cells) + " |")
    print(f"\n{lab}：校准与边际量（dev 框推正确率；H1 / H3 的 τ 是否对上 dev 的直接回答正确率；模拟里 test 的框推 / 直接回答正确率中位数）\n")
    print("| 模型 | " + " | ".join(f"{d[1]}" for d in DS) + " |\n|" + "---|" * (len(DS) + 1))
    for m in MODELS:
        cells = []
        for ds, _ in DS:
            sc = f"real_{ds}{suffix}"
            if m not in P.get(sc, {}):
                cells.append("—"); continue
            inf = P[sc][m]["info"]
            ok = "✓" if inf["tau_H1"]["matched"] and inf["tau_H3"]["matched"] else ("H1✗" if not inf["tau_H1"]["matched"] else "H3✗")
            cells.append(f"框 {f2(inf['p_box_dev'])}，目标 {inf['tau_H1']['target_dev']:.2f} {ok}，模拟 {f2(g(sc, m, 'H1', 'median_box_acc'))} / {f2(g(sc, m, 'H1', 'median_direct_acc'))}")
        print(f"| {sh(m)} | " + " | ".join(cells) + " |")
