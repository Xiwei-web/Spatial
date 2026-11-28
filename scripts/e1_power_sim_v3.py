"""E1 功效模拟第三版（planner 第 11 轮第 4–7 条）。以后所有功效模拟都按“排除单物体检查不通过的尺寸档”之后的条目算。
情景（--scenario）：
  synth_main_so     合成 v2 主集 test 设计；噪声 = 单物体检查的有向框中心误差（按 轴 × 尺寸档 分池）；L：κ = 3、ν = 0.5（同第二版），只是排除了不通过的尺寸档
  synth_main_pilot  同上设计；噪声 = dev 试跑两物体场景的框误差（轴 × 尺寸档；某池 < 30 个就并为该轴全部尺寸）；
                    L = 试跑估计的饱和局部线性模型（manifests/e1_pilot_nuisance.json 的 L_local.mcq；src/e1_sim.local_L）
  synth_main_pilot_wins 同上，但误差池截到 中位数 ± 3·稳健 sd（1.4826·MAD），看离群值的影响；只算方案 A
                    （局部线性模型里 L 不再除以 sd(e_t − e_r)，“改用稳健尺度”落在误差池上；见 round_11.md）
  synth_overlap_pilot   合成 v2 重叠子集 test 设计（三档遮挡合并 + 每档单独），噪声与 κ、ν 同 synth_main_pilot
  real_<数据集>_pair 同下，但直接回答换成成对二元（一次观测；对上 dev 的成对二元正确率）。test 上两物体的成对二元明显低于 1（Table 1），选择题不低
  real_<数据集>      What's Up A / B、COCO-two、GQA-two：dev 边际量（scripts/e1_real_inputs.py），test 样本量；真值 H1 / H3；
                     原判定 = 按 dev 三分位距离档分层的 MH OR（95% CI 下界 > 2 → 相关；CI 含 1 → 与独立相容）；方案 A 同合成集
每个（模型, 情景, 真值, 重复）一行追加到进度文件，断点续跑。内部 bootstrap --boot 次（默认 300；正式分析 10,000 次）。
用法：python -m scripts.e1_power_sim_v3 --scenario synth_main_so --models A,B --reps 100 --workers 8
      python -m scripts.e1_power_sim_v3 --merge   → manifests/e1_power_sim_v3.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse      # noqa: E402
import json      # noqa: E402
import time      # noqa: E402
from multiprocessing import Pool      # noqa: E402

import numpy as np      # noqa: E402
import yaml      # noqa: E402

from src import e1, paths      # noqa: E402
from src import e1_alt as X      # noqa: E402
from src import e1_sim as S      # noqa: E402
from src.e1_analysis import mh_or      # noqa: E402
from scripts.e1_recovery_alt import jsonable      # noqa: E402

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "e1.yaml"))
SCFG = e1.synth_cfg()
OUT = paths.get("outputs") / "e1_power_v3"
MODELS = SCFG["single_object_check"]["models"]
SO = json.load(open(paths.REPO_ROOT / "manifests" / "single_object_check.json"))
NUIS_F = paths.REPO_ROOT / "manifests" / "e1_pilot_nuisance.json"
REAL = ["whatsup_a", "whatsup_b", "coco_spatial_two", "gqa_spatial_two"]
SCEN = ["synth_main_so", "synth_main_pilot", "synth_main_pilot_wins", "synth_overlap_pilot"] + [f"real_{d}" for d in REAL] + [f"real_{d}_pair" for d in REAL]
MIN_POOL = 30


def failed_sizes(model):
    return sorted(int(s) for s, v in SO["models"][model].items() if not v["pass"])


def so_pools(model):
    P = {}
    for s, v in SO["models"][model].items():
        P[("x", int(s))] = np.array([e["ex_tok"] for e in v["signed_center_err_tok"]])
        P[("y", int(s))] = np.array([e["ey_tok"] for e in v["signed_center_err_tok"]])
    return P


def nuis(model):
    return json.load(open(NUIS_F))["models"][model]


def pilot_pools(model, wins=False):
    """{(轴, 尺寸): 误差}；某池 < MIN_POOL 个时用该轴全部尺寸合并的池。wins：截尾到 中位数 ± 3·稳健 sd。"""
    raw = nuis(model)["box_error_pools"]
    if wins:
        raw = {k: list(S.winsorize_pool(v)) if len(v) else v for k, v in raw.items()}
    by_axis = {a: np.concatenate([np.array(v) for k, v in raw.items() if k.startswith(a + "|")]) for a in ("x", "y")}
    P, merged = {}, []
    for a in ("x", "y"):
        for s in (36, 72, 108, 161):
            v = np.array(raw.get(f"{a}|{s}", []))
            if len(v) >= MIN_POOL:
                P[(a, s)] = v
            else:
                P[(a, s)] = by_axis[a]; merged.append(f"{a}|{s}")
    return P, merged


def synth_design(model, side, subset):
    tk, _ = e1.token_key(model, SCFG)
    bad = set(failed_sizes(model))
    xs, ax, sz, tier = [], [], [], []
    for l in open(paths.get("data") / "synth" / "synth_v2" / "metadata.jsonl"):
        m = json.loads(l)
        if m.get("variant") == "base" and m["side"] == side and m["subset"] == subset and m["geometry"]["token_bin"][tk] is not None and m["sqrt_area_px"] not in bad:
            xs.append(m["geometry"]["center_along_tokens"][tk]); ax.append("x" if m["axis"] == "horizontal" else "y"); sz.append(m["sqrt_area_px"]); tier.append(m["level"])
    return np.array(xs), np.array(ax), np.array(sz), np.array(tier)


def Lparams(model, scen):
    if scen == "synth_main_so":
        return {"kappa": 3.0, "nu_sd": 0.5, "scale": "sd", "Lmodel": None}
    which = "binary" if scen.endswith("_pair") else "mcq"
    return {"kappa": None, "nu_sd": None, "scale": "sd", "Lmodel": nuis(model)["L_local"][which], "L_source": f"L_local.{which}"}


def sub_items(it, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in it.items()}


def or_verdict(items, n_boot, rng, th):
    keys = items["key"].astype(int)
    est = mh_or(items["box_ok"], items["mcq_ok"], np.ones(len(keys)), items["bin"])
    bs = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
        bs.append(mh_or(items["box_ok"], items["mcq_ok"], cnt, items["bin"]))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return {"or": float(est), "ci": ci, "verdict": "dependent" if ci[0] > th["or_dependent_min"] else ("independent_compatible" if ci[0] <= 1 <= ci[1] else "inconclusive"),
            "box_acc": float(items["box_ok"].mean()), "direct_acc": float(np.nanmean(items["mcq_ok"]))}


def calibrate_tau(agent, R, Lp, target, seed=7, n_orders=2):
    """二分找 tau，使模拟的直接回答正确率（dev 几何，2 万个条目）对上 dev 的边际正确率。返回 (tau, 达到的正确率, 是否能对上)。"""
    rng0 = np.random.default_rng(seed)
    idx = rng0.integers(0, len(R["delta"]), 20000)
    d, a = np.array(R["delta"])[idx], np.array(R["axis"])[idx]

    def acc(t):
        it = S.simulate_real(agent, d, a, np.random.default_rng(seed + 1), R["pools"], Lp["kappa"], Lp["nu_sd"], t, R["bin_edges_abs_delta"], Lp["scale"], n_orders, Lp["Lmodel"])
        return float(it["mcq_ok"].mean())
    lo, hi = 0.0, 4096.0
    a_lo, a_hi = acc(lo), acc(hi)
    if a_lo < target:
        return lo, a_lo, False
    if a_hi > target:
        return hi, a_hi, False
    for _ in range(40):
        mid = (lo + hi) / 2
        if acc(mid) > target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    return t, acc(t), True


def one(a):
    model, scen, agent, rep, boot, ctx = a
    rng = np.random.default_rng([20261128, SCEN.index(scen), MODELS.index(model), ["H1", "H2", "H3"].index(agent), rep])
    if scen.startswith("real_"):
        R = ctx["real"]
        idx = rng.integers(0, len(R["delta"]), R["n_test"])
        it = S.simulate_real(agent, np.array(R["delta"])[idx], np.array(R["axis"])[idx], rng, R["pools"], ctx["L"]["kappa"], ctx["L"]["nu_sd"], ctx["tau"][agent],
                             R["bin_edges_abs_delta"], ctx["L"]["scale"], ctx["n_orders"], ctx["L"]["Lmodel"])
        res = {"main_or": or_verdict(it, boot, rng, CFG["thresholds"])}
        try:
            res["A"] = X.coupling(it, boot, rng, CFG["alternatives"]["A_coupling"])
        except Exception as e:
            res["A"] = {"verdict": "error", "equivalence_null": False, "error": f"{type(e).__name__}: {e}"}
        return {"model": model, "scenario": scen, "agent": agent, "rep": rep, "boot": boot, "result": jsonable(res)}
    T, D, P, Lp = ctx["T"], ctx["D"], ctx["P"], ctx["L"]
    kw = dict(pool=P, kappa=Lp["kappa"] if Lp["kappa"] is not None else 3.0, nu_sd=Lp["nu_sd"] if Lp["nu_sd"] is not None else 0.5, scale=Lp["scale"], Lmodel=Lp["Lmodel"])
    it = S.simulate_L(agent, T[0], rng, axis=T[1], size=T[2], **kw)
    dv = S.simulate_L(agent, D[0], rng, axis=D[1], size=D[2], **kw)
    seed = int(rng.integers(2**31))
    if scen == "synth_main_pilot_wins":
        res = {"A": X.coupling(it, boot, np.random.default_rng(seed), CFG["alternatives"]["A_coupling"])}
    else:
        res = X.analyze_all(it, dv, CFG, boot, seed)
    if scen == "synth_overlap_pilot":
        res["by_tier"] = {}
        for tr in sorted(set(T[3].tolist())):
            m = T[3] == tr
            md = D[3] == tr
            if m.sum() < 20:
                continue
            res["by_tier"][tr] = X.analyze_all(sub_items(it, m), sub_items(dv, md), CFG, boot, seed + 1)
    return {"model": model, "scenario": scen, "agent": agent, "rep": rep, "boot": boot, "result": jsonable(res)}


EXPECT_REAL = {"main_or": {"H1": lambda r: r["main_or"]["verdict"] == "dependent", "H3": lambda r: r["main_or"]["verdict"] == "independent_compatible"},
               "A": {"H1": lambda r: r["A"]["verdict"] == "coupled", "H3": lambda r: r["A"]["verdict"] == "null_compatible"},
               "A_equiv": {"H3": lambda r: r["A"]["equivalence_null"]}}


def summarize(rows, scen):
    if scen.startswith("real_"):
        out = {}
        for ag in ("H1", "H3"):
            R = [r["result"] for r in rows if r["agent"] == ag]
            d = {"n": len(R)}
            for meth, ex in EXPECT_REAL.items():
                if ag in ex:
                    d[f"p_correct_{meth}"] = float(np.mean([ex[ag](r) for r in R]))
            d["median_or"] = float(np.median([r["main_or"]["or"] for r in R]))
            d["median_box_acc"] = float(np.median([r["main_or"]["box_acc"] for r in R]))
            d["median_direct_acc"] = float(np.median([r["main_or"]["direct_acc"] for r in R]))
            d["median_A_std_effect"] = float(np.median([r["A"].get("std_effect", np.nan) for r in R]))
            d["n_A_errors"] = int(sum("error" in r["A"] for r in R))
            out[ag] = d
        return out
    if scen == "synth_main_pilot_wins":
        out = {}
        for ag in ("H1", "H2", "H3"):
            R = [r["result"] for r in rows if r["agent"] == ag]
            d = {"n": len(R)}
            for meth in ("A", "A_equiv"):
                if ag in X.EXPECT[meth]:
                    d[f"p_correct_{meth}"] = float(np.mean([X.EXPECT[meth][ag](r) for r in R]))
            d["median_A_std_effect"] = float(np.median([r["A"]["std_effect"] for r in R]))
            out[ag] = d
        return out
    out = X.summarize_alt(rows)
    if scen == "synth_overlap_pilot":
        tiers = sorted({t for r in rows for t in r["result"].get("by_tier", {})})
        out["by_tier"] = {t: X.summarize_alt([{"agent": r["agent"], "result": r["result"]["by_tier"][t]} for r in rows if t in r["result"].get("by_tier", {})]) for t in tiers}
    return out


def context(model, scen):
    info = {"failed_sizes_excluded": failed_sizes(model)}
    if scen.startswith("real_"):
        pair = scen.endswith("_pair")
        ds = scen[5:-5] if pair else scen[5:]
        R = json.load(open(OUT / "real_inputs.json"))[model][ds]
        Lp = Lparams(model, scen)
        no = 1 if pair else 2
        target = R["p_pair_dev"] if pair else R["p_dir_dev"]
        tau = {}
        for ag in ("H1", "H3"):
            t, acc, ok = calibrate_tau(ag, R, Lp, target, n_orders=no)
            tau[ag] = t
            info[f"tau_{ag}"] = {"tau": t, "sim_direct_acc_dev_geometry": acc, "target_dev": target, "matched": ok}
        info.update(n_test=R["n_test"], p_box_dev=R["p_box_dev"], bin_edges=R["bin_edges_abs_delta"], L={k: v for k, v in Lp.items() if k != "Lmodel"}, direct="pair" if pair else "mcq")
        return {"real": R, "L": Lp, "tau": tau, "n_orders": no}, info
    subset = "overlap" if "overlap" in scen else "main"
    T, D = synth_design(model, "test", subset), synth_design(model, "dev", subset)
    if scen == "synth_main_so":
        P, merged = so_pools(model), []
    else:
        P, merged = pilot_pools(model, wins=scen.endswith("_wins"))
    Lp = Lparams(model, scen)
    info.update(n_test_items=int(len(T[0])), n_dev_items=int(len(D[0])), pools_merged_over_sizes=merged, L={k: v for k, v in Lp.items() if k != "Lmodel"},
                test_items_by_size={str(s): int((T[2] == s).sum()) for s in sorted(set(T[2].tolist()))})
    return {"T": T, "D": D, "P": P, "L": Lp}, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", choices=SCEN); ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=300); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--merge", action="store_true"); ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.merge:
        res = {"note": "排除单物体检查不通过的尺寸档；内部 bootstrap 300 次（正式分析 10,000 次）", "scenarios": {}}
        for sc in SCEN:
            for m in MODELS:
                f = OUT / sc / f"{m}.json"
                if f.is_file():
                    res["scenarios"].setdefault(sc, {})[m] = json.load(open(f))
        json.dump(res, open(paths.REPO_ROOT / "manifests" / "e1_power_sim_v3.json", "w"), indent=1, ensure_ascii=False)
        print({k: len(v) for k, v in res["scenarios"].items()})
        raise SystemExit(0)
    sc = a.scenario
    od = OUT / sc
    od.mkdir(parents=True, exist_ok=True)
    agents = ("H1", "H3") if sc.startswith("real_") else ("H1", "H2", "H3")
    for m in a.models.split(","):
        ctx, info = context(m, sc)
        print(m, sc, json.dumps(info, ensure_ascii=False), flush=True)
        tasks = [(m, sc, ag, r, a.boot, ctx) for ag in agents for r in range(a.reps)]
        prog = od / f"{m}.progress.jsonl"
        done = {}
        if prog.is_file():
            for l in open(prog):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r.get("boot") == a.boot:
                    done[(r["agent"], r["rep"])] = r
        tasks = [t for t in tasks if (t[2], t[3]) not in done]
        if a.timing:
            t0 = time.time(); one(tasks[0]); print(m, "timing one:", round(time.time() - t0, 1), "s", flush=True)
            continue
        t0 = time.time()
        with Pool(a.workers) as p, open(prog, "a") as fo:
            for k, r in enumerate(p.imap_unordered(one, tasks, chunksize=1)):
                fo.write(json.dumps(r) + "\n"); fo.flush()
                done[(r["agent"], r["rep"])] = r
                if (k + 1) % 20 == 0:
                    print(m, sc, "progress", len(done), "/", len(agents) * a.reps, round(time.time() - t0), "s", flush=True)
        rows = sorted((r for r in done.values() if r["rep"] < a.reps), key=lambda r: (r["agent"], r["rep"]))
        assert len(rows) == len(agents) * a.reps
        json.dump({"model": m, "scenario": sc, "reps": a.reps, "boot": a.boot, "info": info, "summary": summarize(rows, sc)}, open(od / f"{m}.json", "w"), indent=1, ensure_ascii=False)
        print(m, sc, "done", flush=True)
