"""E1 功效模拟（planner 第 10 轮第 4 条）：只用 dev 的噪声参数，不碰 test 的回答。
噪声：E0 What's Up dev（outputs/e0/<模型>/screen.jsonl）上实测的框中心误差，x、y 方向分开，以该模型的视觉 token 为单位（解析不出的框不进池，比例另报）。
样本量：合成 v2 的 test 主集基础场景，按该模型的 token 档截断后的实际距离和轴（只读几何元数据，不读任何模型回答）。
对每个模型 × 真值类型（H1 / H2 / H3）模拟 R 次，每次走与真实数据相同的分析代码（bootstrap B 次），统计判定结果与 CI 宽度。
用法：python scripts/e1_power_sim.py --models A,B --reps 100 --boot 300 --workers 8   → <outputs>/e1_power/<模型>.json
      python scripts/e1_power_sim.py --merge                                         → manifests/e1_power_sim.json + 说明表（markdown，打印）"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import yaml

from src import e0, e1, paths
from src import e1_analysis as A
from src import e1_sim as S

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "e1.yaml"))
SCFG = e1.synth_cfg()
OUT = paths.get("outputs") / "e1_power"
MODELS = SCFG["single_object_check"]["models"]


def noise_pool(model):
    recs = [r for r in e0.load_records(paths.get("outputs") / "e0" / model / "screen.jsonl").values() if r["task"] == "box"]
    assert all(r["stage"] == "screen" and r["dataset"] in ("whatsup_a", "whatsup_b") for r in recs)
    ex, ey, n_fail = [], [], 0
    for r in recs:
        c = A.pred_center(r)
        if c is None:
            n_fail += 1
            continue
        t = r["truth_xyxy"]
        sx, sy = A.token_scale(r)
        ex.append((c[0] - (t[0] + t[2]) / 2) / sx)
        ey.append((c[1] - (t[1] + t[3]) / 2) / sy)
    return np.array(ex), np.array(ey), n_fail, len(recs)


def design(model):
    tk, _ = e1.token_key(model, SCFG)
    xs, ax = [], []
    for l in open(paths.get("data") / "synth" / "synth_v2" / "metadata.jsonl"):
        m = json.loads(l)
        if m.get("variant") == "base" and m["side"] == "test" and m["subset"] == "main" and m["geometry"]["token_bin"][tk] is not None:
            xs.append(m["geometry"]["center_along_tokens"][tk]); ax.append("x" if m["axis"] == "horizontal" else "y")
    return np.array(xs), np.array(ax), tk


def one(args):
    model, agent, rep, boot, x, ax, px, py = args
    rng = np.random.default_rng([20261127, MODELS.index(model), ["H1", "H2", "H3"].index(agent), rep])
    it = S.simulate(agent, x, rng, pool={"x": px, "y": py}, axis=ax)
    edges = S.edges9()
    it["bin"] = S.token_bins(x, edges)
    r = A.analyze(it, CFG, n_boot=boot, seed=int(rng.integers(2**31)))
    return {"agent": agent, "rep": rep, "ratio": r.get("ratio"), "ratio_ci": r["ci"].get("ratio"), "or": r["or"], "or_ci": r["ci"]["or"], "or_pooled": r["or_pooled"],
            "theta_dir": r.get("theta_dir"), "theta_io": r.get("theta_io"), "direct_acc": r["direct_acc"], "box_acc": r["box_acc"], "decision": r["decision"]}


def summarize(rows, th):
    e_lo, e_hi = th["ratio_equivalence"]
    out = {}
    for ag in ("H1", "H2", "H3"):
        R = [r for r in rows if r["agent"] == ag]
        rc = np.array([r["ratio_ci"] for r in R]); oc = np.array([r["or_ci"] for r in R])
        out[ag] = {"n_reps": len(R),
                   "p_ratio_ci_within_equivalence": float(np.mean((rc[:, 0] >= e_lo) & (rc[:, 1] <= e_hi))),
                   "p_ratio_ci_upper_below_h2": float(np.mean(rc[:, 1] < th["ratio_h2_upper"])),
                   "p_or_ci_lower_above_min": float(np.mean(oc[:, 0] > th["or_dependent_min"])),
                   "p_or_ci_contains_1": float(np.mean((oc[:, 0] <= 1) & (oc[:, 1] >= 1))),
                   "median_ratio": float(np.median([r["ratio"] for r in R])), "median_ratio_ci_width": float(np.median(rc[:, 1] - rc[:, 0])),
                   "median_or": float(np.median([r["or"] for r in R])), "median_log_or_ci_width": float(np.median(np.log(oc[:, 1]) - np.log(oc[:, 0]))),
                   "median_theta_io_tokens": float(np.median([r["theta_io"] for r in R])), "median_theta_dir_tokens": float(np.median([r["theta_dir"] for r in R])),
                   "median_direct_acc": float(np.median([r["direct_acc"] for r in R])), "median_box_acc": float(np.median([r["box_acc"] for r in R]))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=300); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        res = {"config_status": CFG["status"], "thresholds": CFG["thresholds"], "or_method": CFG.get("or_method"), "models": {}}
        for m in MODELS:
            f = OUT / f"{m}.json"
            if f.is_file():
                res["models"][m] = json.load(open(f))
                res["reps"], res["boot"] = res["models"][m]["reps"], res["models"][m]["boot"]
        json.dump(res, open(paths.REPO_ROOT / "manifests" / "e1_power_sim.json", "w"), indent=1, ensure_ascii=False)
        print("| 模型 | token | n_test 条目 | 噪声 sd x / y（token） | θ_IO 中位（token） | H1：比值 CI ⊂ [0.8,1.25] | H1：OR 下界 > 2 | H3：OR CI ∋ 1 | H3：比值 CI ⊂ 等价区间 | H2：比值 CI 上界 < 0.8 | 比值 CI 宽度中位 H1 / H3 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for m, v in res["models"].items():
            s = v["summary"]
            print(f"| {m} | {v['token_key']} | {v['n_items']} | {v['noise']['sd_x']:.2f} / {v['noise']['sd_y']:.2f} | {s['H1']['median_theta_io_tokens']:.2f} | {s['H1']['p_ratio_ci_within_equivalence']:.2f} | "
                  f"{s['H1']['p_or_ci_lower_above_min']:.2f} | {s['H3']['p_or_ci_contains_1']:.2f} | {s['H3']['p_ratio_ci_within_equivalence']:.2f} | {s['H2']['p_ratio_ci_upper_below_h2']:.2f} | "
                  f"{s['H1']['median_ratio_ci_width']:.2f} / {s['H3']['median_ratio_ci_width']:.2f} |")
        raise SystemExit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    for m in a.models.split(","):
        px, py, nf, nb = noise_pool(m)
        x, ax, tk = design(m)
        tasks = [(m, ag, r, a.boot, x, ax, px, py) for ag in ("H1", "H2", "H3") for r in range(a.reps)]
        with Pool(a.workers) as p:
            rows = p.map(one, tasks, chunksize=1)
        q = lambda v: [float(np.percentile(v, k)) for k in (5, 25, 50, 75, 95)]      # noqa: E731
        rep = {"model": m, "token_key": tk, "n_items": int(len(x)), "reps": a.reps, "boot": a.boot, "k_h2": 0.5,
               "noise": {"source": "E0 What's Up dev (screen.jsonl) box records", "n_box": nb, "n_unparsed": nf, "sd_x": float(px.std()), "sd_y": float(py.std()),
                         "mad_x": float(np.median(np.abs(px - np.median(px)))), "mad_y": float(np.median(np.abs(py - np.median(py)))), "pct_x": q(px), "pct_y": q(py)},
               "summary": summarize(rows, CFG["thresholds"]), "rows": rows}
        json.dump(rep, open(OUT / f"{m}.json", "w"), indent=1)
        print(m, json.dumps(rep["summary"]), flush=True)
