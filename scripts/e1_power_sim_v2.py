"""E1 功效模拟第二版（planner 第 10 轮补充第 1、2、4–6 条）：噪声换成合成集 dev 上实测的框中心误差。
噪声：单物体定位检查（manifests/single_object_check.json 的 signed_center_err_tok；400 张 dev 图，11 个模型），x、y 分开，按尺寸档（36/72/108/161 px）分池；
      每个条目的两个物体都从该场景尺寸档的池里各抽一个误差（合成 v2 主集里一对物体同尺寸）。解析失败的框不进池（比例另报）。
样本量与距离：合成 v2 的 test 主集基础场景（按模型自己的 9 档截断）；方案 B 选档用的 dev 样本 = 合成 v2 的 dev 主集基础场景的距离（只读几何元数据）。
智能体：src/e1_sim.simulate_L（H1 / H2 / H3，带 logit 差 L）；每次走 src/e1_alt.analyze_all（主判定 + A、B、C + logistic）。
功效模拟内部 bootstrap 用 --boot 次（默认 300），正式分析用 10,000 次。进度按（模型, 真值类型, 重复）逐条追加，断点续跑。
用法：python -m scripts.e1_power_sim_v2 --models A,B --reps 100 --boot 300 --workers 8；python -m scripts.e1_power_sim_v2 --merge"""
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
from scripts.e1_recovery_alt import jsonable      # noqa: E402

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "e1.yaml"))
SCFG = e1.synth_cfg()
OUT = paths.get("outputs") / "e1_power_v2"
MODELS = SCFG["single_object_check"]["models"]
SO = json.load(open(paths.REPO_ROOT / "manifests" / "single_object_check.json")) if (paths.REPO_ROOT / "manifests" / "single_object_check.json").is_file() else None


def pools(model):
    per = SO["models"][model]
    P, info = {}, {}
    for s, v in per.items():
        ex = np.array([e["ex_tok"] for e in v["signed_center_err_tok"]]); ey = np.array([e["ey_tok"] for e in v["signed_center_err_tok"]])
        P[("x", int(s))], P[("y", int(s))] = ex, ey
        info[s] = {"n": len(ex), "parse_fail": v["parse_fail"], "sd_x": float(ex.std()), "sd_y": float(ey.std()),
                   "mad_x": float(np.median(np.abs(ex - np.median(ex)))), "mad_y": float(np.median(np.abs(ey - np.median(ey))))}
    return P, info


def design(model, side):
    tk, _ = e1.token_key(model, SCFG)
    xs, ax, sz = [], [], []
    for l in open(paths.get("data") / "synth" / "synth_v2" / "metadata.jsonl"):
        m = json.loads(l)
        if m.get("variant") == "base" and m["side"] == side and m["subset"] == "main" and m["geometry"]["token_bin"][tk] is not None:
            xs.append(m["geometry"]["center_along_tokens"][tk]); ax.append("x" if m["axis"] == "horizontal" else "y"); sz.append(m["sqrt_area_px"])
    return np.array(xs), np.array(ax), np.array(sz), tk


def one(a):
    model, agent, rep, boot, T, D, P = a
    rng = np.random.default_rng([20261128, MODELS.index(model), ["H1", "H2", "H3"].index(agent), rep])
    it = S.simulate_L(agent, T[0], rng, pool=P, axis=T[1], size=T[2])
    dv = S.simulate_L(agent, D[0], rng, pool=P, axis=D[1], size=D[2])
    return {"model": model, "agent": agent, "rep": rep, "boot": boot, "result": jsonable(X.analyze_all(it, dv, CFG, boot, int(rng.integers(2**31))))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=300); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--merge", action="store_true"); ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.merge:
        mf = paths.REPO_ROOT / "manifests" / "e1_power_sim.json"
        res = json.load(open(mf))
        if "v1_whatsup_dev" not in res:
            res = {"v1_whatsup_dev": res}
        v2 = {"noise": "合成集 dev 单物体定位检查的有向框中心误差，按 (轴, 尺寸档) 分池", "boot_note": "功效模拟内部 bootstrap 300 次，正式分析 10,000 次", "models": {}}
        for m in MODELS:
            f = OUT / f"{m}.json"
            if f.is_file():
                v2["models"][m] = json.load(open(f))
        res["v2_synth_dev"] = v2
        json.dump(res, open(mf, "w"), indent=1, ensure_ascii=False)
        print(f"merged {len(v2['models'])} models")
        raise SystemExit(0)
    assert SO is not None, "manifests/single_object_check.json 还没有：先跑单物体定位检查与 scripts/single_object_metrics.py"
    OUT.mkdir(parents=True, exist_ok=True)
    for m in a.models.split(","):
        P, info = pools(m)
        T, D = design(m, "test")[:3], design(m, "dev")[:3]
        tasks = [(m, ag, r, a.boot, T, D, P) for ag in ("H1", "H2", "H3") for r in range(a.reps)]
        prog = OUT / f"{m}.progress.jsonl"
        done = {}
        if prog.is_file():
            for l in open(prog):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r.get("boot") == a.boot:
                    done[(r["agent"], r["rep"])] = r
        tasks = [t for t in tasks if (t[1], t[2]) not in done]
        if a.timing:
            t0 = time.time(); one(tasks[0]); print(m, "timing one:", round(time.time() - t0, 1), "s", flush=True)
            continue
        t0 = time.time()
        with Pool(a.workers) as p, open(prog, "a") as fo:
            for k, r in enumerate(p.imap_unordered(one, tasks, chunksize=1)):
                fo.write(json.dumps(r) + "\n"); fo.flush()
                done[(r["agent"], r["rep"])] = r
                if (k + 1) % 10 == 0:
                    print(m, "progress", len(done), "/", 3 * a.reps, round(time.time() - t0), "s", flush=True)
        rows = sorted((r for r in done.values() if r["rep"] < a.reps), key=lambda r: (r["agent"], r["rep"]))
        assert len(rows) == 3 * a.reps
        json.dump({"model": m, "n_test_items": int(len(T[0])), "n_dev_items": int(len(D[0])), "reps": a.reps, "boot": a.boot, "noise_by_size": info,
                   "summary": X.summarize_alt(rows)}, open(OUT / f"{m}.json", "w"), indent=1, ensure_ascii=False)
        print(m, "done", flush=True)
