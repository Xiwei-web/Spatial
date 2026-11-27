"""参数恢复（备选方案，planner 第 10 轮补充第 7、8 条）：模拟智能体 H1 / H2 / H3（src/e1_sim.simulate_L）× 两种噪声（0.1 token 接近满分；2 token 中间设定），
每种跑 --seeds 个种子，主判定与备选 A、B、C、档内 logistic 回归都报。test 样本：0.71–11.3 token 的 8 档各 160 个；B 选档用的 dev 样本：各档 80 个（另抽）。
结果：manifests/e1_recovery_alt.json。CPU 作业：python -m scripts.e1_recovery_alt --seeds 20 --boot 200 --workers 8"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse      # noqa: E402
import json      # noqa: E402
from multiprocessing import Pool      # noqa: E402

import numpy as np      # noqa: E402
import yaml      # noqa: E402

from src import e1_alt as X      # noqa: E402
from src import e1_sim as S      # noqa: E402
from src import paths      # noqa: E402

CFG = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "e1.yaml"))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def one(a):
    sigma, agent, seed, boot = a
    rng = np.random.default_rng([int(sigma * 1000), ["H1", "H2", "H3"].index(agent), seed])
    it = S.simulate_L(agent, S.design_x(rng, 160), rng, sigma=sigma)
    dv = S.simulate_L(agent, S.design_x(rng, 80), rng, sigma=sigma)
    return {"sigma": sigma, "agent": agent, "seed": seed, "result": jsonable(X.analyze_all(it, dv, CFG, boot, int(rng.integers(2**31))))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--boot", type=int, default=200); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = [(s, ag, k, a.boot) for s in (0.1, 2.0) for ag in ("H1", "H2", "H3") for k in range(a.seeds)]
    with Pool(a.workers) as p:
        rows = p.map(one, tasks, chunksize=1)
    out = {"config_status": CFG["status"], "seeds": a.seeds, "boot": a.boot, "design": "test 8 档 × 160（0.71–11.3 token），dev 8 档 × 80；kappa 3，nu_sd 0.5，H2: k 0.5、量化 0.5 token",
           "summary": {str(s): X.summarize_alt([r for r in rows if r["sigma"] == s]) for s in (0.1, 2.0)}, "rows": rows}
    json.dump(out, open(paths.REPO_ROOT / "manifests" / "e1_recovery_alt.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
