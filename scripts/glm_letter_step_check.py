"""GLM-4.6V-Flash 的 L 口径验证（planner 第 11 轮第 3 条；GPU，≤ 0.1 h；只用冒烟的 dev 记录）。
对冒烟里 GLM 的 30 条二选一（mcq）和 30 条成对二元（binary）：
  1. 用同样的提示词与贪心解码重新生成，每一步按 fp32 打分（src/e1_fp32.py），检查生成文本与冒烟记录逐字相同；
  2. 在生成序列里找最终答案 token（mcq：最后一个 A/B；binary：最后一个 Yes/No 类 token），取该步的 fp32 logprob（“最终字母步”）；
  3. teacher-forced 重算：把提示词 + 生成序列中该步之前的 token 一次前向，取最后一个位置的 fp32 logprob，与第 2 步比较（检查逐步钩子取对了位置）。
只改记录方式，不改提示和解码。输出 <outputs>/synth/GLM-4.6V-Flash/glm_letter_step_check.jsonl 与 manifests/glm_letter_step_check.json（汇总，由 --summarize 生成）。
用法：python scripts/glm_letter_step_check.py            （GPU）
      python scripts/glm_letter_step_check.py --summarize（CPU）"""
import argparse
import json
import os
import time

import numpy as np

from src import e0, paths
from src.prompts import LETTERS, NO_VARIANTS, YES_VARIANTS, letter_variants

NAME = "GLM-4.6V-Flash"
D = paths.get("outputs") / "synth" / NAME
SMOKE = D / "e1_smoke5.jsonl"
OUT = D / "glm_letter_step_check.jsonl"
MAN = paths.REPO_ROOT / "manifests" / "glm_letter_step_check.json"
LETTER_V = [x for L in ("A", "B") for x in letter_variants(L)]


def best(lp, vs):
    v = [lp[x] for x in vs if x in lp]
    return max(v) if v else float("nan")


def L_of(r, lp):
    """正确方向为正的 logit 差（fp32 logprob 之差）：mcq = 正确字母 − 错误字母（各取变体最大）；binary = 正确回答（Yes/No）− 另一个。"""
    if lp is None:
        return float("nan")
    if r["task"] == "mcq":
        k = r["options"].index(r["relation"])
        right, wrong = LETTERS[k], LETTERS[1 - k]
        return best(lp, letter_variants(right)) - best(lp, letter_variants(wrong))
    y, n = best(lp, YES_VARIANTS), best(lp, NO_VARIANTS)
    return (y - n) if r["asked_relation"] == r["relation"] else (n - y)


def run():
    import torch
    from PIL import Image
    from src.e1_fp32 import FP32Head, final_letter_step
    from src.models.adapter import load_adapter
    from src.models.attn_switch import AttnSwitch
    recs = [json.loads(l) for l in open(SMOKE)]
    recs = [r for r in recs if r["task"] in ("mcq", "binary") and not r.get("error")]
    assert len(recs) == 60, len(recs)
    ad = load_adapter(NAME)
    AttnSwitch(ad).use("eager")
    head = FP32Head(ad, LETTER_V + list(YES_VARIANTS) + list(NO_VARIANTS), all_steps=True)
    ids = {"mcq": {v: head.ids[v] for v in LETTER_V if v in head.ids}, "binary": {v: head.ids[v] for v in YES_VARIANTS + NO_VARIANTS if v in head.ids}}
    root = paths.get("data") / "synth" / "synth_v2" / "images"
    t0 = time.time()
    with open(OUT, "w") as fo:
        for r in recs:
            img = Image.open(root / f"{r['sample_id']}.png").convert("RGB")
            prep = ad.prepare(img, r["prompt"])
            head.reset()
            g = ad.generate(prep, r["max_new_tokens"])
            gen = g["token_ids"]
            fin = final_letter_step(gen, head.steps, ids[r["task"]])
            first = head.res
            tf = None
            if fin and "error" not in fin:
                k = fin["step"]
                inp = dict(prep.inputs)
                n_in = inp["input_ids"].shape[1]
                extra = torch.tensor([gen[:k]], dtype=inp["input_ids"].dtype, device=inp["input_ids"].device)
                for key, v in list(inp.items()):
                    if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, n_in):
                        pad = extra if key == "input_ids" else (torch.ones_like(extra) if key == "attention_mask" else torch.zeros_like(extra))
                        inp[key] = torch.cat([v, pad.to(v.dtype)], 1)
                head.reset()
                with torch.no_grad():
                    ad.model(**inp, return_dict=True)
                tf = {"fp32_logprob": head.res["fp32_logprob"], "fp32_lse": head.res["fp32_lse"], "n_tokens": int(inp["input_ids"].shape[1])}
            out = {k: r[k] for k in ("sample_id", "task", "rotation", "asked_relation", "relation", "options", "text_raw")}
            out.update(regen_text_raw=g["text_raw"], text_identical=g["text_raw"] == r["text_raw"], n_new=len(gen), final=fin, first_fp32_logprob=first["fp32_logprob"],
                       smoke_first_fp32_logprob=r.get("fp32_logprob"), teacher_forced=tf, slurm_job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0))
            fo.write(json.dumps(out, ensure_ascii=False) + "\n"); fo.flush()
    print("done", round(time.time() - t0, 1), "s")


def summarize():
    rs = [json.loads(l) for l in open(OUT)]
    S = {}
    for task in ("mcq", "binary"):
        R = [r for r in rs if r["task"] == task]
        Lf = np.array([L_of(r, r["first_fp32_logprob"]) for r in R])
        Lfin = np.array([L_of(r, (r["final"] or {}).get("fp32_logprob")) for r in R])
        Ltf = np.array([L_of(r, (r["teacher_forced"] or {}).get("fp32_logprob")) for r in R])
        dif = [max(abs(r["final"]["fp32_logprob"][v] - r["teacher_forced"]["fp32_logprob"][v]) for v in r["final"]["fp32_logprob"]) for r in R if r["teacher_forced"]]
        smoke_dif = [max(abs(r["first_fp32_logprob"][v] - r["smoke_first_fp32_logprob"][v]) for v in r["first_fp32_logprob"]) for r in R if r.get("smoke_first_fp32_logprob")]
        q = lambda a: [float(np.nanpercentile(a, p)) for p in (0, 10, 25, 50, 75, 90, 100)] if np.isfinite(a).any() else None      # noqa: E731
        S[task] = {"n": len(R), "text_identical": int(sum(r["text_identical"] for r in R)), "final_found": int(sum(bool(r["final"]) for r in R)),
                   "final_step_hist": dict(sorted({str(k): int(sum((r["final"] or {}).get("step") == k for r in R)) for k in {(r["final"] or {}).get("step") for r in R}}.items())),
                   "final_fp32_argmax_is_generated": int(sum(bool((r["final"] or {}).get("fp32_argmax_is_generated")) for r in R)),
                   "L_first_step_quantiles": q(Lf), "L_final_step_quantiles": q(Lfin), "L_teacher_forced_quantiles": q(Ltf),
                   "frac_abs_L_final_gt10": float(np.nanmean(np.abs(Lfin) > 10)) if np.isfinite(Lfin).any() else None,
                   "frac_abs_L_first_gt10": float(np.nanmean(np.abs(Lf) > 10)),
                   "max_abs_diff_final_vs_teacher_forced_logprob": float(max(dif)) if dif else None,
                   "max_abs_diff_L_final_vs_teacher_forced": float(np.nanmax(np.abs(Lfin - Ltf))) if np.isfinite(Ltf).any() else None,
                   "max_abs_diff_first_step_vs_smoke": float(max(smoke_dif)) if smoke_dif else None,
                   "sign_L_final_matches_parsed_answer_correct": None}
        if task == "mcq":
            ok = [(L_of(r, r["final"]["fp32_logprob"]) > 0) == (e0.answer({**r, "text": r["regen_text_raw"], "text_raw": r["regen_text_raw"]}, "text64") == LETTERS[r["options"].index(r["relation"])])
                  for r in R if r["final"]]
            S[task]["sign_L_final_matches_parsed_answer_correct"] = f"{sum(ok)}/{len(ok)}"
    json.dump(S, open(MAN, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run()
