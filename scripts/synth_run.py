"""合成集推理（GPU）。复用 E0 的框提示词（configs/bbox_conventions.yaml 校准结果）、答案方式（configs/e0.yaml）和冻结校验。
用法：
  python scripts/synth_run.py <模型> single_object               单物体定位检查：<data>/synth/single_object_v2 的全部图（dev 抠图），每张问一次框
  python scripts/synth_run.py <模型> e1 --smoke 5                E1 解析冒烟：合成 v2 的前 5 个 dev 主集基础场景（含两张孪生图），问框（两个物体）、同轴二选一 × 2 顺序、成对二元
  python scripts/synth_run.py <模型> e1 --pilot 200              E1 dev 试跑（planner 第 11 轮第 5 条）：dev 主集基础场景按设计距离层（10 层）等量分层抽样，不含孪生图，题型同冒烟；
                                                                  只用来估“讨厌参数”（L 的尺度与噪声、框误差分布、边际正确率），不在上面算判定量
  E1 正式运行尚未开放：要等 docs/PREREGISTRATION.md 提交、Xiwei 批 GPU 预算之后再加。
直接回答的每一步都按 fp32 打分（src/e1_fp32.py），记录首步（fp32_logprob）和“最终答案 token 所在的那一步”（fp32_final_answer）。
输出 <outputs>/synth/<模型>/<stage>.jsonl；断点续跑；出错即停。"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from scripts.smoke_test import git_commit
from src import e0, e1, paths
from src.e1_fp32 import FP32Head, answer_step_flexible, final_letter_step
from src.models.adapter import adapter_spec, config_hash, load_adapter, model_facts
from src.models.attn_switch import AttnSwitch
from src.prompts import OPPOSITE, parse_four_numbers, to_original_xyxy

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("stage", choices=["single_object", "e1"]); ap.add_argument("--smoke", type=int, default=None)
ap.add_argument("--pilot", type=int, default=None)
ap.add_argument("--direct-only-tag", default=None, help="只重跑直接回答（mcq、binary），输出 <stage>_<tag>.jsonl（第 11 轮：RoboBrain 的答案步定位补跑）")
ap.add_argument("--impl", default="eager", choices=["eager", "sdpa"]); ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()
cfg = e1.synth_cfg()
name = A.name
tk, tpx = e1.token_key(name, cfg)
if A.stage == "e1" and not ((A.smoke and 1 <= A.smoke <= 5 and not A.pilot) or (A.pilot and 10 <= A.pilot <= 200 and A.pilot % 10 == 0 and not A.smoke)):
    raise SystemExit("E1 正式运行尚未开放（需预注册提交 + Xiwei 批 GPU 预算）；现在只允许 --smoke N（N <= 5）或 --pilot N（N <= 200，10 的倍数），都只用 dev")
frozen = e0.check_frozen() if not A.dry_run else {"dry_run": True}
cal, modes = e0.calibrated(name), e0.answer_modes(name)
if not modes and not A.dry_run:
    raise SystemExit(f"configs/e0.yaml 里没有 {name} 的 answer_modes")
root = paths.get("data") / "synth"
jobs = []
if A.stage == "single_object":
    for l in open(root / "single_object_v2" / "metadata.jsonl"):
        m = json.loads(l)
        o = m["object"]
        jobs.append(dict(dataset="single_object_v2", sample_id=m["scene_id"], image_key=m["scene_id"], image=root / "single_object_v2" / m["image"], split=m["side"], task="box",
                         role="target", rotation=None, asked_relation=None, options=None, prompt=cal["prompt_template"].format(object=o["category"]), max_new=e0.BOX_MAX_NEW,
                         truth_xyxy=[float(v) for v in o["box_xyxy"]], box_object=o["category"], relation=None, target=o["category"], reference=None,
                         extra={"sqrt_area_px": m["sqrt_area_px"], "truth_centroid_xy": o["centroid_xy"]}))
else:
    metas = [json.loads(l) for l in open(root / "synth_v2" / "metadata.jsonl")]
    dev_base = [m for m in metas if m.get("variant") == "base" and m["side"] == "dev" and m["subset"] == "main"]
    if A.smoke:
        bases = [m["base_id"] for m in dev_base][:A.smoke]
        sel = [m for m in metas if m.get("base_id") in set(bases)]
    else:
        import random
        strata = sorted({tuple(m["design"]["stratum_px"]) for m in dev_base})
        bases = []
        for st in strata:
            ids = sorted(m["base_id"] for m in dev_base if tuple(m["design"]["stratum_px"]) == st)
            bases += random.Random(f"20261128:e1_pilot:{st}").sample(ids, A.pilot // len(strata))
        bases = set(bases)
        sel = [m for m in dev_base if m["base_id"] in bases]          # 只有基础场景，不含孪生图
        assert all(m["side"] == "dev" for m in sel) and len(sel) == A.pilot
    for m in sel:
        T, R = m["objects"]["target"], m["objects"]["reference"]
        g = m["geometry"]
        ex = {"variant": m["variant"], "base_id": m["base_id"], "level": m["level"], "subset": m["subset"], "axis": m["axis"], "token_key": tk, "token_px": tpx,
              "center_along_tokens": g["center_along_tokens"][tk], "token_bin": g["token_bin"][tk], "edge_gap_px": g["edge_gap_px"], "sqrt_area_px": m["sqrt_area_px"],
              "truth_centroid_xy": {"target": T["centroid_xy"], "reference": R["centroid_xy"]}, "occluded_fraction": m["occlusion"]["occluded_fraction"],
              "identity_swap_flag": (m.get("identity_swap") or {}).get("identity_swap_flag")}
        common = dict(dataset="synth_v2", sample_id=m["scene_id"], image_key=m["base_id"], image=root / "synth_v2" / m["image"], split=m["side"], relation=m["relation"],
                      target=T["category"], reference=R["category"], extra=ex)
        for role, o in (("target", T), ("reference", R)):
            jobs.append(dict(common, task="box", role=role, rotation=None, asked_relation=None, options=None, prompt=cal["prompt_template"].format(object=o["category"]),
                             max_new=e0.BOX_MAX_NEW, truth_xyxy=[float(v) for v in o["box_xyxy"]], box_object=o["category"]))
        for r in (0, 1):
            p, opts = e1.synth_mcq(T["category"], R["category"], m["axis"], r)
            jobs.append(dict(common, task="mcq", role=None, rotation=r, asked_relation=None, options=opts, prompt=p, max_new=e0.MAX_NEW[(modes or {}).get("mcq", "text16")],
                             truth_xyxy=None, box_object=None))
        for rel in (m["relation"], OPPOSITE[m["relation"]]):
            jobs.append(dict(common, task="binary", role=None, rotation=None, asked_relation=rel, options=None, prompt=e1.synth_binary(T["category"], R["category"], rel),
                             max_new=e0.MAX_NEW[(modes or {}).get("binary", "text16")], truth_xyxy=None, box_object=None))
stage = A.stage + (f"_smoke{A.smoke}" if A.smoke else "") + (f"_pilot{A.pilot}" if A.pilot else "")
if A.direct_only_tag:
    jobs = [j for j in jobs if j["task"] in ("mcq", "binary")]
    stage += "_" + A.direct_only_tag
out_dir = paths.get("outputs") / "synth" / name
out_dir.mkdir(parents=True, exist_ok=True)
f = out_dir / (stage + ("" if A.impl == "eager" else "_" + A.impl) + ".jsonl")
done = {k for k, r in e0.load_records(f).items() if not r.get("error")} if f.is_file() else set()
key = lambda j: (stage, j["dataset"], j["sample_id"], j["task"], j["role"], j["rotation"], j["asked_relation"])      # noqa: E731
todo = [j for j in jobs if key(j) not in done]
print(name, stage, A.impl, "jobs", len(jobs), "todo", len(todo), "frozen", frozen, flush=True)
if A.dry_run:
    for j in jobs[:6]:
        print(repr(j["prompt"]), j["options"], j["asked_relation"], j["max_new"], j["extra"])
    raise SystemExit(0)
if todo:
    import transformers
    from src.prompts import NO_VARIANTS, YES_VARIANTS, letter_variants
    ad = load_adapter(name)
    sw = AttnSwitch(ad)
    sw.use(A.impl)
    LETTER_V = [x for L in ("A", "B") for x in letter_variants(L)]
    try:
        fp32 = FP32Head(ad, LETTER_V + list(YES_VARIANTS) + list(NO_VARIANTS), all_steps=True)
        fp32.keep_hidden = True
        ans_ids = {"mcq": {v: fp32.ids[v] for v in LETTER_V if v in fp32.ids}, "binary": {v: fp32.ids[v] for v in YES_VARIANTS + NO_VARIANTS if v in fp32.ids}}
    except RuntimeError as ex:
        if A.stage != "single_object":          # 单物体检查只问框，用不到选项打分；E1 必须有
            raise
        print("WARN fp32 head hook unavailable:", ex, flush=True)
        fp32 = type("NoHook", (), {"reset": lambda self: None, "res": None, "all_steps": False})()
    meta = {"model": name, "revision": model_facts(name).get("revision"), "config_hash": config_hash(name), **frozen, "git_commit_dirty_check": git_commit(),
            "env": adapter_spec(name).get("env"), "torch": torch.__version__, "transformers": transformers.__version__, "dtype": str(ad.dtype), "impl": A.impl,
            "eager_fp32_patch": ad.eager_fp32, "decoding": "greedy", "slurm_job": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
            "stage": stage, "answer_modes": modes, "bbox_prompt_id": cal["prompt_id"], "bbox_convention": cal["convention"]}
    t0, cur, img = time.time(), None, None
    with open(f, "a") as fout:
        for n, j in enumerate(todo):
            rec = {**meta, **{k: j[k] for k in ("dataset", "sample_id", "image_key", "split", "relation", "target", "reference", "task", "role", "rotation", "asked_relation",
                                               "options", "prompt", "truth_xyxy", "box_object")}, **j["extra"], "max_new_tokens": j["max_new"], "error": None}
            try:
                t = time.time()
                if j["image"] != cur:
                    img, cur = Image.open(j["image"]).convert("RGB"), j["image"]
                prep = ad.prepare(img, j["prompt"])
                fp32.reset()
                fp32.all_steps = j["task"] != "box"          # 框题只要首步（其实也用不到）；直接回答每一步都打分
                g = ad.generate(prep, j["max_new"])
                ids = g["token_ids"]
                vb = prep.extra.get("view_box")
                rec.update(text=g["text"], text_raw=g["text_raw"], text16=ad.tokenizer.decode(ids[:16], skip_special_tokens=True), text16_raw=ad.tokenizer.decode(ids[:16], skip_special_tokens=False),
                           n_new_tokens=len(ids), hit_max_tokens=len(ids) >= j["max_new"], variant_logits=e0.variant_logits(g["first_logits"], ad.tokenizer), orig_wh=list(prep.orig_wh),
                           model_wh=list(prep.model_wh) if prep.model_wh else None, view_box=list(vb) if vb else None, n_input_tokens=int(prep.inputs["attention_mask"].shape[1]),
                           seconds=round(time.time() - t, 3))
                if j["task"] == "box":
                    rec["parsed_numbers"] = parse_four_numbers(g["text"]) or parse_four_numbers(g["text_raw"])
                    if rec["parsed_numbers"]:
                        rec["pred_xyxy_orig"] = [round(float(v), 3) for v in to_original_xyxy(rec["parsed_numbers"], cal["convention"], tuple(prep.orig_wh),
                                                                                            tuple(prep.model_wh) if prep.model_wh else None, tuple(vb) if vb else None)]
                else:
                    if fp32.res is None:
                        raise RuntimeError("fp32 打分钩子没有被调用")
                    rec.update(fp32.res)
                    rec["fp32_final_answer"] = final_letter_step(ids, fp32.steps, ans_ids[j["task"]])
                    rec["fp32_n_steps_hooked"] = len(fp32.steps)
                    rec["fp32_answer_step_flexible"] = answer_step_flexible(ad.tokenizer, ids, fp32, {"A": ["A"], "B": ["B"]} if j["task"] == "mcq" else {"yes": ["Yes", "yes"], "no": ["No", "no"]})
            except Exception as ex:
                rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            if rec["error"]:
                print("ERROR", rec["error"], flush=True)
                raise SystemExit(4)
            if n % 100 == 0:
                print(name, stage, n, "/", len(todo), round(time.time() - t0), "s", flush=True)
    print(name, stage, "finished", round(time.time() - t0, 1), "s", "n", len(todo), "peak_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2), "lm_calls", sw.calls, flush=True)
print("done", name, stage)
