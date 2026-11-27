"""合成集推理（GPU）。复用 E0 的框提示词（configs/bbox_conventions.yaml 校准结果）、答案方式（configs/e0.yaml）和冻结校验。
用法：
  python scripts/synth_run.py <模型> single_object               单物体定位检查：<data>/synth/single_object_v2 的全部图（dev 抠图），每张问一次框
  python scripts/synth_run.py <模型> e1 --smoke 5                E1 解析冒烟：合成 v2 的前 5 个 dev 主集基础场景（含两张孪生图），问框（两个物体）、同轴二选一 × 2 顺序、成对二元
  E1 正式运行尚未开放：要等 docs/PREREGISTRATION.md 提交、Xiwei 批 GPU 预算之后再加。
输出 <outputs>/synth/<模型>/<stage>.jsonl；断点续跑；出错即停。"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from scripts.smoke_test import git_commit
from src import e0, e1, paths
from src.models.adapter import adapter_spec, config_hash, load_adapter, model_facts
from src.models.attn_switch import AttnSwitch
from src.prompts import OPPOSITE, parse_four_numbers, to_original_xyxy

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("stage", choices=["single_object", "e1"]); ap.add_argument("--smoke", type=int, default=None)
ap.add_argument("--impl", default="eager", choices=["eager", "sdpa"]); ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()
cfg = e1.synth_cfg()
name = A.name
tk, tpx = e1.token_key(name, cfg)
if A.stage == "e1" and not (A.smoke and 1 <= A.smoke <= 5):
    raise SystemExit("E1 正式运行尚未开放（需预注册提交 + Xiwei 批 GPU 预算）；现在只允许 --smoke N（N <= 5，只用 dev）")
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
    bases = [m["base_id"] for m in metas if m.get("variant") == "base" and m["side"] == "dev" and m["subset"] == "main"][:A.smoke]
    for m in [m for m in metas if m.get("base_id") in set(bases)]:
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
stage = A.stage + (f"_smoke{A.smoke}" if A.smoke else "")
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
class FP32Head:
    """输出层（lm_head）的前向钩子：取首步最后一个位置的输入（最终归一化后的隐状态），分块按 fp32 重算 logit 与全词表 logsumexp。
    模型本身照常用 bf16 生成；这里只为直接回答的选项打分（planner 第 10 轮补充第 3 条：每个选项的 logprob 用 fp32）。"""

    def __init__(self, ad, variants):
        head = None
        for obj in (ad.model, getattr(ad.model, "language_model", None), getattr(ad.model, "llm", None)):
            if obj is not None and hasattr(obj, "get_output_embeddings") and obj.get_output_embeddings() is not None:
                head = obj.get_output_embeddings()
                break
        if head is None:
            raise RuntimeError("找不到输出层")
        self.ids = {}
        for v in variants:
            t = ad.tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1:
                self.ids[v] = t[0]
        self.res = None
        head.register_forward_hook(self.hook)

    def reset(self):
        self.res = None

    @torch.no_grad()
    def hook(self, m, inp, out):
        if self.res is not None:
            return
        x = inp[0][0, -1].float()
        W, b = m.weight, getattr(m, "bias", None)
        lse_parts, sel = [], {}
        for i in range(0, W.shape[0], 16384):
            z = W[i:i + 16384].float() @ x
            if b is not None:
                z = z + b[i:i + 16384].float()
            lse_parts.append(torch.logsumexp(z, 0))
            for v, t in self.ids.items():
                if i <= t < i + 16384:
                    sel[v] = float(z[t - i])
        lse = float(torch.logsumexp(torch.stack(lse_parts), 0))
        self.res = {"fp32_lse": lse, "fp32_logprob": {v: z - lse for v, z in sel.items()}, "fp32_argmax_logit_check": float(out[0, -1].float().max())}


if todo:
    import transformers
    from src.prompts import NO_VARIANTS, YES_VARIANTS, letter_variants
    ad = load_adapter(name)
    sw = AttnSwitch(ad)
    sw.use(A.impl)
    fp32 = FP32Head(ad, [x for L in ("A", "B") for x in letter_variants(L)] + list(YES_VARIANTS) + list(NO_VARIANTS))
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
