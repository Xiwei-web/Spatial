"""E0 生成（GPU）：dev 解析检查（parse_check）与 What's Up dev 入选筛选（screen）。只记录，不算指标；断点续跑。
用法：python scripts/e0_run.py <模型> <parse_check|screen> [--impl eager|sdpa]
启动时做冻结校验（src/e0.py: check_frozen）；screen 还要求 configs/e0.yaml 里已有该模型的 answer_modes。
输出 <outputs>/e0/<模型>/<stage>[_sdpa].jsonl；字段见 docs/E0_PLAN.md 第 4 节。"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from scripts.smoke_test import git_commit
from src import e0, paths
from src.models.adapter import adapter_spec, config_hash, load_adapter, model_facts
from src.models.attn_switch import AttnSwitch
from src.prompts import OPPOSITE, binary_question, e0_mcq, n_choices, parse_four_numbers

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("stage", choices=["parse_check", "screen"]); ap.add_argument("--impl", default="eager", choices=["eager", "sdpa"])
ap.add_argument("--dry-run", action="store_true", help="只列任务清单（CPU），不加载模型")
A = ap.parse_args()
frozen = e0.check_frozen() if not A.dry_run else {"dry_run": True}
if not A.dry_run and not torch.cuda.is_available():
    raise SystemExit("没有可用的 GPU")
name, stage = A.name, A.stage
cal = e0.calibrated(name)
modes = e0.answer_modes(name)
if stage == "screen" and not modes and not A.dry_run:
    raise SystemExit(f"configs/e0.yaml 里没有 {name} 的 answer_modes：先完成解析检查并提交")
out_dir = paths.get("outputs") / "e0" / name
out_dir.mkdir(parents=True, exist_ok=True)
f = out_dir / (stage + ("" if A.impl == "eager" else "_" + A.impl) + ".jsonl")
done = set()
if f.is_file():
    for k, r in e0.load_records(f).items():
        if not r.get("error"):
            done.add(k)

# ---- 任务清单 ----
jobs = []
if stage == "parse_check":
    for ds in e0.PARSE_CHECK_DATASETS:
        for s in e0.parse_check_samples(ds):
            p, o = e0_mcq(ds, s.target, s.reference, 0, s.relation)
            jobs.append(dict(s=s, task="mcq", role=None, rotation=0, asked_relation=None, options=o, prompt=p, max_new=64))
            for rel in (s.relation, OPPOSITE[s.relation]):
                jobs.append(dict(s=s, task="binary", role=None, rotation=None, asked_relation=rel, options=None, prompt=binary_question(ds, s.target, s.reference, rel), max_new=64))
else:
    from src.data.whatsup_annotations import load_annotations
    ann = {(a["dataset"], a["sample_id"]): a for a in load_annotations("auto_v3")}
    for ds in ("whatsup_a", "whatsup_b"):
        for s in sorted(e0.dev_samples(ds), key=lambda s: s.sample_id):
            a = ann[(ds, s.sample_id)]
            for role in ("target", "reference"):
                o = a["objects"][role]
                assert o["name"] == getattr(s, role)
                jobs.append(dict(s=s, task="box", role=role, rotation=None, asked_relation=None, options=None, prompt=cal["prompt_template"].format(object=o["name"]),
                                 max_new=e0.BOX_MAX_NEW, truth_xyxy=o["box_gt_xyxy"]))
            for r in range(n_choices(ds)):
                p, opts = e0_mcq(ds, s.target, s.reference, r, s.relation)
                jobs.append(dict(s=s, task="mcq", role=None, rotation=r, asked_relation=None, options=opts, prompt=p, max_new=e0.MAX_NEW[(modes or {}).get("mcq", "text16")]))
            for rel in (s.relation, OPPOSITE[s.relation]):
                jobs.append(dict(s=s, task="binary", role=None, rotation=None, asked_relation=rel, options=None, prompt=binary_question(ds, s.target, s.reference, rel),
                                 max_new=e0.MAX_NEW[(modes or {}).get("binary", "text16")]))
key = lambda j: (stage, j["s"].dataset, j["s"].sample_id, j["task"], j["role"], j["rotation"], j["asked_relation"])      # noqa: E731
todo = [j for j in jobs if key(j) not in done]
print(name, stage, A.impl, "jobs", len(jobs), "todo", len(todo), "frozen", frozen, flush=True)
if A.dry_run:
    from collections import Counter
    print(Counter((j["s"].dataset, j["task"]) for j in jobs))
    for j in jobs[:3] + jobs[-2:]:
        print(repr(j["prompt"]), j["options"], j["asked_relation"], j["max_new"])
    raise SystemExit(0)
if todo:
    import transformers
    ad = load_adapter(name)
    sw = AttnSwitch(ad)
    sw.use(A.impl)
    meta = {"model": name, "revision": model_facts(name).get("revision"), "config_hash": config_hash(name), **frozen, "git_commit_dirty_check": git_commit(),
            "env": adapter_spec(name).get("env"), "torch": torch.__version__, "transformers": transformers.__version__, "dtype": str(ad.dtype), "impl": A.impl,
            "eager_fp32_patch": ad.eager_fp32, "decoding": "greedy", "slurm_job": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
            "stage": stage, "answer_modes": modes, "bbox_prompt_id": cal["prompt_id"], "bbox_convention": cal["convention"]}
    t0, n_err = time.time(), 0
    cur_img, img = None, None
    with open(f, "a") as fout:
        for n, j in enumerate(todo):
            s = j["s"]
            rec = {**meta, "dataset": s.dataset, "sample_id": s.sample_id, "image_key": e0.image_key(s), "relation": s.relation, "target": s.target, "reference": s.reference,
                   "task": j["task"], "role": j["role"], "rotation": j["rotation"], "asked_relation": j["asked_relation"], "options": j["options"], "prompt": j["prompt"],
                   "max_new_tokens": j["max_new"], "truth_xyxy": j.get("truth_xyxy"), "error": None}
            try:
                t = time.time()
                if s.image_path != cur_img:
                    img, cur_img = Image.open(s.image_path).convert("RGB"), s.image_path
                prep = ad.prepare(img, j["prompt"])
                g = ad.generate(prep, j["max_new"])
                ids = g["token_ids"]
                vb = prep.extra.get("view_box")
                rec.update(text=g["text"], text_raw=g["text_raw"], text16=ad.tokenizer.decode(ids[:16], skip_special_tokens=True), text16_raw=ad.tokenizer.decode(ids[:16], skip_special_tokens=False), n_new_tokens=len(ids),
                           hit_max_tokens=len(ids) >= j["max_new"], variant_logits=e0.variant_logits(g["first_logits"], ad.tokenizer), orig_wh=list(prep.orig_wh),
                           model_wh=list(prep.model_wh) if prep.model_wh else None, view_box=list(vb) if vb else None, n_input_tokens=int(prep.inputs["attention_mask"].shape[1]),
                           seconds=round(time.time() - t, 3))
                if j["task"] == "box":
                    rec["parsed_numbers"] = parse_four_numbers(g["text"]) or parse_four_numbers(g["text_raw"])
            except Exception as ex:                                       # 照实记录，不静默丢弃；出错就停（停止条件）
                rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
                n_err += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            if rec["error"]:
                print("ERROR", rec["error"], flush=True)
                raise SystemExit(4)
            if n % 200 == 0:
                print(name, stage, n, "/", len(todo), round(time.time() - t0), "s", flush=True)
    print(name, stage, "finished", round(time.time() - t0, 1), "s", "n", len(todo), "peak_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2), "lm_calls", sw.calls, flush=True)
print("done", name, stage)
