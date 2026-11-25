"""阶段 E 冒烟测试：只确认“管线能跑通、内部状态拿得到”，不是评测。
用法：python scripts/smoke_test.py <模型名>
输出：outputs/smoke/<模型名>/{records.jsonl, summary.json, token_region.png}
records.jsonl 每条带：模型名、revision、提示词、配置文件哈希、Git commit。
注意：这 20 张图上的对错不构成结果，本脚本不计算也不输出任何正确率。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback

import torch
import yaml
from PIL import Image

from src import checks, paths
from src.data import load_dataset
from src.models.adapter import adapter_spec, config_hash, load_adapter, model_facts
from src.prompts import LETTERS, parse_four_numbers, parse_mcq, relation_mcq, to_original_xyxy

GENERIC_BBOX_PROMPT = "Output the bounding box of the {object} in the image as [x1, y1, x2, y2]."


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(paths.REPO_ROOT), "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def bbox_convention(name: str) -> dict:
    f = paths.REPO_ROOT / "configs" / "bbox_conventions.yaml"
    if f.is_file():
        for key, fam in (yaml.safe_load(open(f)) or {}).items():
            if isinstance(fam, dict) and name in (fam.get("applies_to") or []):
                if fam.get("found") and fam.get("prompt_template") and fam.get("convention", "unknown") != "unknown":
                    return {"official": True, "family": key, "prompt_template": fam["prompt_template"], "convention": fam["convention"]}
                return {"official": False, "family": key, "prompt_template": GENERIC_BBOX_PROMPT, "convention": "unknown",
                        "why": "未找到官方提示词或坐标约定（见 configs/bbox_conventions.yaml）"}
    return {"official": False, "family": None, "prompt_template": GENERIC_BBOX_PROMPT, "convention": "unknown",
            "why": "configs/bbox_conventions.yaml 中没有该模型"}


def gpu_mem_gb(peak=False) -> float:
    fn = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return round(sum(fn(i) for i in range(torch.cuda.device_count())) / 2**30, 2)


def run_check(summary: dict, key: str, fn):
    try:
        r = fn()
    except Exception as e:
        r = {"pass": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
    summary["tests"][key] = r
    print(f"[{key}] pass={r.get('pass')} {json.dumps({k: v for k, v in r.items() if k not in ('traceback', 'layers')}, default=str, ensure_ascii=False)[:600]}", flush=True)
    torch.cuda.empty_cache()
    return r


def draw_token_region(prep, image, k, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    gh, gw = prep.grid_hw
    fig, ax = plt.subplots(figsize=(7, 7 * image.size[1] / image.size[0]))
    ax.imshow(image)
    for j in range(gh * gw):
        x1, y1, x2, y2 = prep.token_box(j)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=0.2, alpha=0.5))
    x1, y1, x2, y2 = prep.token_box(k)
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, fc=(1, 0, 0, 0.35), ec="red", lw=2))
    r, c = divmod(k, gw)
    ax.set_title(f"grid {gh}x{gw}; token k={k} (row {r}, col {c}) -> seq pos {prep.grid_positions[k]}", fontsize=9)
    ax.axis("off")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(name: str) -> int:
    out_dir = paths.get("outputs") / "smoke" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, facts = adapter_spec(name), model_facts(name)
    meta = {"model": name, "revision": facts.get("revision"), "config_hash": config_hash(name), "git_commit": git_commit(),
            "env": spec.get("env"), "torch": torch.__version__}
    import transformers
    meta["transformers"] = transformers.__version__
    summary = {**meta, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "tests": {}, "note": "smoke test only; no accuracy is computed or reported"}

    def save():
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=1, ensure_ascii=False, default=str)

    # ---- 1. 加载 ----
    if spec["kind"] == "unsupported":
        summary["tests"]["1_load"] = {"pass": False, "reason": spec.get("reason")}
        save()
        return 1
    try:
        torch.cuda.reset_peak_memory_stats()
        ad = load_adapter(name)
        summary["tests"]["1_load"] = {"pass": True, "load_seconds": round(ad.load_seconds, 1), "gpu_mem_after_load_gb": gpu_mem_gb(),
                                      "lm_layers_path": ad.lm_layers_path, "n_lm_layers": len(ad.lm_layers),
                                      "device_map": sorted({str(v) for v in getattr(ad.model, "hf_device_map", {"": 0}).values()})}
    except Exception as e:
        summary["tests"]["1_load"] = {"pass": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        print(traceback.format_exc(), flush=True)
        save()
        return 1
    print("[1_load]", summary["tests"]["1_load"], flush=True)
    save()

    items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
    by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
    samples = [by_id[(i["dataset"], i["sample_id"])] for i in items]
    bc = bbox_convention(name)
    opt_ids = ad.option_token_ids()
    summary["bbox_convention"] = bc
    summary["option_token_ids"] = opt_ids

    # ---- 2 & 3. 框预测 / 关系回答（保存原始生成文本）----
    n_box = n_box_parsed = n_box_restored = n_rel = n_rel_parsed = n_logit = 0
    times, errors = [], []
    with open(out_dir / "records.jsonl", "w") as fout:
        for s in samples:
            image = Image.open(s.image_path).convert("RGB")
            tasks = [("bbox_target", bc["prompt_template"].format(object=s.target), 96),
                     ("bbox_reference", bc["prompt_template"].format(object=s.reference), 96)]
            mcq, opts = relation_mcq(s.dataset, s.target, s.reference)
            tasks.append(("relation_mcq", mcq, 16))
            for task, prompt, mnt in tasks:
                rec = {**meta, "dataset": s.dataset, "sample_id": s.sample_id, "image_path": s.image_path, "task": task,
                       "prompt": prompt, "max_new_tokens": mnt, "decoding": "greedy"}
                try:
                    t0 = time.time()
                    prep = ad.prepare(image, prompt)
                    g = ad.generate(prep, mnt)
                    torch.cuda.synchronize()
                    rec.update(seconds=round(time.time() - t0, 3), n_input_tokens=int(prep.inputs["attention_mask"].shape[1]),
                               n_visual_tokens=len(prep.visual_positions), raw_text=g["text_raw"], text=g["text"],
                               model_wh=prep.model_wh, orig_wh=prep.orig_wh)
                    times.append(rec["seconds"])
                    if task.startswith("bbox"):
                        n_box += 1
                        nums = parse_four_numbers(g["text"]) or parse_four_numbers(g["text_raw"])
                        rec.update(bbox_prompt_official=bc["official"], convention=bc["convention"], parsed_numbers=nums, bbox_xyxy_orig=None)
                        if nums is not None:
                            n_box_parsed += 1
                            if bc["convention"] != "unknown":
                                rec["bbox_xyxy_orig"] = to_original_xyxy(nums, bc["convention"], prep.orig_wh, prep.model_wh)
                                n_box_restored += 1
                    else:
                        n_rel += 1
                        letter = parse_mcq(g["text"])
                        rec.update(options=dict(zip(LETTERS, opts)), parsed_option=letter, gold_relation=s.relation)
                        fl = g["first_logits"]
                        rec["option_logits"] = {L: {tag: (None if tid is None else float(fl[tid])) for tag, tid in ids.items()}
                                                for L, ids in opt_ids.items()}
                        n_rel_parsed += letter is not None
                        n_logit += all(v["bare"] is not None for v in rec["option_logits"].values())
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    errors.append(rec["error"])
                    if len(errors) <= 2:
                        print(traceback.format_exc(), flush=True)
                fout.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                fout.flush()
                torch.cuda.empty_cache()
    summary["tests"]["1_load"].update(seconds_per_generation_mean=round(sum(times) / max(len(times), 1), 3),
                                      gpu_mem_peak_gb=gpu_mem_gb(peak=True))
    summary["tests"]["2_bbox"] = {
        "pass": bool(bc["official"] and n_box == 40 and n_box_parsed == 40 and n_box_restored == 40),
        "official_prompt": bc["official"], "convention": bc["convention"], "n_generated": n_box, "n_parsed_four_numbers": n_box_parsed,
        "n_restored_to_original": n_box_restored, "errors": errors[:3],
        "criterion": "有官方提示词与坐标约定，且 40 次生成全部解析出四个坐标并还原到原图坐标系（不判断框是否正确）"}
    summary["tests"]["3_relation"] = {
        "pass": bool(n_rel == 20 and n_rel_parsed == 20 and n_logit == 20), "n_generated": n_rel, "n_parsed_option": n_rel_parsed,
        "n_with_four_option_logits": n_logit, "criterion": "20 次生成全部解析出选项字母，且四个选项 token 的 logit 都已保存（不判断对错）"}
    print("[2_bbox]", summary["tests"]["2_bbox"], "\n[3_relation]", summary["tests"]["3_relation"], flush=True)
    save()

    # ---- 4-8：在固定的一张图 + 关系提示词上做 ----
    s = samples[0]
    image = Image.open(s.image_path).convert("RGB")
    prep = ad.prepare(image, relation_mcq(s.dataset, s.target, s.reference)[0])
    summary["probe_sample"] = {"dataset": s.dataset, "sample_id": s.sample_id, "seq_len": int(prep.inputs["attention_mask"].shape[1]),
                               "n_visual_tokens": len(prep.visual_positions)}
    run_check(summary, "4_hidden_states", lambda: checks.check_hidden_states(ad, prep))
    a = run_check(summary, "5a_attention_weights", lambda: checks.check_attention_weights(ad, prep))
    h = run_check(summary, "5b_head_hook", lambda: checks.check_head_hook(ad, prep))
    ns = run_check(summary, "5c_nonstandard_layers", lambda: {"pass": True, **checks.describe_nonstandard_layers(ad, prep)})
    summary["tests"]["5_attention_and_heads"] = {"pass": bool(a.get("pass") and h.get("pass")),
                                                 "nonstandard": {k: v for k, v in ns.items() if k != "pass"}}

    def vis_check():
        vp = prep.visual_positions
        contiguous = bool(vp) and vp == list(range(vp[0], vp[-1] + 1))
        r = {"n_visual_tokens": len(vp), "start": vp[0] if vp else None, "end": vp[-1] if vp else None, "contiguous": contiguous,
             "grid_hw": prep.grid_hw, "model_wh": prep.model_wh, "orig_wh": prep.orig_wh, "note": prep.region_note, **prep.extra}
        r.pop("eos_token_id", None); r.pop("input_ids", None)
        if prep.grid_hw is None:
            r["pass"] = False
            r["reason"] = r.get("grid_mismatch") or "视觉 token 没有固定的图像区域对应"
            return r
        gh, gw = prep.grid_hw
        k = (gh // 3) * gw + (gw // 4)       # 一个已知位置：约 1/3 高、1/4 宽处
        png = out_dir / "token_region.png"
        draw_token_region(prep, image, k, png)
        r.update(known_token_k=k, known_token_box_xyxy=[round(v, 1) for v in prep.token_box(k)], figure=str(png),
                 n_grid_tokens=len(prep.grid_positions), n_unmapped_visual_tokens=len(vp) - len(prep.grid_positions))
        r["pass"] = True
        return r

    run_check(summary, "6_visual_token_mapping", vis_check)
    t_noop = run_check(summary, "7a_noop_hooks_bitwise", lambda: checks.check_noop_identity(ad, prep))
    t_patch = run_check(summary, "7b_self_patch", lambda: checks.check_self_patch(ad, prep))
    t_ko = run_check(summary, "7c_attention_knockout", lambda: checks.check_knockout(ad, prep))
    summary["tests"]["7_interventions"] = {"pass": bool(t_noop.get("pass") and t_patch.get("pass") and t_ko.get("pass"))}
    run_check(summary, "8_determinism", lambda: checks.check_determinism(ad, prep, 32))
    summary["gpu_mem_peak_gb_overall"] = gpu_mem_gb(peak=True)
    save()
    eight = ["1_load", "2_bbox", "3_relation", "4_hidden_states", "5_attention_and_heads", "6_visual_token_mapping",
             "7_interventions", "8_determinism"]
    print("SUMMARY", name, {k: summary["tests"].get(k, {}).get("pass") for k in eight}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
