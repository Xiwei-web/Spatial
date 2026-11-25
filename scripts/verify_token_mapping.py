"""1.4：视觉 token → 图像区域映射的行为验证（不涉及任何正确率）。
用法：python scripts/verify_token_mapping.py <模型名>
做法：取 5 张图，每张把一个已知矩形区域（位置由固定种子决定）替换成均匀噪声，比较替换前后
“送入语言模型的视觉 token 嵌入”（hidden_states[0] 在视觉 token 位置上的向量）的 L2 变化量。
  预测集合 P = 映射表中格子与该矩形相交面积 ≥ 50% 格子面积的 token
  实测集合 A = 变化量最大的 |P| 个 token
  重合率 = |P ∩ A| / |P|
另外报告：把 P 放宽为“与矩形有任何相交”的集合 P_any 后，A 落在 P_any 内的比例；以及第 1、2、3 层输出处的同一重合率
（Qwen3-VL 的 DeepStack 在这几层之后向图像 token 再次注入视觉特征）。
输出：<outputs>/token_mapping_verify/<模型>/{result.json, compare.png}"""
import json
import sys

import numpy as np
import torch
from PIL import Image

from src import paths
from src.data import load_dataset
from src.models.adapter import load_adapter
from src.prompts import relation_mcq

name = sys.argv[1]
SEED = 20261125
items = json.load(open(paths.REPO_ROOT / "splits" / "smoke20.json"))["items"]
by_id = {(ds, s.sample_id): s for ds in {i["dataset"] for i in items} for s in load_dataset(ds)}
samples = [by_id[(i["dataset"], i["sample_id"])] for i in items][::4][:5]      # 固定的 5 张
ad = load_adapter(name)
rng = np.random.default_rng(SEED)
results, panels = [], []
for s in samples:
    image = Image.open(s.image_path).convert("RGB")
    W, H = image.size
    rw, rh = int(W * 0.25), int(H * 0.25)                      # 矩形：宽高各占 25%
    x0, y0 = int(rng.integers(0, W - rw)), int(rng.integers(0, H - rh))
    rect = (x0, y0, x0 + rw, y0 + rh)
    arr = np.asarray(image).copy()
    arr[y0:y0 + rh, x0:x0 + rw] = rng.integers(0, 256, size=(rh, rw, 3), dtype=np.uint8)
    noised = Image.fromarray(arr)
    prompt = relation_mcq(s.dataset, s.target, s.reference)[0]
    p0, p1 = ad.prepare(image, prompt), ad.prepare(noised, prompt)
    assert p0.grid_hw == p1.grid_hw and p0.grid_positions == p1.grid_positions, "替换噪声后网格发生变化"
    h0 = ad.forward(p0, output_hidden_states=True).hidden_states
    h1 = ad.forward(p1, output_hidden_states=True).hidden_states
    gp = torch.as_tensor(p0.grid_positions)
    gh, gw = p0.grid_hw
    frac = np.zeros(gh * gw)
    for k in range(gh * gw):
        bx0, by0, bx1, by1 = p0.token_box(k)
        ix = max(0, min(bx1, rect[2]) - max(bx0, rect[0])); iy = max(0, min(by1, rect[3]) - max(by0, rect[1]))
        frac[k] = ix * iy / ((bx1 - bx0) * (by1 - by0))
    P = set(np.nonzero(frac >= 0.5)[0].tolist())
    P_any = set(np.nonzero(frac > 0)[0].tolist())
    per_layer = {}
    for li in (0, 1, 2, 3):
        d = (h0[li][0, gp].float() - h1[li][0, gp].float()).norm(dim=-1).cpu().numpy()
        A = set(np.argsort(-d)[: len(P)].tolist())
        per_layer[li] = {"overlap": len(P & A) / len(P), "top_in_any_intersect": len(A & P_any) / len(A),
                         "mean_change_in_P": float(d[list(P)].mean()), "mean_change_outside_P_any": float(np.delete(d, list(P_any)).mean())}
        if li == 0:
            d0 = d
            # 补充指标：(1) 预测覆盖比例图与实测变化图的相关系数；(2) 实测变化的加权质心与矩形中心的距离（单位：token）；
            # (3) 对照：把预测图左右翻转 / 上下翻转 / 同时翻转后的重合率——映射若有系统性错位，对照不应明显低于正确映射
            fm, dm = frac.reshape(gh, gw), d.reshape(gh, gw)
            wgt = np.clip(dm - np.median(dm), 0, None)
            yy, xx = np.mgrid[0:gh, 0:gw]
            cy, cx = (wgt * yy).sum() / wgt.sum(), (wgt * xx).sum() / wgt.sum()
            ry, rx = (fm * yy).sum() / fm.sum(), (fm * xx).sum() / fm.sum()
            ctrl = {}
            for tag, alt in (("hflip", fm[:, ::-1]), ("vflip", fm[::-1]), ("hvflip", fm[::-1, ::-1])):
                Pa = set(np.nonzero(alt.ravel() >= 0.5)[0].tolist())
                ctrl[tag] = len(Pa & A) / max(len(Pa), 1)
            per_layer[0].update(corr_pred_vs_measured=float(np.corrcoef(fm.ravel(), dm.ravel())[0, 1]),
                                centroid_error_tokens=float(np.hypot(cy - ry, cx - rx)),
                                argmax_inside_rect=bool(frac[int(np.argmax(d))] > 0), control_overlap=ctrl)
    # 非网格的其余视觉 token（如 LLaVA-OV 切片）不在此验证范围
    results.append({"dataset": s.dataset, "sample_id": s.sample_id, "rect_xyxy": rect, "grid_hw": [gh, gw], "n_pred": len(P),
                    "n_any": len(P_any), "per_layer": per_layer})
    panels.append((noised, rect, frac.reshape(gh, gw), d0.reshape(gh, gw), s.sample_id, per_layer[0]["overlap"]))
    print(s.sample_id, "grid", (gh, gw), "|P|", len(P), {f"hs{k}": round(v["overlap"], 3) for k, v in per_layer.items()}, flush=True)
summary = {"model": name, "seed": SEED, "n_images": len(results), "rect_frac": 0.25,
           "definition": "P=格子≥50%落在噪声矩形内的 token；A=嵌入变化最大的 |P| 个 token；overlap=|P∩A|/|P|",
           "mean_overlap_hs0": float(np.mean([r["per_layer"][0]["overlap"] for r in results])),
           "min_overlap_hs0": float(np.min([r["per_layer"][0]["overlap"] for r in results])),
           "mean_top_in_any_intersect_hs0": float(np.mean([r["per_layer"][0]["top_in_any_intersect"] for r in results])),
           "mean_corr_pred_vs_measured": float(np.mean([r["per_layer"][0]["corr_pred_vs_measured"] for r in results])),
           "mean_centroid_error_tokens": float(np.mean([r["per_layer"][0]["centroid_error_tokens"] for r in results])),
           "max_centroid_error_tokens": float(np.max([r["per_layer"][0]["centroid_error_tokens"] for r in results])),
           "n_argmax_inside_rect": int(sum(r["per_layer"][0]["argmax_inside_rect"] for r in results)),
           "mean_control_overlap": {k: float(np.mean([r["per_layer"][0]["control_overlap"][k] for r in results])) for k in ("hflip", "vflip", "hvflip")},
           "mean_change_ratio_in_vs_out": float(np.mean([r["per_layer"][0]["mean_change_in_P"] / max(r["per_layer"][0]["mean_change_outside_P_any"], 1e-9) for r in results])),
           "mean_overlap_after_layer": {str(k): float(np.mean([r["per_layer"][k]["overlap"] for r in results])) for k in (1, 2, 3)},
           "images": results}
out = paths.get("outputs") / "token_mapping_verify" / name
out.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out / "result.json", "w"), indent=1, ensure_ascii=False)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, axes = plt.subplots(3, len(panels), figsize=(4.2 * len(panels), 10))
for j, (img, rect, frac, d, sid, ov) in enumerate(panels):
    axes[0, j].imshow(img); axes[0, j].add_patch(Rectangle(rect[:2], rect[2] - rect[0], rect[3] - rect[1], fill=False, ec="red", lw=2))
    axes[0, j].set_title(f"{sid}\nnoise rect (red)", fontsize=8)
    axes[1, j].imshow(frac, cmap="gray", vmin=0, vmax=1); axes[1, j].set_title("predicted by mapping\n(cell fraction inside rect)", fontsize=8)
    axes[2, j].imshow(d, cmap="magma"); axes[2, j].set_title(f"measured |Δ embedding| per token\noverlap={ov:.2f}", fontsize=8)
    for a in axes[:, j]:
        a.axis("off")
fig.suptitle(f"{name}: visual-token → image-region mapping, behavioural check (mean overlap {summary['mean_overlap_hs0']:.3f})", fontsize=11)
fig.tight_layout()
fig.savefig(out / "compare.png", dpi=110)
print("SUMMARY", name, "mean_overlap_hs0", round(summary["mean_overlap_hs0"], 3), "min", round(summary["min_overlap_hs0"], 3),
      "after layers 1-3", {k: round(v, 3) for k, v in summary["mean_overlap_after_layer"].items()},
      "| corr", round(summary["mean_corr_pred_vs_measured"], 3), "centroid_err_tok", round(summary["mean_centroid_error_tokens"], 2),
      "argmax_in_rect", summary["n_argmax_inside_rect"], "/", len(results), "controls", {k: round(v, 3) for k, v in summary["mean_control_overlap"].items()},
      "in/out", round(summary["mean_change_ratio_in_vs_out"], 1))
