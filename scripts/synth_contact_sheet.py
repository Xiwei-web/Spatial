"""合成场景的人工检查用联系表（F4）。每行一个基础场景：base | position_swap | identity_swap，
画出 target（红）/ reference（蓝）/ 干扰物（灰）的完整 mask 框，被遮挡物体的可见框用同色虚线，行首写参数。

场景顺序：按间距档位轮流取（每一轮每个档位取一个，轮内按间距从小到大），所以最前面的
ceil(档位数 / 每页行数) 页一定覆盖全部间距档位；同一档位内按序号取，大小 / 轴 / 干扰物数自然轮换。

输出：<outputs>/synth_review/<version>/sheet_XXX.jpg + index.json（每页有哪些场景、覆盖了哪些间距）

用法：python scripts/synth_contact_sheet.py [--config configs/synth/trial_v0.yaml] [--n-sheets 6] [--rows 8]
      python scripts/synth_contact_sheet.py --data-dir <临时目录> --out-dir <临时目录>      # 小试
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src import paths
from src.synth.generator import VARIANTS, load_config

COL = {"target": (220, 30, 30), "reference": (30, 80, 230), "distractor": (150, 150, 150)}
THUMB, CAP_W, PAD, HEAD = 448, 300, 6, 22


def _font(size: int):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def _dashed(draw, box, color, dash=6):
    x1, y1, x2, y2 = box
    for x in range(int(x1), int(x2), 2 * dash):
        draw.line([x, y1, min(x + dash, x2), y1], fill=color, width=2)
        draw.line([x, y2, min(x + dash, x2), y2], fill=color, width=2)
    for y in range(int(y1), int(y2), 2 * dash):
        draw.line([x1, y, x1, min(y + dash, y2)], fill=color, width=2)
        draw.line([x2, y, x2, min(y + dash, y2)], fill=color, width=2)


def _thumb(data_dir: Path, m: dict, font) -> Image.Image:
    im = Image.open(data_dir / m["image"]).convert("RGB")
    s = THUMB / im.width
    im = im.resize((THUMB, THUMB), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    sc = lambda b: [v * s for v in b]  # noqa: E731
    for d in m["distractors"]:
        dr.rectangle(sc(d["box_xyxy"]), outline=COL["distractor"], width=1)
    for role in m["z_order"][-2:]:
        o = m["objects"][role]
        dr.rectangle(sc(o["box_xyxy"]), outline=COL[role], width=2)
        if o["visible_box_xyxy"] and o["visible_box_xyxy"] != o["box_xyxy"]:
            _dashed(dr, sc(o["visible_box_xyxy"]), COL[role])
    t, r = m["objects"]["target"], m["objects"]["reference"]
    txt = f"{m['variant']}: {t['category']} [{m['relation']}] {r['category']}"
    dr.rectangle([0, 0, THUMB, 18], fill=(255, 255, 255))
    dr.text((3, 2), txt, fill=(0, 0, 0), font=font)
    return im


def _caption(ms: dict[str, dict]) -> str:
    b = ms["base"]
    d, dist, occ = b["design"], b["distance"], b["occlusion"]
    lines = [
        b["base_id"],
        f"axis={d['axis']}  dist={d['distance_px']}px",
        f"  actual={dist['along_axis_px']}  jitter={dist['jitter_requested_px']}",
        f"size={d['size_frac']} ({d['size_px']}px)  n_dis={d['n_distractors']}",
        f"bg={d['background']}",
        f"top={occ['top']}",
        f"pair_center={b['pair_position']['pair_center_xy']}",
        f"seed={b['seed']['scene_seed']} try={b['seed']['attempt']}",
        "",
        "occluded fraction of lower / IoU:",
    ]
    for v in VARIANTS:
        o = ms[v]["occlusion"]
        lines.append(f"  {v[:8]:8s} {o['occluded_fraction']:.3f} / {o['iou_complete']:.3f}")
    lines += ["", f"T: {b['objects']['target']['cutout_id']}", f"R: {b['objects']['reference']['cutout_id']}",
              f"T(id-swap): {ms['identity_swap']['objects']['target']['cutout_id']}",
              f"  ({ms['identity_swap']['objects']['target']['category']})",
              "", "red=target blue=reference", "grey=distractor dashed=visible box"]
    return "\n".join(lines)


def order_scenes(metas: list[dict], distances: list[int]) -> list[int]:
    by_d = defaultdict(list)
    for m in metas:
        if m["variant"] == "base":
            by_d[m["design"]["distance_px"]].append(m["index"])
    out, r = [], 0
    while any(r < len(by_d[d]) for d in distances):
        out += [sorted(by_d[d])[r] for d in distances if r < len(by_d[d])]
        r += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth/trial_v0.yaml")
    ap.add_argument("--n-sheets", type=int, default=6)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_dir = args.data_dir or paths.get("data") / "synth" / cfg["version"]
    out = args.out_dir or paths.get("outputs") / "synth_review" / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads(line) for line in open(data_dir / "metadata.jsonl")]
    by_scene = defaultdict(dict)
    for m in metas:
        by_scene[m["index"]][m["variant"]] = m
    order = order_scenes(metas, cfg["factors"]["distance_px"])[: args.n_sheets * args.rows]
    f_small, f_cap = _font(12), _font(13)
    index = []
    W = CAP_W + 3 * (THUMB + PAD) + PAD
    for p in range(0, len(order), args.rows):
        chunk = order[p:p + args.rows]
        sheet = Image.new("RGB", (W, HEAD + len(chunk) * (THUMB + PAD) + PAD), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        dists = [by_scene[i]["base"]["design"]["distance_px"] for i in chunk]
        dr.text((PAD, 4), f"{cfg['version']}  sheet {p // args.rows:03d}  distances on this sheet: {dists}   "
                          f"columns: base | position_swap | identity_swap", fill=(0, 0, 0), font=f_cap)
        for k, i in enumerate(chunk):
            y = HEAD + k * (THUMB + PAD)
            dr.multiline_text((PAD, y + 4), _caption(by_scene[i]), fill=(0, 0, 0), font=f_cap, spacing=3)
            for c, v in enumerate(VARIANTS):
                sheet.paste(_thumb(data_dir, by_scene[i][v], f_small), (CAP_W + PAD + c * (THUMB + PAD), y))
        name = f"sheet_{p // args.rows:03d}.jpg"
        sheet.save(out / name, quality=88)
        index.append({"sheet": name, "scene_indices": chunk, "distances_px": dists})
    covered, n_first = set(), 0
    for e in index:
        covered |= set(e["distances_px"])
        n_first += 1
        if covered >= set(cfg["factors"]["distance_px"]):
            break
    json.dump({"version": cfg["version"], "rows_per_sheet": args.rows, "sheets": index,
               "all_distance_levels_covered_by_first_n_sheets": n_first if covered >= set(cfg["factors"]["distance_px"]) else None},
              open(out / "index.json", "w"), indent=1)
    print(f"[ok] {len(index)} sheets → {out}; 前 {n_first} 页覆盖的间距档位：{sorted(covered)}")


if __name__ == "__main__":
    main()
