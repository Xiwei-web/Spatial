"""给 GQA-spatial（one / two）匹配 GQA scene graph 的框。匹配规则见 src/data/gqa_boxes.py。

输入：data/gqa/sceneGraphs/{val,train}_sceneGraphs.json（scripts/download_data.sh gqa_sg）、What's Up 的标注和图像。
输出：
  data/gqa/gqa_spatial_boxes.json           每个样本的匹配记录 + 框（xyxy，原图像素）
  manifests/gqa_spatial_boxes.json          同一文件的仓库副本（< 10 MB 才写）
  manifests/gqa_spatial_boxes_stats.json    统计、抽样例子（文档里的表格由它生成）
  <outputs>/sanity/gqa_boxes.png            20 个随机已匹配样本的拼图
确定性：不含时间戳，抽样用固定种子，输出 JSON 键有序。
用法：source scripts/env.sh tf5 && python scripts/build_gqa_boxes.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import shutil

from PIL import Image, ImageDraw

from src import paths
from src.data import gqa_boxes as gb
from src.data.loaders import _parse_caption, _vg_img, _wu_root

SEED = "gqa_boxes:20261125"
SUBSETS = {"gqa_spatial_one": "vg_qa_one_obj.json", "gqa_spatial_two": "vg_qa_two_obj.json"}
SIDES = ("left", "right", "top", "bottom")


def sha256(f) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_graphs(image_ids: set[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """先读 val；只有还有图没找到时才读 train（317 MB）。返回 (图 id → scene graph, 图 id → 来自哪个文件)。"""
    sg_dir = paths.get("data") / "gqa" / "sceneGraphs"
    graphs, src = {}, {}
    for split in ("val", "train"):
        if not (image_ids - set(graphs)):
            break
        d = json.load(open(sg_dir / f"{split}_sceneGraphs.json"))
        for i in sorted(image_ids - set(graphs)):
            if i in d:
                graphs[i], src[i] = d[i], split
    return graphs, src


def plural_variants(phrase: str) -> list[str]:
    """只用于诊断「单复数规则能救回多少」，不参与匹配。"""
    w = phrase.split()
    last = w[-1]
    alts = [last + "s", last + "es"]
    if last.endswith("es"):
        alts.append(last[:-2])
    if last.endswith("s"):
        alts.append(last[:-1])
    return [" ".join(w[:-1] + [a]) for a in alts]


def main() -> None:
    rows = {name: json.load(open(_wu_root() / f)) for name, f in SUBSETS.items()}
    graphs, graph_src = load_graphs({str(r[0]) for rs in rows.values() for r in rs})

    # ---- 图像尺寸核对：scene graph 的 width/height 对比实际文件（全部涉及的图像）----
    size_check = {"n_images": 0, "n_match": 0, "mismatch": []}
    real_wh = {}
    for i in sorted(graphs):
        with Image.open(_vg_img(i)) as im:
            real_wh[i] = im.size
        size_check["n_images"] += 1
        if real_wh[i] == (graphs[i]["width"], graphs[i]["height"]):
            size_check["n_match"] += 1
        else:
            size_check["mismatch"].append({"image_id": i, "file_wh": list(real_wh[i]),
                                           "scene_graph_wh": [graphs[i]["width"], graphs[i]["height"]]})

    samples: dict[str, dict] = {}
    stats: dict = {"size_check": size_check, "subsets": {}}
    out_of_bounds = 0
    max_overflow = 0.0
    plural_rescue = 0
    for name, rs in rows.items():
        table = {}
        c_status = {"target": collections.Counter(), "reference": collections.Counter()}
        c_rule = {"target": collections.Counter(), "reference": collections.Counter()}
        both = collections.Counter()
        side = collections.Counter()
        attr_same_name_gt1 = 0
        for idx, (img_id, cap_true, _cap_false) in enumerate(rs):
            img_id = str(img_id)
            t, ref, rel = _parse_caption(cap_true)
            g = graphs.get(img_id)
            rec = {"image_id": img_id, "scene_graph_file": graph_src.get(img_id), "relation": rel,
                   "target": gb.match_phrase(t, g), "reference": None if ref is None else gb.match_phrase(ref, g)}
            for role in ("target", "reference"):
                m = rec[role]
                if m is None:
                    continue
                c_status[role][m["status"]] += 1
                c_rule[role][f"{m['status']}/{m['rule']}"] += 1
                if m["status"] == gb.UNIQUE:
                    if m.get("n_same_name", 1) > 1:
                        attr_same_name_gt1 += 1
                    # scene graph 的框偶尔比实际图像多出 1 像素：裁到图像范围内，原值留在 bbox_raw
                    w, h = real_wh[img_id]
                    b = m["bbox"]
                    clipped = [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]
                    if clipped != b:
                        out_of_bounds += 1
                        max_overflow = max(max_overflow, *(abs(x - y) for x, y in zip(b, clipped)))
                        m["bbox_raw"], m["bbox"] = b, clipped
                if m["status"] == gb.NONE and g is not None:
                    if any(gb.find_candidates(v, g["objects"])[1] for v in plural_variants(m["name"])):
                        plural_rescue += 1
            # one-object：与 caption 所说一侧的一致性（只统计，不用来挑框）
            if ref is None and rel in SIDES and g is not None:
                m = rec["target"]
                if m["status"] == gb.UNIQUE:
                    ok = gb.side_of(m["bbox"], real_wh[img_id], rel)
                    rec["side_consistent"] = bool(ok)
                    side["unique_consistent" if ok else "unique_inconsistent"] += 1
                elif m["status"] == gb.MULTIPLE:
                    n_ok = sum(bool(gb.side_of(b, real_wh[img_id], rel)) for b in m["candidate_boxes"])
                    rec["n_candidates_on_stated_side"] = n_ok
                    side["multiple_would_disambiguate" if n_ok == 1 else "multiple_still_ambiguous"] += 1
            if ref is not None:
                ts, rs_ = rec["target"]["status"], rec["reference"]["status"]
                both["both_unique" if ts == rs_ == gb.UNIQUE else f"target={ts}|reference={rs_}"] += 1
            table[f"{idx:05d}_{img_id}"] = rec
        samples[name] = table
        stats["subsets"][name] = {
            "n": len(rs), "status": {k: dict(sorted(v.items())) for k, v in c_status.items() if v},
            "status_by_rule": {k: dict(sorted(v.items())) for k, v in c_rule.items() if v},
            "pair_status": dict(sorted(both.items())), "side_check": dict(sorted(side.items())),
            "unique_via_attr_with_same_name_gt1": attr_same_name_gt1,
            "scene_graph_file": dict(collections.Counter(r["scene_graph_file"] for r in table.values())),
        }
    stats["unique_boxes_out_of_image_bounds"] = out_of_bounds   # 已裁剪
    stats["max_overflow_px"] = max_overflow
    stats["none_rescued_by_plural_singular"] = plural_rescue

    # ---- 抽样例子（固定种子）----
    rng = random.Random(SEED)
    flat = [(n, sid, r) for n in sorted(samples) for sid, r in sorted(samples[n].items())]
    def all_unique(r):
        return r["target"]["status"] == gb.UNIQUE and (r["reference"] is None or r["reference"]["status"] == gb.UNIQUE)
    matched = [x for x in flat if all_unique(x[2])]
    unmatched = [x for x in flat if not all_unique(x[2])]
    def brief(n, sid, r):
        d = {"dataset": n, "sample_id": sid, "relation": r["relation"]}
        for role in ("target", "reference"):
            if r[role] is not None:
                d[role] = {k: r[role][k] for k in ("name", "status", "rule", "n_candidates", "bbox")}
        return d
    stats["examples_matched"] = [brief(*x) for x in rng.sample(matched, 10)]
    stats["examples_unmatched"] = [brief(*x) for x in rng.sample(unmatched, 10)]
    stats["n_fully_matched"], stats["n_not_fully_matched"] = len(matched), len(unmatched)

    arch = paths.get("data") / "_archives" / "gqa_sceneGraphs.zip"
    meta = {
        "source": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip (GQA ver 1.1)",
        "archive_sha256": sha256(arch) if arch.is_file() else None,
        "box_format": "xyxy，原图像素（scene graph 的 x, y, w, h 换算）",
        "rules": "R1 exact name；R2 attr+name（仅当 R1 无候选）；恰好 1 个候选才给框。详见 src/data/gqa_boxes.py",
        "status_values": [gb.UNIQUE, gb.NONE, gb.MULTIPLE, gb.NO_IMAGE],
    }
    out = paths.get("data") / "gqa" / gb.BOXES_FILENAME
    out.write_text(json.dumps({"meta": meta, "samples": samples}, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    man = paths.REPO_ROOT / "manifests"
    if out.stat().st_size < 10 * 1024 * 1024:
        shutil.copyfile(out, man / gb.BOXES_FILENAME)
    stats["boxes_file_bytes"], stats["boxes_file_sha256"] = out.stat().st_size, sha256(out)
    (man / "gqa_spatial_boxes_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True) + "\n")

    # ---- 拼图：20 个随机已匹配样本；红 = target，蓝 = reference ----
    pick = random.Random(SEED + ":sheet").sample(matched, 20)
    cell_w, cell_h, pad = 360, 300, 22
    sheet = Image.new("RGB", (5 * cell_w, 4 * (cell_h + pad)), "white")
    for k, (n, sid, r) in enumerate(pick):
        im = Image.open(_vg_img(r["image_id"])).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle(r["target"]["bbox"], outline=(255, 0, 0), width=3)
        if r["reference"] is not None:
            d.rectangle(r["reference"]["bbox"], outline=(0, 90, 255), width=3)
        im.thumbnail((cell_w - 4, cell_h - 4))
        x0, y0 = (k % 5) * cell_w, (k // 5) * (cell_h + pad)
        sheet.paste(im, (x0 + 2, y0 + pad))
        label = f"{n[-3:]} {sid} | {r['target']['name']} [{r['relation']}]" + (f" {r['reference']['name']}" if r["reference"] else "")
        ImageDraw.Draw(sheet).text((x0 + 3, y0 + 4), label[:58], fill="black")
    fig = paths.get("outputs") / "sanity" / "gqa_boxes.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(fig)
    stats_print = {k: v for k, v in stats.items() if not k.startswith("examples")}
    print(json.dumps(stats_print, ensure_ascii=False, indent=1))
    print("sheet:", fig, "| samples:", [(n, sid) for n, sid, _ in pick])


if __name__ == "__main__":
    main()
