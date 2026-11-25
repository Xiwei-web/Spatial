"""切图库：从带 mask 的图像里抠出 RGBA 小图（alpha 来自 mask），供合成场景使用。

两个来源：
  coco_val2017     COCO val2017 的实例标注（人工多边形 mask）；
  whatsup_auto_v1  What's Up A/B 的**自动**标注（GroundingDINO 框 + SAM 2.1 mask），只取自动质检 pass 的图。
                   这份标注没有经过人工核对，metadata 里一律标 `source: whatsup_auto_v1`。

筛选规则（阈值在 configs/synth/*.yaml 的 cutouts 段；按下面的顺序判，第一条不满足的记为淘汰原因）：
  small_area      mask 面积 < min_mask_area_px
  small_frac      mask 面积 / 图像面积 < min_mask_area_frac
  touches_border  mask 的框离图像某条边不足 border_margin_px（多半被图像边界截断）
  fragmented      最大连通块（8 连通）面积 / mask 面积 < min_largest_component_frac
  low_fill        最大连通块面积 / 它的紧致框面积 < min_fill
  overlapped      同图某个其他实例的 mask 落在最大连通块紧致框内的面积 / 该框面积 > max_other_overlap
后两条是“没有被严重遮挡”的近似：COCO 没有遮挡标注，只能用形状和邻居来猜。

入库的小图 = 最大连通块的紧致框裁剪（其余小碎块的 alpha 置 0），长边缩到不超过 max_side。
RGB 通道保留原图像素（透明处也是），这样后续缩放不会把黑边混进物体边缘。
坐标约定同 src/data/coco_pairs.py：像素 (i, j) 覆盖 [i, i+1) × [j, j+1)，框为 [x1, y1, x2, y2)，质心 = 下标均值 + 0.5。
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from src import paths

SEED = 20261125
ALPHA_THR = 128
COCO_SOURCE = "coco_val2017"
WHATSUP_SOURCE = "whatsup_auto_v1"          # 自动标注，未经人工核对
REASONS = ["small_area", "small_frac", "touches_border", "fragmented", "low_fill", "overlapped"]


def library_dir() -> Path:
    return paths.get("data") / "synth" / "cutouts"


# =====================================================================
# mask 质量
# =====================================================================
def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """二值 mask 的紧致框 [x1, y1, x2, y2)（x2 / y2 为最大下标 + 1）；空 mask 返回 None。"""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    """返回 (最大连通块, 它占 mask 面积的比例, 连通块个数)；8 连通。"""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n <= 1:
        return mask.astype(bool), 1.0 if n == 1 else 0.0, n
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes))               # 并列时取标号最小的，结果确定
    return lab == k + 1, float(sizes[k] / sizes.sum()), n


def check_mask(mask: np.ndarray, others: list[np.ndarray], cfg: dict) -> tuple[str | None, dict, np.ndarray | None]:
    """按模块说明里的顺序判一个 mask。返回 (淘汰原因 或 None, 统计量, 最大连通块)。others 是同图其他实例的 mask。"""
    mask = mask.astype(bool)
    H, W = mask.shape
    area = int(mask.sum())
    st = {"mask_area": area, "area_frac": round(area / (H * W), 5)}
    if area < cfg["min_mask_area_px"]:
        return "small_area", st, None
    if area / (H * W) < cfg["min_mask_area_frac"]:
        return "small_frac", st, None
    x1, y1, x2, y2 = mask_bbox(mask)
    st["mask_bbox_xyxy"] = [x1, y1, x2, y2]
    m = cfg["border_margin_px"]
    if x1 < m or y1 < m or x2 > W - m or y2 > H - m:
        return "touches_border", st, None
    big, frac, n = largest_component(mask)
    st.update(largest_component_frac=round(frac, 5), n_components=n)
    if frac < cfg["min_largest_component_frac"]:
        return "fragmented", st, None
    # 以下两条针对真正入库的那一块（最大连通块）及其紧致框
    x1, y1, x2, y2 = mask_bbox(big)
    box_area = (x2 - x1) * (y2 - y1)
    fill = int(big.sum()) / box_area
    st.update(component_bbox_xyxy=[x1, y1, x2, y2], fill=round(fill, 5))
    if fill < cfg["min_fill"]:
        return "low_fill", st, None
    ov = max((int(o[y1:y2, x1:x2].sum()) / box_area for o in others), default=0.0)
    st["max_other_overlap"] = round(ov, 5)
    if ov > cfg["max_other_overlap"]:
        return "overlapped", st, None
    return None, st, big


# =====================================================================
# 裁剪 / 缩放（切图库构建和场景生成共用，保证两边的做法一致）
# =====================================================================
def tight_crop(rgba: np.ndarray, thr: int = ALPHA_THR) -> np.ndarray:
    """裁到 alpha >= thr 的紧致框。"""
    box = mask_bbox(rgba[..., 3] >= thr)
    if box is None:
        raise ValueError("alpha 全部低于阈值，无法裁剪")
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(rgba[y1:y2, x1:x2])


def _resize_once(rgba: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    s = long_side / max(h, w)
    nw, nh = max(1, math.floor(w * s + 0.5)), max(1, math.floor(h * s + 0.5))
    rgb = Image.fromarray(np.ascontiguousarray(rgba[..., :3])).resize((nw, nh), Image.LANCZOS)
    a = Image.fromarray(np.ascontiguousarray(rgba[..., 3])).resize((nw, nh), Image.BILINEAR)
    return np.dstack([np.asarray(rgb), np.asarray(a)])


def resize_cutout(rgba: np.ndarray, long_side: int, thr: int = ALPHA_THR) -> np.ndarray:
    """把小图缩放到「alpha >= thr 的紧致框长边 = long_side」，返回紧致裁剪后的 RGBA（uint8）。

    RGB 用 LANCZOS，alpha 用 BILINEAR（不产生振铃），两者分开缩放。缩放后边缘 alpha 可能低于阈值使紧致框
    缩 1–2 px，所以最多修正 3 次目标边长；极少数仍差 1 px 的情况如实保留（实际边长写在 metadata 里）。
    """
    target = long_side
    best = None
    for _ in range(4):
        out = tight_crop(_resize_once(rgba, target), thr)
        got = max(out.shape[:2])
        if best is None or abs(got - long_side) < abs(max(best.shape[:2]) - long_side):
            best = out
        if got == long_side:
            break
        target += long_side - got
        if target < 1:
            break
    return best


def make_cutout(image_rgb: np.ndarray, mask: np.ndarray, max_side: int) -> np.ndarray:
    """原图 + 二值 mask（只含要保留的连通块）→ 紧致裁剪的 RGBA；长边超过 max_side 时缩小。"""
    x1, y1, x2, y2 = mask_bbox(mask)
    rgba = np.dstack([image_rgb[y1:y2, x1:x2], mask[y1:y2, x1:x2].astype(np.uint8) * 255])
    if max(rgba.shape[:2]) > max_side:
        rgba = resize_cutout(rgba, max_side)
    return np.ascontiguousarray(rgba)


def png_bytes(arr: np.ndarray, compress_level: int = 6) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


# =====================================================================
# 候选收集（第一遍只算 mask，不读图像）
# =====================================================================
def _rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


def rle_decode(rle: dict) -> np.ndarray:
    from pycocotools import mask as mu
    c = rle["counts"]
    return mu.decode({"size": rle["size"], "counts": c.encode() if isinstance(c, str) else c}).astype(bool)


def _coco_ann_mask(ann: dict, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mu
    seg = ann["segmentation"]
    if isinstance(seg, list):
        r = mu.merge(mu.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        r = mu.frPyObjects(seg, h, w)
    else:
        r = seg
    return mu.decode(r).astype(bool)


def coco_candidates(cfg: dict, exclude_image_ids: set[int] = frozenset(), limit_images: int | None = None,
                    membership: dict[str, set[int]] | None = None) -> tuple[list[dict], Counter]:
    """扫 COCO val2017 的全部实例，返回通过筛选的候选（含最大连通块的 RLE，不含像素）和漏斗计数。

    exclude_image_ids：整张图排除（splits/coco_holdout.json）。membership：{名字: 图像 id 集合}，只做记录不做排除。
    """
    root = paths.get("data") / "coco"
    d = json.load(open(root / "annotations" / "instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    by_img: dict[int, list] = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    funnel: Counter = Counter()
    out = []
    for img_id in sorted(by_img)[:limit_images]:
        anns = sorted(by_img[img_id], key=lambda a: a["id"])
        info = imgs[img_id]
        H, W = info["height"], info["width"]
        if img_id in exclude_image_ids:
            funnel["instances_in_excluded_holdout_images"] += len(anns)
            continue
        cache: dict[int, np.ndarray] = {}

        def mask_of(a):
            if a["id"] not in cache:
                cache[a["id"]] = _coco_ann_mask(a, H, W)
            return cache[a["id"]]

        for a in anns:
            funnel["instances_total"] += 1
            if a.get("iscrowd", 0):
                funnel["iscrowd"] += 1
                continue
            # 便宜的预筛：标注自带的 area 与解码后的 mask 面积只差在边缘取整上，留一半余量
            if a["area"] < 0.5 * cfg["min_mask_area_px"]:
                funnel["small_area"] += 1
                continue
            m = mask_of(a)
            x, y, w, h = a["bbox"]
            others = []
            for o in anns:
                ox, oy, ow, oh = o["bbox"]
                if o["id"] != a["id"] and ox < x + w and x < ox + ow and oy < y + h and y < oy + oh:
                    others.append(mask_of(o))
            reason, st, big = check_mask(m, others, cfg)
            if reason:
                funnel[reason] += 1
                continue
            funnel["pass"] += 1
            src = {"image_id": img_id, "ann_id": a["id"], "file_name": info["file_name"],
                   "image_wh": [W, H]}
            for k, ids in (membership or {}).items():
                src[k] = img_id in ids
            out.append({"cutout_id": f"coco_{a['id']}", "source": COCO_SOURCE, "category": cats[a["category_id"]],
                        "src": src, "quality": st,
                        "_image_path": str(root / "val2017" / info["file_name"]), "_mask_rle": _rle(big)})
    return out, funnel


def whatsup_candidates(records: list[dict], image_paths: dict[tuple[str, str], str], cfg: dict,
                       source_split: dict[tuple[str, str], str] | None = None) -> tuple[list[dict], Counter]:
    """What's Up 自动标注 → 候选。只用 auto_qc == "pass" 的图；target / reference 各算一个候选。

    records：annotations.jsonl 的每行；image_paths：(dataset, sample_id) → 图像路径；
    source_split：(dataset, sample_id) → 该图在 splits/whatsup_*.json 里属于 dev 还是 test（只做记录）。
    """
    funnel: Counter = Counter()
    out = []
    for rec in sorted(records, key=lambda r: (r["dataset"], r["sample_id"])):
        funnel["whatsup_images_total"] += 1
        if rec.get("auto_qc") != "pass":
            funnel["whatsup_images_not_pass"] += 1
            continue
        key = (rec["dataset"], rec["sample_id"])
        if key not in image_paths:
            funnel["whatsup_images_missing"] += 1
            continue
        objs = rec["objects"]
        masks = {r: rle_decode(o["mask_rle"]) for r, o in objs.items() if o and o.get("mask_rle")}
        for role in ("target", "reference"):
            if role not in masks:
                funnel["whatsup_no_mask"] += 1
                continue
            funnel["instances_total"] += 1
            reason, st, big = check_mask(masks[role], [m for r, m in masks.items() if r != role], cfg)
            if reason:
                funnel[reason] += 1
                continue
            funnel["pass"] += 1
            o = objs[role]
            out.append({"cutout_id": f"wu_{rec['dataset']}_{rec['sample_id']}_{role}", "source": WHATSUP_SOURCE,
                        "category": o["name"].strip().lower(),
                        "src": {"dataset": rec["dataset"], "sample_id": rec["sample_id"], "role": role,
                                "relation": rec.get("relation"), "detector_score": o.get("score"),
                                "auto_qc": rec["auto_qc"], "annotation": "automatic, not human-verified",
                                "source_split": (source_split or {}).get(key),
                                "image_wh": [rec.get("width"), rec.get("height")]},
                        "quality": st, "_image_path": image_paths[key], "_mask_rle": _rle(big)})
    return out, funnel


# =====================================================================
# 截断 / 划分
# =====================================================================
def _rank(seed: int, cutout_id: str) -> str:
    return hashlib.sha256(f"{seed}:{cutout_id}".encode()).hexdigest()


def cap_and_filter(cands: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """每个（来源, 类别）按 sha256(seed:cutout_id) 排序取前 max_per_category 个；合并来源后不足 min_per_category 的类别整类丢掉。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cands:
        groups[(c["source"], c["category"])].append(c)
    kept = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda c: _rank(cfg["seed"], c["cutout_id"]))
        kept += g[: cfg["max_per_category"]]
    per_cat = Counter(c["category"] for c in kept)
    dropped = sorted(k for k, v in per_cat.items() if v < cfg["min_per_category"])
    kept = sorted((c for c in kept if c["category"] not in dropped), key=lambda c: c["cutout_id"])
    return kept, {"n_candidates": len(cands), "n_after_cap": sum(per_cat.values()), "n_kept": len(kept),
                  "dropped_categories_too_few": {k: per_cat[k] for k in dropped}}


def make_split(entries: list[dict], seed: int = SEED) -> dict[str, str]:
    """按 cutout id 对半划分，按类别分层；每个类别用独立且确定的随机流；奇数时多出的 1 个归 test（同 make_splits.py）。"""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e["cutout_id"])
    side = {}
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        random.Random(f"{seed}:synth_cutouts:{cat}").shuffle(ids)
        h = len(ids) // 2
        side.update({i: "dev" for i in ids[:h]})
        side.update({i: "test" for i in ids[h:]})
    return side


# =====================================================================
# 读取
# =====================================================================
class CutoutLibrary:
    """切图库的只读视图。磁盘布局：<root>/library.jsonl（每行一个切图的 metadata）+ <root>/png/<cutout_id>.png。"""

    def __init__(self, root: Path | str | None = None, entries: list[dict] | None = None,
                 arrays: dict[str, np.ndarray] | None = None):
        self.root = Path(root) if root is not None else (library_dir() if entries is None else None)
        if entries is None:
            raw = (self.root / "library.jsonl").read_bytes()
            self.sha256 = hashlib.sha256(raw).hexdigest()
            entries = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        else:                                   # 内存里的库（单元测试用）
            self.sha256 = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        self.entries = sorted(entries, key=lambda e: e["cutout_id"])
        self.by_id = {e["cutout_id"]: e for e in self.entries}
        self._arrays = dict(arrays or {})
        self._resized: dict[tuple[str, int], np.ndarray] = {}

    def categories(self, split: str = "both") -> dict[str, list[str]]:
        """{类别: 排序后的 cutout id}；split ∈ {dev, test, both}。"""
        assert split in ("dev", "test", "both")
        out: dict[str, list[str]] = defaultdict(list)
        for e in self.entries:
            if split == "both" or e["split"] == split:
                out[e["category"]].append(e["cutout_id"])
        return {k: out[k] for k in sorted(out)}

    def rgba(self, cutout_id: str) -> np.ndarray:
        if cutout_id not in self._arrays:
            self._arrays[cutout_id] = np.asarray(Image.open(self.root / "png" / f"{cutout_id}.png").convert("RGBA"))
        return self._arrays[cutout_id]

    def resized(self, cutout_id: str, long_side: int) -> np.ndarray:
        key = (cutout_id, long_side)
        if key not in self._resized:
            self._resized[key] = resize_cutout(self.rgba(cutout_id), long_side)
        return self._resized[key]
