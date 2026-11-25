"""统一数据加载器。每个数据集返回 list[Sample]，字段一致：
image_path / target / reference / relation / bbox_* / mask_*（没有则为 None）。

标注格式依据官方仓库 amitakamath/whatsup_vlms 的 dataset_zoo/aro_datasets.py：
caption_options（或列表第 2 项）的第一个选项永远是正确描述。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import paths
from src.data import gqa_boxes


@dataclass
class Sample:
    dataset: str
    sample_id: str               # 数据集内唯一、稳定（划分文件引用它）
    image_path: str
    target: str
    reference: str | None        # one-object 子集没有 reference
    relation: str                # 规范化关系标签
    caption_true: str
    caption_options: list[str]
    bbox_target: list[float] | None = None      # [x1, y1, x2, y2]，原图像素
    bbox_reference: list[float] | None = None
    mask_target: Any | None = None              # COCO segmentation（polygon / RLE）或 None
    mask_reference: Any | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _wu_root() -> Path:
    return paths.get("data") / "whatsup"


# ---------- What's Up Subset A / B ----------
_A_RELS = {"left_of": "left", "right_of": "right", "on": "on", "under": "under"}
_B_RELS = {"left_of": "left", "right_of": "right", "in-front_of": "front", "behind": "behind"}


def _controlled(name: str, ann: str, img_dir: str, rels: dict[str, str]) -> list[Sample]:
    root = _wu_root()
    out = []
    pat = re.compile(r"^(?P<t>[^_]+)_(?P<r>" + "|".join(sorted(map(re.escape, rels), key=len, reverse=True)) + r")_(?P<ref>[^_]+)\.jpeg$")
    for d in json.load(open(root / ann)):
        fn = Path(d["image_path"]).name
        m = pat.match(fn)
        if m is None:
            raise ValueError(f"{name}: 无法解析文件名 {fn}")
        out.append(Sample(
            dataset=name, sample_id=fn[: -len(".jpeg")], image_path=str(root / img_dir / fn),
            target=m["t"].replace("-", " "), reference=m["ref"].replace("-", " "), relation=rels[m["r"]],
            caption_true=d["caption_options"][0], caption_options=list(d["caption_options"]),
        ))
    return out


def whatsup_a() -> list[Sample]:
    return _controlled("whatsup_a", "controlled_images_dataset.json", "controlled_images", _A_RELS)


def whatsup_b() -> list[Sample]:
    return _controlled("whatsup_b", "controlled_clevr_dataset.json", "controlled_clevr", _B_RELS)


# ---------- COCO-spatial / GQA-spatial ----------
_ONE = re.compile(r"^A photo of an? (?P<t>.+) on the (?P<r>left|right|top|bottom|front|behind)$")
_TWO = re.compile(r"^A photo of an? (?P<t>.+) to the (?P<r>left|right|front|behind|top) of an? (?P<ref>.+)$|"
                  r"^A photo of an? (?P<t2>.+) (?P<r2>above|below) an? (?P<ref2>.+)$")


def _parse_caption(cap: str) -> tuple[str, str | None, str]:
    m = _ONE.match(cap)
    if m:
        return m["t"], None, m["r"]
    m = _TWO.match(cap)
    if m:
        if m["t"] is not None:
            return m["t"], m["ref"], m["r"]
        return m["t2"], m["ref2"], m["r2"]
    raise ValueError(f"无法解析 caption: {cap!r}")


@lru_cache(maxsize=1)
def _coco_index():
    f = paths.get("data") / "coco" / "annotations" / "instances_val2017.json"
    if not f.is_file():
        return None
    d = json.load(open(f))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by_img: dict[int, dict[str, list]] = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], {}).setdefault(cats[a["category_id"]], []).append(a)
    sizes = {im["id"]: (im["height"], im["width"]) for im in d["images"]}
    return by_img, sizes


def _coco_box_mask(image_id: int, name: str | None):
    """该类别在图中恰好只有一个非 crowd 实例时才给框和 mask，否则返回 None（不猜）。
    返回 (真值框 = mask 外接框, mask, COCO 自带的框)；真值框的规定见 configs/annotation/object_conventions.yaml。"""
    idx = _coco_index()
    if idx is None or name is None:
        return None, None, None
    by_img, sizes = idx
    anns = [a for a in by_img.get(image_id, {}).get(name, []) if not a.get("iscrowd", 0)]
    if len(anns) != 1:
        return None, None, None
    from src.annotate.masks import coco_segmentation_bbox
    x, y, w, h = anns[0]["bbox"]
    seg = anns[0]["segmentation"]
    return coco_segmentation_bbox(seg, *sizes[image_id]), seg, [x, y, x + w, y + h]


def _qa(name: str, ann: str, image_fmt, coco: bool) -> list[Sample]:
    out = []
    for i, (img_id, cap_true, cap_false) in enumerate(json.load(open(_wu_root() / ann))):
        t, ref, rel = _parse_caption(cap_true)
        bt = mt = br = mr = None
        sid = f"{i:05d}_{img_id}"
        extra = {}
        if coco:
            bt, mt, extra["coco_bbox_target"] = _coco_box_mask(int(img_id), t)
            br, mr, extra["coco_bbox_reference"] = _coco_box_mask(int(img_id), ref)
        else:
            # GQA-spatial：框来自 GQA scene graph 的名称匹配（scripts/build_gqa_boxes.py 预先算好）。
            # 只有唯一匹配才有框；其余样本保留、框为 None，原因写在 extra['box_status']。没有 mask。
            bt, br, extra["box_status"] = gqa_boxes.lookup(name, sid)
        out.append(Sample(
            dataset=name, sample_id=sid, image_path=str(image_fmt(img_id)),
            target=t, reference=ref, relation=rel, caption_true=cap_true, caption_options=[cap_true, cap_false],
            bbox_target=bt, bbox_reference=br, mask_target=mt, mask_reference=mr, extra=extra,
        ))
    return out


def _coco_img(i):
    return paths.get("data") / "coco" / "val2017" / f"{int(i):012d}.jpg"


def _vg_img(i):
    return _wu_root() / "vg_images" / f"{i}.jpg"


def coco_spatial_one():
    return _qa("coco_spatial_one", "coco_qa_one_obj.json", _coco_img, True)


def coco_spatial_two():
    return _qa("coco_spatial_two", "coco_qa_two_obj.json", _coco_img, True)


def gqa_spatial_one():
    return _qa("gqa_spatial_one", "vg_qa_one_obj.json", _vg_img, False)


def gqa_spatial_two():
    return _qa("gqa_spatial_two", "vg_qa_two_obj.json", _vg_img, False)


DATASETS = {
    "whatsup_a": whatsup_a, "whatsup_b": whatsup_b,
    "coco_spatial_one": coco_spatial_one, "coco_spatial_two": coco_spatial_two,
    "gqa_spatial_one": gqa_spatial_one, "gqa_spatial_two": gqa_spatial_two,
}


def load_dataset(name: str) -> list[Sample]:
    return DATASETS[name]()


@lru_cache(maxsize=1)
def excluded_samples() -> frozenset:
    """更正表（annotations/whatsup/corrections.yaml）里 action=exclude 的 (dataset, sample_id)。"""
    import yaml
    f = paths.REPO_ROOT / "annotations" / "whatsup" / "corrections.yaml"
    if not f.is_file():
        return frozenset()
    return frozenset((c["dataset"], c["sample_id"]) for c in yaml.safe_load(open(f))["corrections"] if c["action"] == "exclude")


def load_split(name: str, split: str) -> list[Sample]:
    """split ∈ {dev, test}。选择类操作只用 dev，报告数字只用 test。
    更正表里标为 exclude 的样本会被剔除（划分文件本身不变；load_dataset 仍返回原始发布的全部样本）。"""
    assert split in ("dev", "test")
    from .splits import read_split
    ids = read_split(name)[split]
    by_id = {s.sample_id: s for s in load_dataset(name)}
    return [by_id[i] for i in ids if (name, i) not in excluded_samples()]
