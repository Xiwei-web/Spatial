#!/usr/bin/env python3
"""
Shared utilities for Experiment 1.

Experiment 1 evaluates whether relation prediction depends on target/reference
localization. The ground-truth boxes are expected to be xyxy boxes from
whatsUP/bbox_mask_xyxy.json.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "whatsUP"
DEFAULT_QA_PATH = DEFAULT_DATA_ROOT / "qa.json"
DEFAULT_BBOX_MASK_XYXY_PATH = DEFAULT_DATA_ROOT / "bbox_mask_xyxy.json"

OPTION_LETTERS = ("A", "B", "C", "D")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_completed_sample_ids(path: Path) -> set[str]:
    return {row["sample_id"] for row in load_jsonl(path) if "sample_id" in row}


def canonical_relation_from_text(text: str) -> str | None:
    text_l = text.lower().replace("-", " ")
    if "left" in text_l:
        return "left_of"
    if "right" in text_l:
        return "right_of"
    if "in front" in text_l or "front of" in text_l:
        return "in_front_of"
    if "behind" in text_l:
        return "behind"
    if "under" in text_l or "below" in text_l:
        return "under"
    if " on " in f" {text_l} " or "on top" in text_l or "above" in text_l:
        return "on"
    return None


def canonical_relation_from_qa(qa_item: dict[str, Any]) -> str | None:
    answer = qa_item.get("answer")
    options = qa_item.get("options", {})
    if answer in options:
        relation = canonical_relation_from_text(options[answer])
        if relation:
            return relation

    image_stem = Path(qa_item.get("image_path", "")).stem
    target = qa_item.get("target", "")
    reference = qa_item.get("reference", "")
    prefix = f"{target}_"
    suffix = f"_{reference}"
    if image_stem.startswith(prefix) and image_stem.endswith(suffix):
        return image_stem[len(prefix) : -len(suffix)]
    return None


def build_relation_prompt(qa_item: dict[str, Any]) -> str:
    question = qa_item["question"].strip()
    options = qa_item["options"]
    option_lines = "\n".join(f"{letter}: {options[letter]}" for letter in sorted(options))
    return (
        f"{question}\n"
        f"{option_lines}\n"
        "Answer with only one option letter: A, B, C, or D."
    )


def build_localization_prompt(object_name: str) -> str:
    return (
        f"Please provide the bounding box coordinates of the {object_name}.\n"
        "Format: [x1, y1, x2, y2]."
    )


def load_exp1_samples(
    data_root: Path = DEFAULT_DATA_ROOT,
    qa_path: Path | None = None,
    bbox_mask_path: Path | None = None,
) -> list[dict[str, Any]]:
    qa_path = qa_path or (data_root / "qa.json")
    bbox_mask_path = bbox_mask_path or (data_root / "bbox_mask_xyxy.json")

    qa_data = load_json(qa_path)
    coco = load_json(bbox_mask_path)

    image_by_file = {image["file_name"]: image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    samples = []
    for index, qa_item in enumerate(qa_data):
        image_rel_path = qa_item["image_path"]
        if image_rel_path not in image_by_file:
            raise KeyError(f"Image path from qa.json not found in bbox_mask_xyxy.json: {image_rel_path}")

        image_info = image_by_file[image_rel_path]
        anns = annotations_by_image[image_info["id"]]
        target_anns = [ann for ann in anns if ann["category_id"] == 1]
        reference_anns = [ann for ann in anns if ann["category_id"] == 2]
        if len(target_anns) != 1 or len(reference_anns) != 1:
            raise ValueError(
                f"Expected one target and one reference annotation for {image_rel_path}; "
                f"got target={len(target_anns)}, reference={len(reference_anns)}"
            )

        target_ann = target_anns[0]
        reference_ann = reference_anns[0]
        qa_id = qa_item.get("id", index + 1)
        source_dataset = qa_item.get("source_dataset", image_info.get("source_dataset", "unknown"))
        sample_id = f"whatsup_{source_dataset}_{int(qa_id):06d}"

        samples.append(
            {
                "sample_id": sample_id,
                "qa_id": qa_id,
                "source_dataset": source_dataset,
                "image_path": str(data_root / image_rel_path),
                "image_rel_path": image_rel_path,
                "image_width": image_info["width"],
                "image_height": image_info["height"],
                "target_name": qa_item["target"],
                "reference_name": qa_item["reference"],
                "question": qa_item["question"],
                "options": qa_item["options"],
                "answer": qa_item["answer"],
                "gt_relation": canonical_relation_from_qa(qa_item),
                "target_bbox": [float(v) for v in target_ann["bbox"]],
                "reference_bbox": [float(v) for v in reference_ann["bbox"]],
                "target_mask_path": target_ann.get("attributes", {}).get("mask_path"),
                "reference_mask_path": reference_ann.get("attributes", {}).get("mask_path"),
                "target_annotation_id": target_ann["id"],
                "reference_annotation_id": reference_ann["id"],
            }
        )
    return samples


def extract_first_number_group(text: str) -> list[float] | None:
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    labeled = re.search(
        rf"x1\s*[:=]\s*({number}).*?y1\s*[:=]\s*({number}).*?"
        rf"x2\s*[:=]\s*({number}).*?y2\s*[:=]\s*({number})",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labeled:
        return [float(v) for v in labeled.groups()]

    bracketed = re.search(rf"[\[\(]\s*({number})\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*[\]\)]", text)
    if bracketed:
        return [float(v) for v in bracketed.groups()]

    nums = re.findall(number, text)
    if len(nums) >= 4:
        return [float(v) for v in nums[:4]]
    return None


def normalize_bbox(
    raw_box: list[float] | None,
    width: int,
    height: int,
    coord_mode: str = "auto",
) -> list[float] | None:
    if raw_box is None or len(raw_box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in raw_box]

    if coord_mode not in {"auto", "pixel", "norm01", "norm1000"}:
        raise ValueError(f"Unsupported coord_mode: {coord_mode}")

    if coord_mode == "norm01" or (
        coord_mode == "auto" and all(-0.05 <= v <= 1.05 for v in (x1, y1, x2, y2))
    ):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif coord_mode == "norm1000" or (
        coord_mode == "auto"
        and all(0 <= v <= 1000 for v in (x1, y1, x2, y2))
        and (y1 > height or y2 > height or x1 > width or x2 > width)
    ):
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_bbox(text: str, width: int, height: int, coord_mode: str = "auto") -> tuple[list[float] | None, str | None]:
    raw_box = extract_first_number_group(text)
    if raw_box is None:
        return None, "no_four_number_bbox_found"
    box = normalize_bbox(raw_box, width=width, height=height, coord_mode=coord_mode)
    if box is None:
        return None, f"invalid_bbox_after_normalization:{raw_box}"
    return box, None


def parse_option(text: str, options: dict[str, str]) -> str | None:
    clean = text.strip()
    upper = clean.upper()

    match = re.match(r"^\s*([ABCD])\s*[\.\):]?\s*$", upper)
    if match:
        return match.group(1)

    match = re.match(r"^\s*([ABCD])\s*[\.\):]", upper)
    if match:
        return match.group(1)

    match = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|-)?\s*([ABCD])\b", upper)
    if match:
        return match.group(1)

    lower = clean.lower()
    for letter, option_text in options.items():
        if option_text.lower() in lower:
            return letter

    relation = canonical_relation_from_text(lower)
    if relation:
        for letter, option_text in options.items():
            if canonical_relation_from_text(option_text) == relation:
                return letter
    return None


def box_iou(box_a: list[float] | None, box_b: list[float] | None) -> float | None:
    if box_a is None or box_b is None:
        return None
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return None
    return inter / union


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_diag(box: list[float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def center_distance(pred_box: list[float] | None, gt_box: list[float] | None) -> float | None:
    if pred_box is None or gt_box is None:
        return None
    px, py = box_center(pred_box)
    gx, gy = box_center(gt_box)
    return math.hypot(px - gx, py - gy)


def relation_from_boxes(
    target_box: list[float] | None,
    reference_box: list[float] | None,
    gt_relation: str | None = None,
) -> str | None:
    if target_box is None or reference_box is None:
        return None
    tx, ty = box_center(target_box)
    rx, ry = box_center(reference_box)
    dx, dy = tx - rx, ty - ry

    if gt_relation in {"left_of", "right_of"}:
        return "right_of" if dx > 0 else "left_of"
    if gt_relation in {"in_front_of", "behind"}:
        return "in_front_of" if dy > 0 else "behind"

    if abs(dx) >= abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "in_front_of" if dy > 0 else "behind"


def compute_object_features(sample: dict[str, Any]) -> dict[str, float | None]:
    width = sample["image_width"]
    height = sample["image_height"]
    image_area = float(width * height)
    t_box = sample["target_bbox"]
    r_box = sample["reference_bbox"]
    t_area = max(0.0, t_box[2] - t_box[0]) * max(0.0, t_box[3] - t_box[1])
    r_area = max(0.0, r_box[2] - r_box[0]) * max(0.0, r_box[3] - r_box[1])
    tx, ty = box_center(t_box)
    rx, ry = box_center(r_box)
    image_diag = math.hypot(width, height)
    return {
        "target_area_ratio": t_area / image_area if image_area else None,
        "reference_area_ratio": r_area / image_area if image_area else None,
        "target_diag": box_diag(t_box),
        "reference_diag": box_diag(r_box),
        "object_center_distance": math.hypot(tx - rx, ty - ry),
        "object_center_distance_norm": math.hypot(tx - rx, ty - ry) / image_diag if image_diag else None,
    }


def build_result_record(
    sample: dict[str, Any],
    pred_target_text: str,
    pred_reference_text: str,
    pred_relation_text: str,
    coord_mode: str = "auto",
) -> dict[str, Any]:
    width = sample["image_width"]
    height = sample["image_height"]

    pred_target_bbox, target_parse_error = parse_bbox(pred_target_text, width, height, coord_mode)
    pred_reference_bbox, reference_parse_error = parse_bbox(pred_reference_text, width, height, coord_mode)
    pred_option = parse_option(pred_relation_text, sample["options"])

    target_iou = box_iou(pred_target_bbox, sample["target_bbox"])
    reference_iou = box_iou(pred_reference_bbox, sample["reference_bbox"])
    target_center_dist = center_distance(pred_target_bbox, sample["target_bbox"])
    reference_center_dist = center_distance(pred_reference_bbox, sample["reference_bbox"])

    pred_relation_from_boxes = relation_from_boxes(
        pred_target_bbox,
        pred_reference_bbox,
        gt_relation=sample["gt_relation"],
    )
    gt_relation_from_boxes = relation_from_boxes(
        sample["target_bbox"],
        sample["reference_bbox"],
        gt_relation=sample["gt_relation"],
    )

    object_features = compute_object_features(sample)
    answer = sample["answer"]

    return {
        "sample_id": sample["sample_id"],
        "qa_id": sample["qa_id"],
        "source_dataset": sample["source_dataset"],
        "image_path": sample["image_rel_path"],
        "image_width": width,
        "image_height": height,
        "target_name": sample["target_name"],
        "reference_name": sample["reference_name"],
        "options": sample["options"],
        "gt_answer": answer,
        "gt_relation": sample["gt_relation"],
        "gt_relation_text": sample["options"].get(answer),
        "gt_target_bbox": sample["target_bbox"],
        "gt_reference_bbox": sample["reference_bbox"],
        "target_mask_path": sample["target_mask_path"],
        "reference_mask_path": sample["reference_mask_path"],
        "pred_target_raw": pred_target_text,
        "pred_reference_raw": pred_reference_text,
        "pred_relation_raw": pred_relation_text,
        "pred_target_bbox": pred_target_bbox,
        "pred_reference_bbox": pred_reference_bbox,
        "target_parse_error": target_parse_error,
        "reference_parse_error": reference_parse_error,
        "pred_option": pred_option,
        "pred_relation_text": sample["options"].get(pred_option) if pred_option else None,
        "pred_relation": canonical_relation_from_text(sample["options"].get(pred_option, "")) if pred_option else None,
        "rel_correct": pred_option == answer,
        "target_iou": target_iou,
        "reference_iou": reference_iou,
        "target_center_dist": target_center_dist,
        "reference_center_dist": reference_center_dist,
        "pred_relation_from_boxes": pred_relation_from_boxes,
        "gt_relation_from_boxes": gt_relation_from_boxes,
        "relative_order_correct": pred_relation_from_boxes == gt_relation_from_boxes,
        **object_features,
    }


def safe_mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compute_dependency_table(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    table = []
    for loc_t in (True, False):
        for loc_r in (True, False):
            subset = []
            for row in records:
                t_ok = row.get("target_iou") is not None and row["target_iou"] >= threshold
                r_ok = row.get("reference_iou") is not None and row["reference_iou"] >= threshold
                if t_ok == loc_t and r_ok == loc_r:
                    subset.append(row)
            count = len(subset)
            rel_acc = safe_mean([1.0 if row.get("rel_correct") else 0.0 for row in subset])
            table.append(
                {
                    "loc_t": "correct" if loc_t else "wrong",
                    "loc_r": "correct" if loc_r else "wrong",
                    "count": count,
                    "rel_acc": rel_acc,
                    "mean_target_iou": safe_mean([row.get("target_iou") for row in subset]),
                    "mean_reference_iou": safe_mean([row.get("reference_iou") for row in subset]),
                    "relative_order_acc": safe_mean(
                        [1.0 if row.get("relative_order_correct") else 0.0 for row in subset]
                    ),
                }
            )
    return table


def compute_relation_breakdown(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[row.get("gt_relation") or "unknown"].append(row)
    output = []
    for relation, rows in sorted(by_relation.items()):
        output.append(
            {
                "relation": relation,
                "count": len(rows),
                "rel_acc": safe_mean([1.0 if row.get("rel_correct") else 0.0 for row in rows]),
                "mean_target_iou": safe_mean([row.get("target_iou") for row in rows]),
                "mean_reference_iou": safe_mean([row.get("reference_iou") for row in rows]),
                "relative_order_acc": safe_mean(
                    [1.0 if row.get("relative_order_correct") else 0.0 for row in rows]
                ),
            }
        )
    return output


def write_analysis_outputs(records: list[dict[str, Any]], output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": model_name,
        "num_records": len(records),
        "overall": {
            "rel_acc": safe_mean([1.0 if row.get("rel_correct") else 0.0 for row in records]),
            "mean_target_iou": safe_mean([row.get("target_iou") for row in records]),
            "mean_reference_iou": safe_mean([row.get("reference_iou") for row in records]),
            "target_parse_success_rate": safe_mean(
                [1.0 if row.get("pred_target_bbox") is not None else 0.0 for row in records]
            ),
            "reference_parse_success_rate": safe_mean(
                [1.0 if row.get("pred_reference_bbox") is not None else 0.0 for row in records]
            ),
            "relation_parse_success_rate": safe_mean(
                [1.0 if row.get("pred_option") is not None else 0.0 for row in records]
            ),
            "relative_order_acc": safe_mean(
                [1.0 if row.get("relative_order_correct") else 0.0 for row in records]
            ),
        },
        "dependency_tables": {
            "iou_0.5": compute_dependency_table(records, 0.5),
            "iou_0.7": compute_dependency_table(records, 0.7),
            "iou_0.9": compute_dependency_table(records, 0.9),
        },
        "relation_breakdown": compute_relation_breakdown(records),
    }
    write_json(output_dir / f"{model_name}_summary.json", summary)
    write_regression_csv(records, output_dir / f"{model_name}_analysis_rows.csv")


def write_regression_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample_id",
        "source_dataset",
        "gt_relation",
        "rel_correct",
        "target_iou",
        "reference_iou",
        "loc_t_iou_0_5",
        "loc_r_iou_0_5",
        "loc_t_iou_0_7",
        "loc_r_iou_0_7",
        "loc_t_iou_0_9",
        "loc_r_iou_0_9",
        "relative_order_correct",
        "target_area_ratio",
        "reference_area_ratio",
        "object_center_distance_norm",
        "pred_option",
        "gt_answer",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "source_dataset": row.get("source_dataset"),
                    "gt_relation": row.get("gt_relation"),
                    "rel_correct": int(bool(row.get("rel_correct"))),
                    "target_iou": row.get("target_iou"),
                    "reference_iou": row.get("reference_iou"),
                    "loc_t_iou_0_5": int(row.get("target_iou") is not None and row["target_iou"] >= 0.5),
                    "loc_r_iou_0_5": int(row.get("reference_iou") is not None and row["reference_iou"] >= 0.5),
                    "loc_t_iou_0_7": int(row.get("target_iou") is not None and row["target_iou"] >= 0.7),
                    "loc_r_iou_0_7": int(row.get("reference_iou") is not None and row["reference_iou"] >= 0.7),
                    "loc_t_iou_0_9": int(row.get("target_iou") is not None and row["target_iou"] >= 0.9),
                    "loc_r_iou_0_9": int(row.get("reference_iou") is not None and row["reference_iou"] >= 0.9),
                    "relative_order_correct": int(bool(row.get("relative_order_correct"))),
                    "target_area_ratio": row.get("target_area_ratio"),
                    "reference_area_ratio": row.get("reference_area_ratio"),
                    "object_center_distance_norm": row.get("object_center_distance_norm"),
                    "pred_option": row.get("pred_option"),
                    "gt_answer": row.get("gt_answer"),
                }
            )
