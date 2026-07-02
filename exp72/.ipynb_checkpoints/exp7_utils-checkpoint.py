#!/usr/bin/env python3
"""Shared utilities for Experiment 7 head ablation.

Experiment 7 is the necessity-test counterpart of Experiment 6 CMA. It ranks
attention heads by Experiment 6 mediation fraction, progressively zeros selected
head outputs in the normal source-image forward pass, and measures relation and
localization performance degradation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP6_DIR = PROJECT_ROOT / "exp6"
EXP7_DIR = PROJECT_ROOT / "exp7"
EXP1_DIR = PROJECT_ROOT / "exp1"
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))
if str(EXP6_DIR) not in sys.path:
    sys.path.insert(0, str(EXP6_DIR))

from exp1_utils import box_iou, parse_bbox  # noqa: E402
from exp6_utils import (  # noqa: E402
    build_localization_prompt,
    find_decoder_layers,
    find_output_projection,
    find_self_attention,
    infer_num_heads,
    load_jsonl,
    parse_yes_no,
    write_json,
)


IOU_THRESHOLDS = (0.5, 0.7, 0.9)
DEFAULT_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
HEAD_GROUPS = ("localization_critical", "relation_critical", "low_importance", "negative")
HEAD_GROUP_LABELS = {
    "localization_critical": "Loc. Important",
    "relation_critical": "Rel. Important",
    "low_importance": "Low Importance",
    "negative": "Negative",
}
HEAD_GROUP_COLORS = {
    "localization_critical": "#0072B2",
    "relation_critical": "#E69F00",
    "low_importance": "#009E73",
    "negative": "#CC79A7",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_float_csv(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_cma_scores(summary_csv: Path | None, cma_jsonl: Path | None) -> dict[str, dict[tuple[int, int], float]]:
    """Load mean MF per task/layer/head from Experiment 6 outputs."""

    if summary_csv is not None and summary_csv.is_file():
        scores: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mf = safe_float(row.get("mean_mediation_fraction"))
                if mf is None:
                    continue
                task = row["task"]
                layer = int(row["layer"])
                head = int(row["head"])
                scores[task][(layer, head)] = mf
        if scores:
            return dict(scores)

    if cma_jsonl is None or not cma_jsonl.is_file():
        raise FileNotFoundError(f"Could not find CMA summary CSV or JSONL: {summary_csv}, {cma_jsonl}")

    by_key: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in load_jsonl(cma_jsonl):
        mf = safe_float(row.get("mediation_fraction"))
        if mf is None:
            continue
        by_key[(row["task"], int(row["layer"]), int(row["head"]))].append(mf)

    scores = defaultdict(dict)
    for (task, layer, head), vals in by_key.items():
        scores[task][(layer, head)] = sum(vals) / len(vals)
    return dict(scores)


def aggregate_head_scores(
    scores: dict[str, dict[tuple[int, int], float]],
    all_heads: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    aggregate = {}
    for head in all_heads:
        vals = [task_scores[head] for task_scores in scores.values() if head in task_scores]
        if vals:
            aggregate[head] = sum(vals) / len(vals)
    return aggregate


def rank_heads(
    scores: dict[str, dict[tuple[int, int], float]],
    num_layers: int,
    num_heads: int,
) -> dict[str, list[tuple[int, int]]]:
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    loc_scores = scores.get("localization", {})
    rel_scores = scores.get("relation", {})
    aggregate = aggregate_head_scores(scores, all_heads)

    loc_rank = sorted(loc_scores, key=lambda item: loc_scores[item], reverse=True)
    rel_rank = sorted(rel_scores, key=lambda item: rel_scores[item], reverse=True)
    low_rank = sorted(aggregate, key=lambda item: abs(aggregate[item]))
    negative_candidates = [item for item in aggregate if aggregate[item] < 0]
    negative_rank = sorted(negative_candidates, key=lambda item: aggregate[item])
    if not negative_rank:
        negative_rank = sorted(aggregate, key=lambda item: aggregate[item])

    return {
        "localization_critical": loc_rank,
        "relation_critical": rel_rank,
        "low_importance": low_rank,
        "negative": negative_rank,
    }


def head_count_for_fraction(num_layers: int, num_heads: int, fraction: float) -> int:
    total = num_layers * num_heads
    if fraction <= 0:
        return 0
    return max(1, min(total, int(math.ceil(total * fraction))))


def build_progressive_specs(
    rankings: dict[str, list[tuple[int, int]]],
    fractions: list[float],
    num_layers: int,
    num_heads: int,
) -> list[dict[str, Any]]:
    specs = []
    for group in HEAD_GROUPS:
        ranking = rankings[group]
        for fraction in [0.0] + fractions:
            k = head_count_for_fraction(num_layers, num_heads, fraction)
            selected = ranking[: min(k, len(ranking))]
            specs.append(
                {
                    "head_group": group,
                    "head_group_label": HEAD_GROUP_LABELS[group],
                    "fraction_pruned": float(fraction),
                    "num_ablated_heads": len(selected),
                    "ablated_heads": [[int(layer), int(head)] for layer, head in selected],
                }
            )
    return specs


@contextmanager
def zero_attention_heads(model: Any, selected_heads: list[list[int]] | list[tuple[int, int]]):
    """Zero selected attention-head output slices before the attention output projection."""

    import torch

    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected_heads:
        heads_by_layer[int(layer)].append(int(head))

    if not heads_by_layer:
        yield
        return

    layers = find_decoder_layers(model)
    handles = []

    def make_hook(layer_idx: int, head_indices: list[int], num_heads: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            if not args:
                raise RuntimeError("o_proj pre-hook received no input")
            tensor = args[0]
            if tensor.ndim != 3:
                raise ValueError(f"Expected o_proj input [B, S, D], got shape {tuple(tensor.shape)}")
            hidden_size = int(tensor.shape[-1])
            if hidden_size % num_heads != 0:
                raise ValueError(f"Hidden size {hidden_size} is not divisible by num_heads={num_heads}")
            head_dim = hidden_size // num_heads
            out = tensor.clone()
            for head_idx in head_indices:
                if head_idx < 0 or head_idx >= num_heads:
                    raise IndexError(f"Head {head_idx} out of range for layer {layer_idx} with {num_heads} heads")
                start = head_idx * head_dim
                end = start + head_dim
                out[:, :, start:end] = torch.zeros_like(out[:, :, start:end])
            return (out,) + tuple(args[1:])

        return hook

    for layer_idx, head_indices in heads_by_layer.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer {layer_idx} out of range for {len(layers)} decoder layers")
        attn = find_self_attention(layers[layer_idx])
        o_proj = find_output_projection(attn)
        num_heads = infer_num_heads(model, attn)
        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx, head_indices, num_heads)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def localization_score(iou: float | None) -> float:
    if iou is None:
        return 0.0
    return sum(1.0 for threshold in IOU_THRESHOLDS if iou >= threshold) / len(IOU_THRESHOLDS)


def evaluate_localization_output(text: str, sample: dict[str, Any], coord_mode: str) -> dict[str, Any]:
    width = int(sample.get("image_width") or 0)
    height = int(sample.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return {
            "pred_bbox": None,
            "bbox_parse_error": "missing_image_size",
            "target_iou": None,
            "localization_score": 0.0,
        }
    pred_bbox, error = parse_bbox(text, width=width, height=height, coord_mode=coord_mode)
    iou = box_iou(pred_bbox, sample.get("target_bbox"))
    return {
        "pred_bbox": pred_bbox,
        "bbox_parse_error": error,
        "target_iou": iou,
        "localization_score": localization_score(iou),
        "iou_at_0_5": int(iou is not None and iou >= 0.5),
        "iou_at_0_7": int(iou is not None and iou >= 0.7),
        "iou_at_0_9": int(iou is not None and iou >= 0.9),
    }


def relation_correct(output_text: str) -> bool:
    return parse_yes_no(output_text) == "yes"


def exp7_record_key(sample_id: str, task: str, head_group: str, fraction: float) -> str:
    return f"{sample_id}::{task}::{head_group}::{fraction:.4f}"


def completed_exp7_keys(path: Path) -> set[str]:
    return {row["exp7_key"] for row in load_jsonl(path) if row.get("exp7_key")}


def safe_mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def summarize_exp7_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["head_group"], float(row["fraction_pruned"]), row["task"])].append(row)

    summaries = []
    for (group, fraction, task), rows in sorted(grouped.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        base = {
            "head_group": group,
            "head_group_label": HEAD_GROUP_LABELS.get(group, group),
            "fraction_pruned": fraction,
            "task": task,
            "num_samples": len(rows),
            "num_ablated_heads": rows[0].get("num_ablated_heads", 0),
        }
        if task == "relation":
            vals = [1.0 if row.get("relation_correct") else 0.0 for row in rows]
            base["accuracy"] = safe_mean(vals)
            base["relation_accuracy"] = base["accuracy"]
        elif task == "localization":
            base["accuracy"] = safe_mean([row.get("localization_score") for row in rows])
            base["localization_accuracy"] = base["accuracy"]
            base["mean_target_iou"] = safe_mean([row.get("target_iou") for row in rows])
            base["iou_at_0_5"] = safe_mean([row.get("iou_at_0_5") for row in rows])
            base["iou_at_0_7"] = safe_mean([row.get("iou_at_0_7") for row in rows])
            base["iou_at_0_9"] = safe_mean([row.get("iou_at_0_9") for row in rows])
        summaries.append(base)
    return summaries


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def auc_for_curve(points: list[tuple[float, float]]) -> float | None:
    clean = sorted((float(x), float(y)) for x, y in points if y is not None and math.isfinite(float(y)))
    if len(clean) < 2:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(clean[:-1], clean[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def add_auc_to_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_curve: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        acc = row.get("accuracy")
        if acc is not None:
            by_curve[(row["task"], row["head_group"])].append((float(row["fraction_pruned"]), float(acc)))
    aucs = {key: auc_for_curve(points) for key, points in by_curve.items()}
    out = []
    for row in rows:
        new_row = dict(row)
        new_row["auc"] = aucs.get((row["task"], row["head_group"]))
        out.append(new_row)
    return out


def plot_exp7_curves(summary_rows: list[dict[str, Any]], model_name: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = ["relation", "localization"]
    titles = {"relation": "Relation", "localization": "Localization"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    axes = list(np.asarray(axes).ravel())

    for ax, task in zip(axes, tasks):
        task_rows = [row for row in summary_rows if row.get("task") == task]
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_group[row["head_group"]].append(row)
        for group in HEAD_GROUPS:
            rows = sorted(by_group.get(group, []), key=lambda row: float(row["fraction_pruned"]))
            if not rows:
                continue
            xs = [float(row["fraction_pruned"]) for row in rows]
            ys = [row.get("accuracy") for row in rows]
            auc = rows[0].get("auc")
            label = HEAD_GROUP_LABELS.get(group, group)
            if auc is not None:
                label = f"{label} AUC={float(auc):.2f}"
            ax.plot(xs, ys, marker="o", linewidth=2, color=HEAD_GROUP_COLORS.get(group), label=label)
        ax.set_title(f"{model_name} {titles[task]}")
        ax.set_xlabel("Percentage pruned")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.35)
        ax.set_xlim(-0.01, 0.41)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
