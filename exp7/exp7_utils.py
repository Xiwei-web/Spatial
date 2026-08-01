#!/usr/bin/env python3
"""Utilities for Experiment 7 head ablation analysis.

Experiment 7 follows the head-ablation check from Sec. 3.3.3 of
"Mechanisms of Object Localization in VLMs": attention heads are ranked by
their causal mediation analysis (CMA) scores, then progressively zeroed during
the forward pass to test whether high-MF heads are necessary for the task.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


EXP7_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP7_DIR.parent
EXP6_DIR = PROJECT_ROOT / "exp6"

if str(EXP6_DIR) not in sys.path:
    sys.path.insert(0, str(EXP6_DIR))

from exp6_utils import (  # noqa: E402
    append_jsonl,
    build_localization_prompt,
    find_decoder_layers,
    find_output_projection,
    find_self_attention,
    infer_num_heads,
    load_json,
    load_jsonl,
    parse_yes_no,
    write_json,
)


DEFAULT_FRACTIONS = tuple(round(i / 100.0, 2) for i in range(0, 41))
TASKS = ("relation", "localization")
HEAD_GROUP_LABELS = {
    "localization_critical": "Loc. Important",
    "relation_critical": "Rel. Important",
    "low_importance": "Low Importance",
    "negative": "Negative",
}
IOU_THRESHOLDS = (0.5, 0.7, 0.9)


def parse_task_list(text: str) -> list[str]:
    tasks = [item.strip() for item in text.split(",") if item.strip()]
    for task in tasks:
        if task not in TASKS:
            raise ValueError("Tasks must be comma-separated values from relation,localization")
    return tasks


def parse_group_list(text: str) -> list[str]:
    groups = [item.strip() for item in text.split(",") if item.strip()]
    for group in groups:
        if group not in HEAD_GROUP_LABELS:
            raise ValueError(f"Unsupported head group {group!r}; choose from {sorted(HEAD_GROUP_LABELS)}")
    return groups


def parse_float_csv(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0:
            raise ValueError("Pruned fractions must be non-negative")
        values.append(round(value, 6))
    if not values:
        raise ValueError("At least one pruned fraction is required")
    if 0.0 not in values:
        values = [0.0] + values
    return sorted(set(values))


def exp7_record_key(sample_id: str, task: str, head_group: str, fraction: float) -> str:
    return f"{sample_id}::{task}::{head_group}::{float(fraction):.6f}"


def completed_exp7_keys(path: Path) -> set[str]:
    return {row["exp7_key"] for row in load_jsonl(path) if row.get("exp7_key")}


def load_cma_scores(
    summary_csv: Path,
    metric: str = "mean",
    min_valid_count: int = 1,
) -> dict[str, dict[tuple[int, int], dict[str, float | int]]]:
    """Load task-specific CMA scores from the corrected Exp6 summary CSV."""

    if metric not in {"mean", "median"}:
        raise ValueError("metric must be mean or median")
    value_field = f"{metric}_mediation_fraction"
    scores: dict[str, dict[tuple[int, int], dict[str, float | int]]] = {task: {} for task in TASKS}
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task")
            if task not in scores:
                continue
            try:
                layer = int(row["layer"])
                head = int(row["head"])
                value = float(row[value_field])
                valid_count = int(float(row.get("valid_count") or 0))
                positive_gap_count = int(float(row.get("positive_gap_count") or 0))
                total_count = int(float(row.get("total_count") or 0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or valid_count < min_valid_count:
                continue
            scores[task][(layer, head)] = {
                "score": value,
                "valid_count": valid_count,
                "positive_gap_count": positive_gap_count,
                "total_count": total_count,
            }
    return scores


def _all_model_heads(num_layers: int, num_heads: int) -> list[tuple[int, int]]:
    return [(layer, head) for layer in range(num_layers) for head in range(num_heads)]


def _score_for(scores: dict[str, dict[tuple[int, int], dict[str, float | int]]], task: str, head: tuple[int, int]) -> float:
    item = scores.get(task, {}).get(head)
    return float(item["score"]) if item is not None else 0.0


def rank_heads(
    cma_scores: dict[str, dict[tuple[int, int], dict[str, float | int]]],
    num_layers: int,
    num_heads: int,
) -> dict[str, list[tuple[int, int]]]:
    """Build the four head lists used by progressive ablation.

    The two critical groups are task-specific descending MF lists. Low
    importance ranks heads by the smallest average absolute MF across both
    tasks. Negative ranks heads by the most negative average MF first.
    """

    all_heads = _all_model_heads(num_layers, num_heads)
    loc_scores = cma_scores.get("localization", {})
    rel_scores = cma_scores.get("relation", {})
    rankings = {
        "localization_critical": sorted(
            all_heads,
            key=lambda head: (_score_for(cma_scores, "localization", head), -head[0], -head[1]),
            reverse=True,
        ),
        "relation_critical": sorted(
            all_heads,
            key=lambda head: (_score_for(cma_scores, "relation", head), -head[0], -head[1]),
            reverse=True,
        ),
    }

    def combined_abs(head: tuple[int, int]) -> tuple[float, int, int]:
        values = []
        if head in loc_scores:
            values.append(abs(float(loc_scores[head]["score"])))
        if head in rel_scores:
            values.append(abs(float(rel_scores[head]["score"])))
        return ((sum(values) / len(values)) if values else 0.0, head[0], head[1])

    def combined_signed(head: tuple[int, int]) -> tuple[float, int, int]:
        values = []
        if head in loc_scores:
            values.append(float(loc_scores[head]["score"]))
        if head in rel_scores:
            values.append(float(rel_scores[head]["score"]))
        return ((sum(values) / len(values)) if values else 0.0, head[0], head[1])

    rankings["low_importance"] = sorted(all_heads, key=combined_abs)
    rankings["negative"] = sorted(all_heads, key=combined_signed)
    return rankings


def build_progressive_specs(
    rankings: dict[str, list[tuple[int, int]]],
    fractions: list[float],
    num_layers: int,
    num_heads: int,
    head_groups: list[str],
) -> list[dict[str, Any]]:
    total_heads = num_layers * num_heads
    specs = []
    for group in head_groups:
        ranked = rankings[group]
        for fraction in fractions:
            count = int(round(float(fraction) * total_heads))
            count = max(0, min(count, len(ranked)))
            heads = ranked[:count]
            specs.append(
                {
                    "head_group": group,
                    "head_group_label": HEAD_GROUP_LABELS[group],
                    "fraction_pruned": float(fraction),
                    "num_ablated_heads": count,
                    "ablated_heads": [[layer, head] for layer, head in heads],
                }
            )
    return specs


@contextmanager
def zero_attention_heads(
    model: Any,
    heads: list[list[int]] | list[tuple[int, int]],
    token_scope: str = "prompt_prefill",
):
    """Zero selected attention-head output slices at the o_proj input.

    By default, the hook is active only on the initial prompt prefill pass
    (`tensor.shape[1] > 1`) and not on the later one-token decoding steps.
    This keeps Exp7 closer to Exp6 CMA, where mediation is estimated from
    prompt-token interventions. Use token_scope="all_tokens" for a more
    aggressive full-generation ablation.
    """

    if not heads:
        yield
        return
    if token_scope not in {"prompt_prefill", "all_tokens"}:
        raise ValueError("token_scope must be prompt_prefill or all_tokens")

    import torch

    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer_idx, head_idx in heads:
        by_layer[int(layer_idx)].append(int(head_idx))

    layers = find_decoder_layers(model)
    handles = []

    def make_hook(layer_idx: int, head_indices: list[int]):
        attn = find_self_attention(layers[layer_idx])
        num_heads = infer_num_heads(model, attn)
        valid_heads = sorted(set(head_indices))
        for head_idx in valid_heads:
            if head_idx < 0 or head_idx >= num_heads:
                raise IndexError(f"Head {head_idx} out of range for layer {layer_idx} with {num_heads} heads")

        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            if not args:
                raise RuntimeError("o_proj pre-hook received no input")
            tensor = args[0]
            if not torch.is_tensor(tensor) or tensor.ndim != 3:
                return args
            if token_scope == "prompt_prefill" and tensor.shape[1] <= 1:
                return args
            hidden_size = int(tensor.shape[-1])
            if hidden_size % num_heads != 0:
                raise ValueError(f"Hidden size {hidden_size} is not divisible by num_heads={num_heads}")
            head_dim = hidden_size // num_heads
            patched = tensor.clone()
            for head_idx in valid_heads:
                start = head_idx * head_dim
                patched[:, :, start : start + head_dim] = 0
            return (patched,) + tuple(args[1:])

        return hook

    for layer_idx, head_indices in by_layer.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer {layer_idx} out of range for {len(layers)} layers")
        o_proj = find_output_projection(find_self_attention(layers[layer_idx]))
        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx, head_indices)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def relation_correct(output: str, expected: str = "yes") -> bool:
    return parse_yes_no(output) == expected.strip().lower()


def parse_bbox_from_text(text: str) -> tuple[list[float] | None, str | None]:
    clean = text.replace("\n", " ")
    bracket_match = re.search(r"\[([^\]]+)\]", clean)
    if bracket_match:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", bracket_match.group(1))
        if len(numbers) >= 4:
            return [float(v) for v in numbers[:4]], None

    # Qwen-style structured answers and loose prose both usually contain four
    # coordinate-like numbers; ignore single object counts or labels by taking
    # the first valid quadruple.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", clean)
    if len(numbers) >= 4:
        return [float(v) for v in numbers[:4]], None
    return None, "could not parse four bbox coordinates"


def ordered_bbox(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def convert_bbox_to_pixels(
    box: list[float],
    sample: dict[str, Any],
    coord_mode: str = "pixel",
    resized_width: int | None = None,
    resized_height: int | None = None,
) -> list[float]:
    width = float(sample.get("image_width") or sample.get("width") or 1)
    height = float(sample.get("image_height") or sample.get("height") or 1)
    x1, y1, x2, y2 = ordered_bbox(box)
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))

    mode = coord_mode
    if mode == "auto":
        if max_coord <= 1.5:
            mode = "norm01"
        elif resized_width and resized_height and max_coord <= max(resized_width, resized_height) + 2:
            mode = "resized"
        elif max_coord <= 1000 and (max_coord > max(width, height) * 1.15):
            mode = "norm1000"
        else:
            mode = "pixel"

    if mode == "norm01":
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    if mode == "norm1000":
        return [x1 / 1000.0 * width, y1 / 1000.0 * height, x2 / 1000.0 * width, y2 / 1000.0 * height]
    if mode == "resized":
        if not resized_width or not resized_height:
            raise ValueError("coord_mode=resized requires resized_width and resized_height")
        return [x1 / resized_width * width, y1 / resized_height * height, x2 / resized_width * width, y2 / resized_height * height]
    if mode == "pixel":
        return [x1, y1, x2, y2]
    raise ValueError("coord_mode must be one of auto,pixel,resized,norm01,norm1000")


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = ordered_bbox(box_a)
    bx1, by1, bx2, by2 = ordered_bbox(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def localization_score(iou: float | None) -> float:
    if iou is None:
        return 0.0
    return sum(1.0 if iou >= threshold else 0.0 for threshold in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)


def evaluate_localization_output(
    output: str,
    sample: dict[str, Any],
    coord_mode: str = "pixel",
    resized_width: int | None = None,
    resized_height: int | None = None,
) -> dict[str, Any]:
    raw_box, parse_error = parse_bbox_from_text(output)
    if raw_box is None:
        metrics = {
            "pred_bbox": None,
            "raw_pred_bbox": None,
            "bbox_parse_error": parse_error,
            "target_iou": None,
            "localization_score": 0.0,
        }
    else:
        pred_box = convert_bbox_to_pixels(raw_box, sample, coord_mode, resized_width, resized_height)
        iou = bbox_iou(pred_box, [float(v) for v in sample["target_bbox"]])
        metrics = {
            "pred_bbox": pred_box,
            "raw_pred_bbox": raw_box,
            "bbox_parse_error": None,
            "target_iou": iou,
            "localization_score": localization_score(iou),
        }
    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        metrics[f"iou_at_{suffix}"] = 1 if metrics["target_iou"] is not None and metrics["target_iou"] >= threshold else 0
    return metrics


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_exp7_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["task"], row["head_group"], float(row["fraction_pruned"]))].append(row)

    summary = []
    for (task, head_group, fraction), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        out: dict[str, Any] = {
            "task": task,
            "head_group": head_group,
            "head_group_label": rows[0].get("head_group_label", HEAD_GROUP_LABELS.get(head_group, head_group)),
            "fraction_pruned": fraction,
            "num_ablated_heads": int(rows[0].get("num_ablated_heads", 0)),
            "num_samples": len(rows),
        }
        if task == "relation":
            values = [1.0 if row.get("relation_correct") else 0.0 for row in rows]
            out["relation_accuracy"] = _mean(values)
            out["accuracy"] = out["relation_accuracy"]
            out["unparsed_rate"] = _mean([1.0 if row.get("relation_prediction") is None else 0.0 for row in rows])
        else:
            loc_values = [float(row.get("localization_score") or 0.0) for row in rows]
            out["localization_accuracy"] = _mean(loc_values)
            out["accuracy"] = out["localization_accuracy"]
            ious = [float(row["target_iou"]) for row in rows if isinstance(row.get("target_iou"), (int, float))]
            out["mean_target_iou"] = _mean(ious)
            out["bbox_parse_success_rate"] = _mean([1.0 if row.get("pred_bbox") is not None else 0.0 for row in rows])
            for threshold in IOU_THRESHOLDS:
                suffix = str(threshold).replace(".", "_")
                out[f"iou_at_{suffix}"] = _mean([float(row.get(f"iou_at_{suffix}") or 0.0) for row in rows])
        summary.append(out)
    return summary


def add_auc_to_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_curve: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_curve[(row["task"], row["head_group"])].append(row)
    auc_by_curve = {}
    for key, curve_rows in by_curve.items():
        ordered = sorted(curve_rows, key=lambda row: float(row["fraction_pruned"]))
        area = 0.0
        for left, right in zip(ordered, ordered[1:]):
            x0 = float(left["fraction_pruned"])
            x1 = float(right["fraction_pruned"])
            y0 = float(left.get("accuracy") or 0.0)
            y1 = float(right.get("accuracy") or 0.0)
            area += (x1 - x0) * (y0 + y1) / 2.0
        auc_by_curve[key] = area
    for row in rows:
        row["auc"] = auc_by_curve[(row["task"], row["head_group"])]
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task",
        "head_group",
        "head_group_label",
        "fraction_pruned",
        "num_ablated_heads",
        "num_samples",
        "accuracy",
        "auc",
        "relation_accuracy",
        "unparsed_rate",
        "localization_accuracy",
        "mean_target_iou",
        "bbox_parse_success_rate",
        "iou_at_0_5",
        "iou_at_0_7",
        "iou_at_0_9",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_exp7_curves(rows: list[dict[str, Any]], model_name: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    tasks = [task for task in TASKS if any(row["task"] == task for row in rows)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.5), squeeze=False)
    colors = {
        "localization_critical": "#C44E52",
        "relation_critical": "#4C72B0",
        "low_importance": "#55A868",
        "negative": "#8172B2",
    }
    for ax, task in zip(axes[0], tasks):
        for group in HEAD_GROUP_LABELS:
            curve = sorted(
                [row for row in rows if row["task"] == task and row["head_group"] == group],
                key=lambda row: float(row["fraction_pruned"]),
            )
            if not curve:
                continue
            ax.plot(
                [row["fraction_pruned"] for row in curve],
                [row.get("accuracy") or 0.0 for row in curve],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=HEAD_GROUP_LABELS[group],
                color=colors[group],
            )
        ax.set_title(task.capitalize())
        ax.set_xlabel("Fraction of heads ablated")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(model_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
