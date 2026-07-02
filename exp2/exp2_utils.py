#!/usr/bin/env python3
"""Shared utilities for Experiment 2 object-token ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(".")
EXP1_DIR = PROJECT_ROOT / "exp1"
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

from exp1_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    append_jsonl,
    build_localization_prompt,
    build_relation_prompt,
    build_result_record,
    load_exp1_samples,
    load_jsonl,
    safe_mean,
    write_json,
)


IOU_THRESHOLDS = (0.5, 0.7, 0.9)
ABLATION_SCOPES = ("target", "reference", "target_reference")
PADDING_VARIANTS = (
    ("object", 0),
    ("minus_1_padding", -1),
    ("minus_2_padding", -2),
    ("plus_1_padding", 1),
    ("plus_2_padding", 2),
)


def select_samples(samples: list[dict[str, Any]], start_index: int, max_samples: int | None) -> list[dict[str, Any]]:
    selected = samples[start_index:]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def condition_name(scope: str, variant: str) -> str:
    return f"{scope}_{variant}"


def object_condition_specs() -> list[dict[str, Any]]:
    specs = [{"condition": "baseline", "kind": "baseline", "scope": None, "padding": None}]
    for scope in ABLATION_SCOPES:
        for variant, padding in PADDING_VARIANTS:
            specs.append(
                {
                    "condition": condition_name(scope, variant),
                    "kind": "object",
                    "scope": scope,
                    "padding": padding,
                }
            )
    return specs


def load_mask_grid(mask_path: Path | None, grid_hw: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = grid_hw
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Invalid token grid size: {grid_hw}")
    if mask_path is None or not mask_path.is_file():
        return np.zeros((grid_h, grid_w), dtype=bool)

    mask = Image.open(mask_path).convert("L")
    try:
        resampling = Image.Resampling.BOX
    except AttributeError:
        resampling = Image.BOX
    pooled = mask.resize((grid_w, grid_h), resampling)
    return np.asarray(pooled) > 0


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dr in range(size):
        for dc in range(size):
            output |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return output


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.ones_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dr in range(size):
        for dc in range(size):
            output &= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return output


def apply_token_padding(mask: np.ndarray, padding: int | None) -> np.ndarray:
    if padding is None or padding == 0:
        return mask.copy()
    if padding > 0:
        return dilate_mask(mask, padding)
    return erode_mask(mask, abs(padding))


def object_mask_for_scope(
    sample: dict[str, Any],
    data_root: Path,
    grid_hw: tuple[int, int],
    scope: str,
    padding: int | None,
) -> np.ndarray:
    masks = []
    if scope in {"target", "target_reference"}:
        target_mask_path = sample.get("target_mask_path")
        target_mask = load_mask_grid(data_root / target_mask_path if target_mask_path else None, grid_hw)
        masks.append(apply_token_padding(target_mask, padding))
    if scope in {"reference", "target_reference"}:
        reference_mask_path = sample.get("reference_mask_path")
        reference_mask = load_mask_grid(data_root / reference_mask_path if reference_mask_path else None, grid_hw)
        masks.append(apply_token_padding(reference_mask, padding))
    if not masks:
        return np.zeros(grid_hw, dtype=bool)
    return np.logical_or.reduce(masks)


def random_mask_for_proportion(
    grid_hw: tuple[int, int],
    proportion: float,
    seed: int,
    sample_id: str,
    condition: str,
) -> np.ndarray:
    grid_h, grid_w = grid_hw
    total = grid_h * grid_w
    k = int(round(max(0.0, min(1.0, proportion)) * total))
    k = max(0, min(total, k))
    mask = np.zeros(total, dtype=bool)
    if k == 0:
        return mask.reshape(grid_hw)

    digest = hashlib.sha256(f"{seed}:{sample_id}:{condition}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(local_seed)
    selected = rng.choice(total, size=k, replace=False)
    mask[selected] = True
    return mask.reshape(grid_hw)


def mask_to_flat(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(-1).astype(bool)


def removed_token_stats(mask: np.ndarray) -> dict[str, Any]:
    total = int(mask.size)
    removed = int(mask.sum())
    return {
        "removed_token_count": removed,
        "total_image_tokens": total,
        "removed_token_proportion": removed / total if total else 0.0,
    }


def completed_record_keys(path: Path) -> set[str]:
    keys = set()
    for row in load_jsonl(path):
        key = row.get("exp2_record_key")
        if key:
            keys.add(key)
    return keys


def make_record_key(sample_id: str, condition: str, prompt_group: str = "all") -> str:
    return f"{sample_id}::{condition}::{prompt_group}"


def localization_score(iou: float | None) -> float:
    if iou is None:
        return 0.0
    return sum(1.0 for threshold in IOU_THRESHOLDS if iou >= threshold) / len(IOU_THRESHOLDS)


def add_exp2_fields(
    record: dict[str, Any],
    condition: dict[str, Any],
    token_stats: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    record["model"] = model_name
    record["exp2_record_key"] = make_record_key(record["sample_id"], condition["condition"])
    record["condition"] = condition["condition"]
    record["condition_kind"] = condition["kind"]
    record["ablation_scope"] = condition.get("scope")
    record["padding"] = condition.get("padding")
    record["random_seed"] = condition.get("random_seed")
    record["random_target_proportion"] = condition.get("random_target_proportion")
    record.update(token_stats)
    record["target_localization_score"] = localization_score(record.get("target_iou"))
    record["reference_localization_score"] = localization_score(record.get("reference_iou"))
    record["spatial_relation_correct"] = bool(record.get("rel_correct"))
    return record


def summarize_records(records: list[dict[str, Any]], output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_condition[row["condition"]].append(row)

    condition_rows = []
    for condition, rows in sorted(by_condition.items()):
        condition_rows.append(summarize_condition(condition, rows))

    random_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        if row["condition_kind"] == "random":
            random_groups[row["random_group"]].append(row)

    random_aggregate_rows = []
    for group, rows in sorted(random_groups.items()):
        random_aggregate_rows.append(summarize_random_group(group, rows))

    summary = {
        "model_name": model_name,
        "num_records": len(records),
        "iou_thresholds_for_localization_score": IOU_THRESHOLDS,
        "conditions": condition_rows,
        "random_condition_aggregates": random_aggregate_rows,
    }
    write_json(output_dir / f"{model_name}_exp2_summary.json", summary)
    write_condition_csv(condition_rows, output_dir / f"{model_name}_exp2_condition_summary.csv")
    if random_aggregate_rows:
        write_random_csv(random_aggregate_rows, output_dir / f"{model_name}_exp2_random_summary.csv")


def summarize_condition(condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind = rows[0].get("condition_kind")
    threshold_metrics: dict[str, Any] = {}
    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        threshold_metrics[f"target_iou_at_{suffix}"] = safe_mean(
            [1.0 if row.get("target_iou") is not None and row["target_iou"] >= threshold else 0.0 for row in rows]
        )
        threshold_metrics[f"reference_iou_at_{suffix}"] = safe_mean(
            [1.0 if row.get("reference_iou") is not None and row["reference_iou"] >= threshold else 0.0 for row in rows]
        )

    return {
        "condition": condition,
        "condition_kind": kind,
        "ablation_scope": rows[0].get("ablation_scope"),
        "padding": rows[0].get("padding"),
        "random_seed": rows[0].get("random_seed"),
        "random_target_proportion": rows[0].get("random_target_proportion"),
        "random_group": random_group_name(rows[0]) if kind == "random" else None,
        "count": len(rows),
        "target_localization_accuracy": safe_mean([row.get("target_localization_score") for row in rows]),
        "reference_localization_accuracy": safe_mean([row.get("reference_localization_score") for row in rows]),
        "spatial_relation_accuracy": safe_mean([1.0 if row.get("rel_correct") else 0.0 for row in rows]),
        "mean_target_iou": safe_mean([row.get("target_iou") for row in rows]),
        "mean_reference_iou": safe_mean([row.get("reference_iou") for row in rows]),
        "avg_removed_token_proportion": safe_mean([row.get("removed_token_proportion") for row in rows]),
        "avg_removed_token_count": safe_mean([row.get("removed_token_count") for row in rows]),
        **threshold_metrics,
    }


def random_group_name(row: dict[str, Any]) -> str:
    p = row.get("random_target_proportion")
    return f"random_p_{p:.8f}" if isinstance(p, (int, float)) else "random_p_unknown"


def summarize_random_group(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "target_localization_accuracy",
        "reference_localization_accuracy",
        "spatial_relation_accuracy",
        "avg_removed_token_proportion",
        "avg_removed_token_count",
    ]
    output: dict[str, Any] = {"random_group": group, "num_seeds": len(rows)}
    for metric in metrics:
        vals = [row.get(metric) for row in rows if row.get(metric) is not None]
        output[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
        output[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
    output["seed_conditions"] = [row["condition"] for row in rows]
    output["random_target_proportion"] = rows[0].get("random_target_proportion")
    return output


def write_condition_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "condition",
        "condition_kind",
        "ablation_scope",
        "padding",
        "random_seed",
        "random_target_proportion",
        "count",
        "target_localization_accuracy",
        "reference_localization_accuracy",
        "spatial_relation_accuracy",
        "mean_target_iou",
        "mean_reference_iou",
        "avg_removed_token_proportion",
        "avg_removed_token_count",
        "target_iou_at_0_5",
        "target_iou_at_0_7",
        "target_iou_at_0_9",
        "reference_iou_at_0_5",
        "reference_iou_at_0_7",
        "reference_iou_at_0_9",
    ]
    write_rows_csv(rows, fields, path)


def write_random_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "random_group",
        "random_target_proportion",
        "num_seeds",
        "target_localization_accuracy_mean",
        "target_localization_accuracy_std",
        "reference_localization_accuracy_mean",
        "reference_localization_accuracy_std",
        "spatial_relation_accuracy_mean",
        "spatial_relation_accuracy_std",
        "avg_removed_token_proportion_mean",
        "avg_removed_token_proportion_std",
        "avg_removed_token_count_mean",
        "avg_removed_token_count_std",
        "seed_conditions",
    ]
    write_rows_csv(rows, fields, path)


def write_rows_csv(rows: list[dict[str, Any]], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            if isinstance(out.get("seed_conditions"), list):
                out["seed_conditions"] = ";".join(out["seed_conditions"])
            writer.writerow(out)


def unique_object_proportions(records: list[dict[str, Any]]) -> list[float]:
    by_condition: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.get("condition_kind") == "object":
            by_condition[row["condition"]].append(float(row.get("removed_token_proportion", 0.0)))
    values = sorted({round(safe_mean(vals) or 0.0, 8) for vals in by_condition.values()})
    return [v for v in values if v > 0.0]


def parse_proportion_condition(condition: str) -> float:
    match = re.search(r"random_p_([0-9]+(?:_[0-9]+)?)", condition)
    if not match:
        raise ValueError(f"Could not parse random proportion from condition: {condition}")
    return float(match.group(1).replace("_", "."))
