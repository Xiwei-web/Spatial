#!/usr/bin/env python3
"""Summarize average removed-token ratios for six target/reference conditions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp2_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    REMOVED_TOKEN_RATIO_CONDITIONS,
    load_exp1_samples,
    load_jsonl,
    object_mask_for_scope,
    removed_token_stats,
    select_samples,
)


DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "removed_token_ratios.csv"
MODEL_GRIDS = {
    "llava": (24, 24),
    "qwen": (34, 46),
}
DATA_ROOT_CANDIDATES = (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT.parent / "whatsUP",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average removed-token ratios for target/reference object-token ablations. "
            "Use --results-jsonl to summarize an existing Exp2 output file, or omit it to recompute "
            "ratios from masks with a specified token grid."
        )
    )
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing qa.json, bbox_mask_xyxy.json, images/, and masks/.",
    )
    parser.add_argument("--model-grid", choices=sorted(MODEL_GRIDS), default="qwen")
    parser.add_argument("--grid-height", type=int, default=None)
    parser.add_argument("--grid-width", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--allow-missing-masks", action="store_true")
    return parser.parse_args()


def looks_like_data_root(path: Path) -> bool:
    return (path / "qa.json").is_file() and (path / "bbox_mask_xyxy.json").is_file() and (path / "masks").is_dir()


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        path = data_root.expanduser().resolve()
        if not looks_like_data_root(path):
            raise FileNotFoundError(
                "Invalid --data-root. Expected qa.json, bbox_mask_xyxy.json, and masks/ under: "
                f"{path}"
            )
        return path

    for candidate in DATA_ROOT_CANDIDATES:
        path = candidate.expanduser().resolve()
        if looks_like_data_root(path):
            return path
    checked = "\n".join(str(candidate) for candidate in DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the What’sUp data root automatically. Pass --data-root explicitly; "
        "see paths.example.sh.\nChecked:\n"
        f"{checked}"
    )


def validate_mask_files(samples: list[dict[str, Any]], data_root: Path, allow_missing: bool) -> None:
    missing = []
    for sample in samples:
        for key in ("target_mask_path", "reference_mask_path"):
            mask_rel = sample.get(key)
            if not mask_rel or not (data_root / mask_rel).is_file():
                missing.append((sample.get("sample_id"), key, mask_rel))
    if missing and not allow_missing:
        preview = "\n".join(f"{sid} {key}: {mask_rel}" for sid, key, mask_rel in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} mask files under {data_root}. First missing entries:\n{preview}\n"
            "Use --allow-missing-masks only if you intentionally want missing masks counted as zero removed tokens."
        )


def selected_grid(args: argparse.Namespace) -> tuple[int, int]:
    if args.grid_height is not None or args.grid_width is not None:
        if args.grid_height is None or args.grid_width is None:
            raise ValueError("--grid-height and --grid-width must be provided together")
        return args.grid_height, args.grid_width
    return MODEL_GRIDS[args.model_grid]


def summarize_values(label: str, condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    props = [float(row["removed_token_proportion"]) for row in rows]
    counts = [float(row["removed_token_count"]) for row in rows]
    totals = [float(row["total_image_tokens"]) for row in rows]
    return {
        "label": label,
        "condition": condition,
        "count": len(rows),
        "avg_removed_token_proportion": mean(props) if props else None,
        "std_removed_token_proportion": stdev(props) if len(props) > 1 else 0.0 if props else None,
        "min_removed_token_proportion": min(props) if props else None,
        "max_removed_token_proportion": max(props) if props else None,
        "avg_removed_token_count": mean(counts) if counts else None,
        "avg_total_image_tokens": mean(totals) if totals else None,
    }


def summarize_from_results(path: Path) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        condition = row.get("condition")
        if condition:
            by_condition[condition].append(row)

    rows = []
    for spec in REMOVED_TOKEN_RATIO_CONDITIONS:
        condition = spec["condition"]
        rows.append(summarize_values(spec["label"], condition, by_condition.get(condition, [])))
    return rows


def summarize_from_masks(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_root = resolve_data_root(args.data_root)
    grid_hw = selected_grid(args)
    samples = select_samples(load_exp1_samples(data_root), args.start_index, args.max_samples)
    validate_mask_files(samples, data_root, args.allow_missing_masks)
    print(f"Data root: {data_root}")
    print(f"Samples: {len(samples)}")
    print(f"Token grid: {grid_hw[0]}x{grid_hw[1]} ({grid_hw[0] * grid_hw[1]} tokens)")
    rows = []
    for spec in REMOVED_TOKEN_RATIO_CONDITIONS:
        stats_rows = []
        for sample in samples:
            mask = object_mask_for_scope(
                sample=sample,
                data_root=data_root,
                grid_hw=grid_hw,
                scope=spec["scope"],
                padding=spec["padding"],
            )
            stats_rows.append(removed_token_stats(mask))
        rows.append(summarize_values(spec["label"], spec["condition"], stats_rows))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "condition",
        "count",
        "avg_removed_token_proportion",
        "std_removed_token_proportion",
        "min_removed_token_proportion",
        "max_removed_token_proportion",
        "avg_removed_token_count",
        "avg_total_image_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json_file(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.results_jsonl is not None:
        rows = summarize_from_results(args.results_jsonl)
    else:
        rows = summarize_from_masks(args)

    write_csv(rows, args.output_csv)
    if args.output_json is not None:
        write_json_file(rows, args.output_json)

    for row in rows:
        avg = row["avg_removed_token_proportion"]
        avg_text = "NA" if avg is None else f"{avg:.6f}"
        print(f"{row['label']}: {avg_text} (n={row['count']})")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
