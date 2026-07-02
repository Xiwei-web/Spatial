#!/usr/bin/env python3
"""Filter Qwen Exp5 results around the first right-padding warning.

Old batched Qwen runs did not store whether a batch triggered the
decoder-only right-padding warning. The conservative recovery strategy is to
keep JSONL records before the first known warning point and rerun everything
from that point with the fixed left-padding script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/Users/liuxiwei/Downloads/conference/AAAI27/Exp/whatsup")
DEFAULT_INPUT = PROJECT_ROOT / "exp5" / "results_qwen" / "qwen_exp5_results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep Exp5 Qwen JSONL rows before the first unsafe padding point.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--safe-jsonl", type=Path, default=None)
    parser.add_argument("--risky-jsonl", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--cutoff-record-key", default=None, help="Exact exp5_record_key where unsafe rows begin.")
    parser.add_argument("--cutoff-layer-group", default=None, help="Example: group4_16_19")
    parser.add_argument("--cutoff-condition", default=None, help="Example: target_object")
    parser.add_argument("--include-cutoff-in-safe", action="store_true", help="Keep the first matching row as safe.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def is_cutoff_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.cutoff_record_key:
        return row.get("exp5_record_key") == args.cutoff_record_key
    if args.cutoff_layer_group and args.cutoff_condition:
        return row.get("layer_group") == args.cutoff_layer_group and row.get("condition") == args.cutoff_condition
    raise ValueError("Provide either --cutoff-record-key or both --cutoff-layer-group and --cutoff-condition")


def default_path(input_jsonl: Path, suffix: str) -> Path:
    return input_jsonl.with_name(f"{input_jsonl.stem}.{suffix}{input_jsonl.suffix}")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    cutoff_index = None
    for idx, row in enumerate(rows):
        if is_cutoff_row(row, args):
            cutoff_index = idx
            break
    if cutoff_index is None:
        raise RuntimeError("Could not find cutoff row in input JSONL")

    split_index = cutoff_index + 1 if args.include_cutoff_in_safe else cutoff_index
    safe_rows = rows[:split_index]
    risky_rows = rows[split_index:]

    safe_jsonl = args.safe_jsonl or default_path(args.input_jsonl, "safe_before_padding_warning")
    risky_jsonl = args.risky_jsonl or default_path(args.input_jsonl, "risky_from_padding_warning")
    report_json = args.report_json or default_path(args.input_jsonl, "padding_filter_report").with_suffix(".json")

    write_jsonl(safe_jsonl, safe_rows)
    write_jsonl(risky_jsonl, risky_rows)

    cutoff_row = rows[cutoff_index]
    report = {
        "input_jsonl": str(args.input_jsonl),
        "safe_jsonl": str(safe_jsonl),
        "risky_jsonl": str(risky_jsonl),
        "total_rows": len(rows),
        "safe_rows": len(safe_rows),
        "risky_rows": len(risky_rows),
        "cutoff_index_0_based": cutoff_index,
        "cutoff_included_in_safe": bool(args.include_cutoff_in_safe),
        "cutoff_record_key": cutoff_row.get("exp5_record_key"),
        "cutoff_layer_group": cutoff_row.get("layer_group"),
        "cutoff_condition": cutoff_row.get("condition"),
    }
    write_json(report_json, report)

    print(f"safe_rows={len(safe_rows)} -> {safe_jsonl}")
    print(f"risky_rows={len(risky_rows)} -> {risky_jsonl}")
    print(f"report -> {report_json}")


if __name__ == "__main__":
    main()
