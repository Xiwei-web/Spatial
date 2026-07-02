#!/usr/bin/env python3
"""
Recompute Experiment 1 analysis files from an existing JSONL result file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp1_utils import load_jsonl, write_analysis_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 1 JSONL results.")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="exp1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_jsonl.parent
    records = load_jsonl(args.results_jsonl)
    write_analysis_outputs(records, output_dir, model_name=args.model_name)
    print(f"Analyzed {len(records)} records")
    print(f"Summary: {output_dir / f'{args.model_name}_summary.json'}")
    print(f"CSV: {output_dir / f'{args.model_name}_analysis_rows.csv'}")


if __name__ == "__main__":
    main()
