#!/usr/bin/env python3
"""Experiment 7 progressive head ablation for LLaVA-1.5-13B."""

from __future__ import annotations

import run_exp7_llava as llava


llava.DEFAULT_MODEL_PATH = llava.PROJECT_ROOT / "models" / "llava-1.5-13b-hf"
llava.DEFAULT_EXP6_RESULTS_DIR = llava.EXP6_DIR / "results_llava13b"
llava.DEFAULT_OUTPUT_DIR = llava.EXP7_DIR / "results_llava13b"
llava.MODEL_NAME = "llava-1.5-13b-hf"


if __name__ == "__main__":
    llava.main()
