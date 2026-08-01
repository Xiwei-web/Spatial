#!/usr/bin/env python3
"""Experiment 6 CMA for LLaVA-1.5-13B.

This wrapper reuses the LLaVA runner while changing the default model path,
output directory, and model name for the 13B checkpoint.
"""

from __future__ import annotations

import run_exp6_llava as llava


llava.DEFAULT_MODEL_PATH = llava.PROJECT_ROOT / "models" / "llava-1.5-13b"
llava.DEFAULT_OUTPUT_DIR = llava.EXP6_DIR / "results_llava13b"
llava.MODEL_NAME = "llava-1.5-13b-hf"


if __name__ == "__main__":
    llava.main()
