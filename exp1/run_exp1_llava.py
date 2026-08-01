#!/usr/bin/env python3
"""
Run Experiment 1 with LLaVA-1.5-7B.

This script does not download models. It expects a local HuggingFace-format
LLaVA directory, for example:
  models/llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp1_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    append_jsonl,
    build_localization_prompt,
    build_relation_prompt,
    build_result_record,
    load_completed_sample_ids,
    load_exp1_samples,
    load_jsonl,
    write_analysis_outputs,
)


PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_llava"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 runner for LLaVA-1.5-7B.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--coord-mode",
        choices=["auto", "pixel", "norm01", "norm1000"],
        default="auto",
        help=(
            "How to interpret generated bbox coordinates before converting to pixels. "
            "Use norm1000 if the model consistently emits 0-1000 coordinates."
        ),
    )
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--max-new-tokens-relation", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_llava_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=resolve_torch_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def format_llava_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"USER: <image>\n{prompt}\nASSISTANT:"


def move_inputs_to_model_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def generate_llava(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text = format_llava_prompt(processor, prompt)
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(dict(inputs), model)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def select_samples(samples: list[dict[str, Any]], start_index: int, max_samples: int | None) -> list[dict[str, Any]]:
    selected = samples[start_index:]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "llava_exp1_results.jsonl")

    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(samples, args.start_index, args.max_samples)
    completed_ids = set() if args.no_resume else load_completed_sample_ids(output_jsonl)

    model, processor = load_llava_model(args)

    for idx, sample in enumerate(samples, start=1):
        if sample["sample_id"] in completed_ids:
            continue

        target_prompt = build_localization_prompt(sample["target_name"])
        reference_prompt = build_localization_prompt(sample["reference_name"])
        relation_prompt = build_relation_prompt(sample)

        pred_target_text = generate_llava(
            model,
            processor,
            sample["image_path"],
            target_prompt,
            max_new_tokens=args.max_new_tokens_bbox,
        )
        pred_reference_text = generate_llava(
            model,
            processor,
            sample["image_path"],
            reference_prompt,
            max_new_tokens=args.max_new_tokens_bbox,
        )
        pred_relation_text = generate_llava(
            model,
            processor,
            sample["image_path"],
            relation_prompt,
            max_new_tokens=args.max_new_tokens_relation,
        )

        record = build_result_record(
            sample=sample,
            pred_target_text=pred_target_text,
            pred_reference_text=pred_reference_text,
            pred_relation_text=pred_relation_text,
            coord_mode=args.coord_mode,
        )
        record["model"] = "llava-1.5-7b-hf"
        record["prompts"] = {
            "target_localization": target_prompt,
            "reference_localization": reference_prompt,
            "relation": relation_prompt,
        }
        append_jsonl(output_jsonl, record)

        if args.print_every > 0 and idx % args.print_every == 0:
            print(f"[llava] processed {idx}/{len(samples)}")

    all_records = load_jsonl(output_jsonl)
    write_analysis_outputs(all_records, args.output_dir, model_name="llava")
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / 'llava_summary.json'}")


if __name__ == "__main__":
    main()
