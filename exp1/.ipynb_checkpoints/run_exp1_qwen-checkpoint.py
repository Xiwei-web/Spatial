#!/usr/bin/env python3
"""
Run Experiment 1 with Qwen2.5-VL-7B-Instruct.

This script does not download models. It expects a local HuggingFace-format
Qwen2.5-VL directory, for example:
  models/Qwen2.5-VL-7B-Instruct
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


DEFAULT_MODEL_PATH = Path(
    "/Users/liuxiwei/Downloads/conference/AAAI27/Exp/whatsup/models/Qwen2.5-VL-7B-Instruct"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/liuxiwei/Downloads/conference/AAAI27/Exp/whatsup/exp1/results_qwen"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 runner for Qwen2.5-VL-7B-Instruct.")
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
        "--attn-implementation",
        default=None,
        help="Optional transformers attention implementation, e.g. flash_attention_2 or sdpa.",
    )
    parser.add_argument(
        "--coord-mode",
        choices=["auto", "pixel", "norm01", "norm1000"],
        default="auto",
        help=(
            "How to interpret generated bbox coordinates before converting to pixels. "
            "Use norm1000 if Qwen emits 0-1000 coordinates."
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


def load_qwen_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    device = next(model.parameters()).device
    return inputs.to(device)


def build_qwen_messages(image_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_qwen_inputs(processor: Any, image_path: str, prompt: str) -> Any:
    from PIL import Image

    messages = build_qwen_messages(image_path, prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        image = Image.open(image_path).convert("RGB")
        return processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )


def generate_qwen(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = build_qwen_inputs(processor, image_path, prompt)
    inputs = move_inputs_to_model_device(inputs, model)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def select_samples(samples: list[dict[str, Any]], start_index: int, max_samples: int | None) -> list[dict[str, Any]]:
    selected = samples[start_index:]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "qwen_exp1_results.jsonl")

    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(samples, args.start_index, args.max_samples)
    completed_ids = set() if args.no_resume else load_completed_sample_ids(output_jsonl)

    model, processor = load_qwen_model(args)

    for idx, sample in enumerate(samples, start=1):
        if sample["sample_id"] in completed_ids:
            continue

        target_prompt = build_localization_prompt(sample["target_name"])
        reference_prompt = build_localization_prompt(sample["reference_name"])
        relation_prompt = build_relation_prompt(sample)

        pred_target_text = generate_qwen(
            model,
            processor,
            sample["image_path"],
            target_prompt,
            max_new_tokens=args.max_new_tokens_bbox,
        )
        pred_reference_text = generate_qwen(
            model,
            processor,
            sample["image_path"],
            reference_prompt,
            max_new_tokens=args.max_new_tokens_bbox,
        )
        pred_relation_text = generate_qwen(
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
        record["model"] = "Qwen2.5-VL-7B-Instruct"
        record["prompts"] = {
            "target_localization": target_prompt,
            "reference_localization": reference_prompt,
            "relation": relation_prompt,
        }
        append_jsonl(output_jsonl, record)

        if args.print_every > 0 and idx % args.print_every == 0:
            print(f"[qwen] processed {idx}/{len(samples)}")

    all_records = load_jsonl(output_jsonl)
    write_analysis_outputs(all_records, args.output_dir, model_name="qwen")
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / 'qwen_summary.json'}")


if __name__ == "__main__":
    main()
