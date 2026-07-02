#!/usr/bin/env python3
"""Experiment 7 progressive head ablation for Qwen2.5-VL-7B-Instruct."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp7_utils import (  # noqa: E402
    DEFAULT_FRACTIONS,
    EXP6_DIR,
    EXP7_DIR,
    PROJECT_ROOT,
    add_auc_to_summary,
    append_jsonl,
    build_localization_prompt,
    build_progressive_specs,
    completed_exp7_keys,
    evaluate_localization_output,
    exp7_record_key,
    find_decoder_layers,
    find_self_attention,
    infer_num_heads,
    load_cma_scores,
    load_json,
    load_jsonl,
    parse_float_csv,
    parse_yes_no,
    plot_exp7_curves,
    rank_heads,
    relation_correct,
    summarize_exp7_records,
    write_json,
    write_summary_csv,
    zero_attention_heads,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_EXP6_RESULTS_DIR = EXP6_DIR / "results_qwen"
DEFAULT_OUTPUT_DIR = EXP7_DIR / "results_qwen"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 7 Qwen progressive head ablation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--exp6-curated-json", type=Path, default=DEFAULT_EXP6_RESULTS_DIR / f"{MODEL_NAME}_curated_subset.json")
    parser.add_argument("--exp6-cma-summary-csv", type=Path, default=DEFAULT_EXP6_RESULTS_DIR / f"{MODEL_NAME}_cma_summary.csv")
    parser.add_argument("--exp6-cma-jsonl", type=Path, default=DEFAULT_EXP6_RESULTS_DIR / f"{MODEL_NAME}_cma_prompt_tokens.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--ranking-json", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--fractions", default=",".join(str(v) for v in DEFAULT_FRACTIONS))
    parser.add_argument("--tasks", default="relation,localization")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--resized-height", type=int, default=448)
    parser.add_argument("--resized-width", type=int, default=448)
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm01", "norm1000"], default="auto")
    parser.add_argument("--max-new-tokens-relation", type=int, default=4)
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl = args.output_jsonl or (args.output_dir / f"{MODEL_NAME}_exp7_head_ablation.jsonl")
    args.summary_json = args.summary_json or (args.output_dir / f"{MODEL_NAME}_exp7_summary.json")
    args.summary_csv = args.summary_csv or (args.output_dir / f"{MODEL_NAME}_exp7_summary.csv")
    args.ranking_json = args.ranking_json or (args.output_dir / f"{MODEL_NAME}_exp7_head_rankings.json")
    args.plot_path = args.plot_path or (args.output_dir / f"{MODEL_NAME}_exp7_accuracy_curves.png")


def parse_task_list(text: str) -> list[str]:
    tasks = [item.strip() for item in text.split(",") if item.strip()]
    for task in tasks:
        if task not in {"relation", "localization"}:
            raise ValueError("Tasks must be comma-separated values from relation,localization")
    return tasks


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


def get_model_device(model: Any) -> Any:
    import torch

    for param in model.parameters():
        if getattr(param.device, "type", None) != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    loader_errors = []
    if model_cls is not None:
        try:
            model = model_cls.from_pretrained(args.model_path, **model_kwargs)
        except Exception as exc:
            loader_errors.append(f"Qwen2_5_VLForConditionalGeneration: {exc}")
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            torch.set_grad_enabled(False)
            return model, processor

    for auto_cls_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        auto_cls = getattr(transformers, auto_cls_name, None)
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_pretrained(args.model_path, **model_kwargs)
        except Exception as exc:
            loader_errors.append(f"{auto_cls_name}: {exc}")
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            torch.set_grad_enabled(False)
            return model, processor

    version = getattr(transformers, "__version__", "unknown")
    details = "\n".join(loader_errors[-3:])
    raise ImportError(
        "Could not load Qwen2.5-VL with the installed transformers package "
        f"(version={version}). Install a release that includes "
        "Qwen2_5_VLForConditionalGeneration. Loader attempts:\n"
        f"{details}"
    )


def build_qwen_messages(image_path: str, prompt: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    image_item: dict[str, Any] = {"type": "image", "image": image_path}
    if args.resized_height is not None:
        image_item["resized_height"] = args.resized_height
    if args.resized_width is not None:
        image_item["resized_width"] = args.resized_width
    return [{"role": "user", "content": [image_item, {"type": "text", "text": prompt}]}]


def move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    import torch

    device = get_model_device(model)
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def build_qwen_inputs(processor: Any, image_path: str, prompt: str, model: Any, args: argparse.Namespace) -> Any:
    messages = build_qwen_messages(image_path, prompt, args)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    except Exception:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if args.resized_height is not None and args.resized_width is not None:
            image = image.resize((args.resized_width, args.resized_height))
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return move_inputs_to_model_device(inputs, model)


def generate_text(model: Any, processor: Any, image_path: str, prompt: str, max_new_tokens: int, args: argparse.Namespace) -> str:
    import torch

    inputs = build_qwen_inputs(processor, image_path, prompt, model, args)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def run_one(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    spec: dict[str, Any],
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = sample["source_image_path"]
    with zero_attention_heads(model, spec["ablated_heads"]):
        if task == "relation":
            prompt = sample["binary_question"]
            output = generate_text(model, processor, image_path, prompt, args.max_new_tokens_relation, args)
            metrics = {
                "relation_prediction": parse_yes_no(output),
                "relation_correct": relation_correct(output),
            }
        elif task == "localization":
            prompt = build_localization_prompt(sample["target_name"])
            output = generate_text(model, processor, image_path, prompt, args.max_new_tokens_bbox, args)
            metrics = evaluate_localization_output(output, sample, coord_mode=args.coord_mode)
        else:
            raise ValueError(f"Unsupported task: {task}")

    record = {
        "model": MODEL_NAME,
        "sample_id": sample["sample_id"],
        "image_stem": sample["image_stem"],
        "task": task,
        "head_group": spec["head_group"],
        "head_group_label": spec["head_group_label"],
        "fraction_pruned": spec["fraction_pruned"],
        "num_ablated_heads": spec["num_ablated_heads"],
        "prompt": prompt,
        "output": output,
        "exp7_key": exp7_record_key(sample["sample_id"], task, spec["head_group"], spec["fraction_pruned"]),
    }
    record.update(metrics)
    return record


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    tasks = parse_task_list(args.tasks)
    fractions = parse_float_csv(args.fractions)

    samples = load_json(args.exp6_curated_json)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    model, processor = load_model(args)
    layers = find_decoder_layers(model)
    num_layers = len(layers)
    num_heads = infer_num_heads(model, find_self_attention(layers[0]))

    cma_scores = load_cma_scores(args.exp6_cma_summary_csv, args.exp6_cma_jsonl)
    rankings = rank_heads(cma_scores, num_layers=num_layers, num_heads=num_heads)
    specs = build_progressive_specs(rankings, fractions, num_layers=num_layers, num_heads=num_heads)
    write_json(
        args.ranking_json,
        {
            "model": MODEL_NAME,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "fractions": fractions,
            "rankings": {group: [[layer, head] for layer, head in heads] for group, heads in rankings.items()},
            "specs": specs,
        },
    )

    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed = set() if args.no_resume or args.overwrite else completed_exp7_keys(args.output_jsonl)

    total = len(samples) * len(specs) * len(tasks)
    idx = 0
    for spec in specs:
        for sample in samples:
            for task in tasks:
                idx += 1
                if task == "localization" and "target_bbox" not in sample:
                    continue
                key = exp7_record_key(sample["sample_id"], task, spec["head_group"], spec["fraction_pruned"])
                if key in completed:
                    continue
                record = run_one(model, processor, sample, spec, task, args)
                append_jsonl(args.output_jsonl, record)
                completed.add(key)
                if args.print_every > 0 and idx % args.print_every == 0:
                    print(f"[exp7] {idx}/{total} group={spec['head_group']} frac={spec['fraction_pruned']} task={task}")

    records = load_jsonl(args.output_jsonl)
    summary_rows = add_auc_to_summary(summarize_exp7_records(records))
    write_json(
        args.summary_json,
        {
            "model": MODEL_NAME,
            "num_records": len(records),
            "num_samples": len(samples),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "summary": summary_rows,
        },
    )
    write_summary_csv(summary_rows, args.summary_csv)
    plot_exp7_curves(summary_rows, MODEL_NAME, args.plot_path)
    print(f"[exp7] records: {args.output_jsonl}")
    print(f"[exp7] summary: {args.summary_json}")
    print(f"[exp7] plot: {args.plot_path}")


if __name__ == "__main__":
    main()
