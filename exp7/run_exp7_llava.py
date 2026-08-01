#!/usr/bin/env python3
"""Experiment 7 progressive head ablation for LLaVA-1.5-7B."""

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
    parse_group_list,
    parse_task_list,
    parse_yes_no,
    plot_exp7_curves,
    rank_heads,
    relation_correct,
    summarize_exp7_records,
    write_json,
    write_summary_csv,
    zero_attention_heads,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "llava-1.5-7b-hf"
DEFAULT_EXP6_RESULTS_DIR = EXP6_DIR / "results_llava"
DEFAULT_OUTPUT_DIR = EXP7_DIR / "results_llava"
MODEL_NAME = "llava-1.5-7b-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 7 LLaVA progressive head ablation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--exp6-results-dir", type=Path, default=DEFAULT_EXP6_RESULTS_DIR)
    parser.add_argument("--exp6-cma-summary-csv", type=Path, default=None)
    parser.add_argument("--relation-curated-json", type=Path, default=None)
    parser.add_argument("--localization-curated-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--ranking-json", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--metric", choices=["mean", "median"], default="mean")
    parser.add_argument("--min-valid-count", type=int, default=1)
    parser.add_argument("--fractions", default=",".join(str(v) for v in DEFAULT_FRACTIONS))
    parser.add_argument("--head-groups", default="localization_critical,relation_critical,low_importance,negative")
    parser.add_argument("--ablate-scope", choices=["prompt_prefill", "all_tokens"], default="prompt_prefill")
    parser.add_argument("--tasks", default="relation,localization")
    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "resized", "norm01", "norm1000"], default="pixel")
    parser.add_argument("--resized-height", type=int, default=None)
    parser.add_argument("--resized-width", type=int, default=None)
    parser.add_argument("--max-new-tokens-relation", type=int, default=4)
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.exp6_cma_summary_csv = args.exp6_cma_summary_csv or (
        args.exp6_results_dir / f"{MODEL_NAME}_cma_summary.csv"
    )
    args.relation_curated_json = args.relation_curated_json or (
        args.exp6_results_dir / f"{MODEL_NAME}_curated_relation_subset.json"
    )
    args.localization_curated_json = args.localization_curated_json or (
        args.exp6_results_dir / f"{MODEL_NAME}_curated_localization_subset.json"
    )
    args.output_jsonl = args.output_jsonl or (args.output_dir / f"{MODEL_NAME}_exp7_head_ablation.jsonl")
    args.summary_json = args.summary_json or (args.output_dir / f"{MODEL_NAME}_exp7_summary.json")
    args.summary_csv = args.summary_csv or (args.output_dir / f"{MODEL_NAME}_exp7_summary.csv")
    args.ranking_json = args.ranking_json or (args.output_dir / f"{MODEL_NAME}_exp7_head_rankings.json")
    args.plot_path = args.plot_path or (args.output_dir / f"{MODEL_NAME}_exp7_accuracy_curves.png")


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
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
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
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"USER: <image>\n{prompt}\nASSISTANT:"


def move_inputs_to_model_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = get_model_device(model)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def build_llava_inputs(processor: Any, image_path: str, prompt: str, model: Any) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text = format_llava_prompt(processor, prompt)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return move_inputs_to_model_device(dict(inputs), model)


def generate_text(model: Any, processor: Any, image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = build_llava_inputs(processor, image_path, prompt, model)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def load_samples_by_task(args: argparse.Namespace, tasks: list[str]) -> dict[str, list[dict[str, Any]]]:
    samples = {}
    if "relation" in tasks:
        samples["relation"] = load_json(args.relation_curated_json)
    if "localization" in tasks:
        samples["localization"] = load_json(args.localization_curated_json)
    if args.max_samples_per_task is not None:
        samples = {task: rows[: args.max_samples_per_task] for task, rows in samples.items()}
    return samples


def run_one(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    spec: dict[str, Any],
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = sample["source_image_path"]
    with zero_attention_heads(model, spec["ablated_heads"], token_scope=args.ablate_scope):
        if task == "relation":
            prompt = sample["binary_question"]
            output = generate_text(model, processor, image_path, prompt, args.max_new_tokens_relation)
            expected = sample.get("binary_answer", "yes")
            metrics = {
                "relation_prediction": parse_yes_no(output),
                "relation_expected": expected,
                "relation_correct": relation_correct(output, expected=expected),
            }
        elif task == "localization":
            prompt = build_localization_prompt(sample["target_name"])
            output = generate_text(model, processor, image_path, prompt, args.max_new_tokens_bbox)
            metrics = evaluate_localization_output(
                output,
                sample,
                coord_mode=args.coord_mode,
                resized_width=args.resized_width,
                resized_height=args.resized_height,
            )
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
    head_groups = parse_group_list(args.head_groups)
    fractions = parse_float_csv(args.fractions)
    samples_by_task = load_samples_by_task(args, tasks)

    model, processor = load_model(args)
    layers = find_decoder_layers(model)
    num_layers = len(layers)
    num_heads = infer_num_heads(model, find_self_attention(layers[0]))

    cma_scores = load_cma_scores(args.exp6_cma_summary_csv, metric=args.metric, min_valid_count=args.min_valid_count)
    rankings = rank_heads(cma_scores, num_layers=num_layers, num_heads=num_heads)
    specs = build_progressive_specs(
        rankings,
        fractions,
        num_layers=num_layers,
        num_heads=num_heads,
        head_groups=head_groups,
    )
    write_json(
        args.ranking_json,
        {
            "model": MODEL_NAME,
            "metric": args.metric,
            "ablate_scope": args.ablate_scope,
            "min_valid_count": args.min_valid_count,
            "exp6_cma_summary_csv": str(args.exp6_cma_summary_csv),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "fractions": fractions,
            "head_groups": head_groups,
            "rankings": {group: [[layer, head] for layer, head in heads] for group, heads in rankings.items()},
            "specs": specs,
        },
    )

    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed = set() if args.no_resume or args.overwrite else completed_exp7_keys(args.output_jsonl)

    total = len(specs) * sum(len(samples_by_task[task]) for task in tasks)
    idx = 0
    for spec in specs:
        for task in tasks:
            for sample in samples_by_task[task]:
                idx += 1
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
            "metric": args.metric,
            "num_records": len(records),
            "num_samples_by_task": {task: len(rows) for task, rows in samples_by_task.items()},
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
