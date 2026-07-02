#!/usr/bin/env python3
"""Experiment 6 CMA for LLaVA-1.5-7B.

Stages:
  prepare: write binary yes/no QA and source/base candidate pairs.
  filter:  run source/base binary relation behavior and curate up to 50 pairs.
  cma:     run head-level causal mediation analysis on the curated pairs.
  all:     run prepare, filter, and cma.

This script is intended for the HPC environment. The default source/removed
paths are placeholders matching the requested cluster layout and can be
overridden from the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp6_utils import (  # noqa: E402
    DEFAULT_BBOX_MASK_XYXY_PATH,
    DEFAULT_CURATED_SIZE,
    DEFAULT_QA_PATH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REMOVED_IMAGE_ROOT,
    DEFAULT_SOURCE_IMAGE_ROOT,
    EXP6_DIR,
    PROJECT_ROOT,
    append_jsonl,
    build_candidate_pairs,
    cache_head_activations,
    clone_inputs_without_labels,
    cma_record_key,
    completed_behavior_ids,
    completed_cma_keys,
    finite_gap,
    find_decoder_layers,
    find_self_attention,
    infer_num_heads,
    load_json,
    load_jsonl,
    make_task_payload,
    mediation_fraction,
    parse_int_csv,
    parse_task_list,
    parse_yes_no,
    patch_single_head_activation,
    path_to_str_dict,
    score_positions_for_target,
    select_curated_subset,
    summarize_cma_records,
    teacher_forced_perplexity,
    write_binary_qa,
    write_cma_summary_csv,
    write_json,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp6" / "results_llava"
MODEL_NAME = "llava-1.5-7b-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 6 LLaVA causal mediation analysis.")
    parser.add_argument("--stage", choices=["prepare", "filter", "cma", "all"], default="all")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--bbox-mask-path", type=Path, default=DEFAULT_BBOX_MASK_XYXY_PATH)
    parser.add_argument("--source-image-root", type=Path, default=DEFAULT_SOURCE_IMAGE_ROOT)
    parser.add_argument("--removed-image-root", type=Path, default=DEFAULT_REMOVED_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary-qa-path", type=Path, default=None)
    parser.add_argument("--pairs-path", type=Path, default=None)
    parser.add_argument("--behavior-jsonl", type=Path, default=None)
    parser.add_argument("--curated-json", type=Path, default=None)
    parser.add_argument("--cma-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--curated-size", type=int, default=DEFAULT_CURATED_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--min-base-source-ppl-gap", type=float, default=0.0)
    parser.add_argument("--gap-eps", type=float, default=1e-6)
    parser.add_argument("--tasks", default="relation,localization")
    parser.add_argument("--layers", default="all", help="Comma-separated layer indices, or all.")
    parser.add_argument("--heads", default="all", help="Comma-separated head indices, or all.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Debug cap for filter stage.")
    parser.add_argument("--max-curated", type=int, default=None, help="Debug cap for CMA stage.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens-relation", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.binary_qa_path = args.binary_qa_path or (EXP6_DIR / "qa.json")
    args.pairs_path = args.pairs_path or (EXP6_DIR / "candidate_pairs.json")
    args.behavior_jsonl = args.behavior_jsonl or (args.output_dir / f"{MODEL_NAME}_source_base_behavior.jsonl")
    args.curated_json = args.curated_json or (args.output_dir / f"{MODEL_NAME}_curated_subset.json")
    args.cma_jsonl = args.cma_jsonl or (args.output_dir / f"{MODEL_NAME}_cma_prompt_tokens.jsonl")
    args.summary_json = args.summary_json or (args.output_dir / f"{MODEL_NAME}_cma_summary.json")
    args.summary_csv = args.summary_csv or (args.output_dir / f"{MODEL_NAME}_cma_summary.csv")


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


def llava_image_token_id(model: Any, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_index", None)
    if token_id is not None:
        return int(token_id)
    tokenizer = processor.tokenizer
    converted = tokenizer.convert_tokens_to_ids("<image>")
    if converted is not None and converted != tokenizer.unk_token_id:
        return int(converted)
    raise AttributeError("Could not infer LLaVA image token id")


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


def build_inputs_from_text(processor: Any, image_path: str, text: str, model: Any) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=text, return_tensors="pt")
    return move_inputs_to_model_device(dict(inputs), model)


def generate_relation(model: Any, processor: Any, image_path: str, prompt: str, args: argparse.Namespace) -> str:
    import torch

    text = format_llava_prompt(processor, prompt)
    inputs = build_inputs_from_text(processor, image_path, text, model)
    generate_kwargs = {"max_new_tokens": args.max_new_tokens_relation, "do_sample": False}
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def prompt_token_positions(input_ids: Any, image_token_id: int, prompt_len: int) -> list[int]:
    import torch

    ids = input_ids[0].detach().cpu()
    image_positions = torch.nonzero(ids == int(image_token_id), as_tuple=False).flatten()
    start = int(image_positions.max().item()) + 1 if image_positions.numel() else 0
    return list(range(start, prompt_len))


def prepare_teacher_forcing(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    target_text: str,
    task: str,
    image_token_id: int,
) -> dict[str, Any]:
    formatted_prompt = format_llava_prompt(processor, prompt)
    prompt_inputs = build_inputs_from_text(processor, image_path, formatted_prompt, model)
    full_inputs = build_inputs_from_text(processor, image_path, formatted_prompt + target_text, model)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    score_positions = score_positions_for_target(
        tokenizer=processor.tokenizer,
        full_input_ids=full_inputs["input_ids"],
        prompt_len=prompt_len,
        task=task,
    )
    return {
        "inputs": full_inputs,
        "prompt_len": prompt_len,
        "prompt_positions": prompt_token_positions(prompt_inputs["input_ids"], image_token_id, prompt_len),
        "score_positions": score_positions,
    }


def prepare_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    binary_qa = write_binary_qa(args.qa_path, args.binary_qa_path)
    pairs = build_candidate_pairs(
        binary_qa=binary_qa,
        source_image_root=args.source_image_root,
        removed_image_root=args.removed_image_root,
        bbox_mask_path=args.bbox_mask_path,
        require_existing_files=True,
    )
    write_json(args.pairs_path, pairs)
    write_json(
        args.output_dir / f"{MODEL_NAME}_exp6_prepare_meta.json",
        {
            "model": MODEL_NAME,
            "args": path_to_str_dict(args),
            "num_binary_qa": len(binary_qa),
            "num_candidate_pairs": len(pairs),
        },
    )
    print(f"[prepare] binary QA: {args.binary_qa_path} ({len(binary_qa)} rows)")
    print(f"[prepare] candidate pairs: {args.pairs_path} ({len(pairs)} rows)")
    return pairs


def load_or_prepare_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pairs_path.is_file():
        return load_json(args.pairs_path)
    return prepare_stage(args)


def run_filter_stage(model: Any, processor: Any, args: argparse.Namespace, image_token_id: int) -> list[dict[str, Any]]:
    pairs = load_or_prepare_pairs(args)
    if args.max_candidates is not None:
        pairs = pairs[: args.max_candidates]
    completed = set() if args.no_resume or args.overwrite else completed_behavior_ids(args.behavior_jsonl)
    if args.overwrite and args.behavior_jsonl.exists():
        args.behavior_jsonl.unlink()

    for idx, sample in enumerate(pairs, start=1):
        if sample["sample_id"] in completed:
            continue
        prompt = sample["binary_question"]
        target_text = " yes"
        source_tf = prepare_teacher_forcing(
            model, processor, sample["source_image_path"], prompt, target_text, "relation", image_token_id
        )
        base_tf = prepare_teacher_forcing(
            model, processor, sample["base_image_path"], prompt, target_text, "relation", image_token_id
        )
        source_text = generate_relation(model, processor, sample["source_image_path"], prompt, args)
        base_text = generate_relation(model, processor, sample["base_image_path"], prompt, args)
        source_ppl = teacher_forced_perplexity(model, source_tf["inputs"], source_tf["score_positions"])
        base_ppl = teacher_forced_perplexity(model, base_tf["inputs"], base_tf["score_positions"])
        record = {
            "model": MODEL_NAME,
            "sample_id": sample["sample_id"],
            "image_stem": sample["image_stem"],
            "sample": sample,
            "binary_question": prompt,
            "source_output": source_text,
            "base_output": base_text,
            "source_prediction": parse_yes_no(source_text),
            "base_prediction": parse_yes_no(base_text),
            "source_yes_ppl": source_ppl,
            "base_yes_ppl": base_ppl,
            "base_minus_source_yes_ppl": base_ppl - source_ppl,
        }
        append_jsonl(args.behavior_jsonl, record)
        if args.print_every > 0 and idx % args.print_every == 0:
            print(f"[filter] evaluated {idx}/{len(pairs)} pairs")

    behavior = load_jsonl(args.behavior_jsonl)
    curated = select_curated_subset(
        behavior,
        max_samples=args.curated_size,
        seed=args.seed,
        min_ppl_gap=args.min_base_source_ppl_gap,
    )
    if args.max_curated is not None:
        curated = curated[: args.max_curated]
    write_json(args.curated_json, curated)
    write_json(
        args.output_dir / f"{MODEL_NAME}_filter_meta.json",
        {
            "model": MODEL_NAME,
            "num_behavior_records": len(behavior),
            "num_curated": len(curated),
            "curated_size_requested": args.curated_size,
            "seed": args.seed,
            "min_base_source_ppl_gap": args.min_base_source_ppl_gap,
        },
    )
    print(f"[filter] behavior records: {args.behavior_jsonl} ({len(behavior)} rows)")
    print(f"[filter] curated subset: {args.curated_json} ({len(curated)} rows)")
    return curated


def load_or_filter_curated(model: Any, processor: Any, args: argparse.Namespace, image_token_id: int) -> list[dict[str, Any]]:
    if args.curated_json.is_file() and args.stage == "cma":
        curated = load_json(args.curated_json)
        return curated[: args.max_curated] if args.max_curated is not None else curated
    return run_filter_stage(model, processor, args, image_token_id)


def run_cma_for_sample_task(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    task: str,
    image_token_id: int,
    layers_to_run: list[int],
    heads_to_run: list[int],
    args: argparse.Namespace,
    completed: set[str],
) -> None:
    prompt, target_text = make_task_payload(sample, task)
    source_tf = prepare_teacher_forcing(
        model, processor, sample["source_image_path"], prompt, target_text, task, image_token_id
    )
    base_tf = prepare_teacher_forcing(
        model, processor, sample["base_image_path"], prompt, target_text, task, image_token_id
    )
    prompt_positions = base_tf["prompt_positions"]

    with cache_head_activations(model, source_tf["prompt_positions"]) as source_store:
        source_ppl = teacher_forced_perplexity(model, source_tf["inputs"], source_tf["score_positions"])
    base_ppl = teacher_forced_perplexity(model, base_tf["inputs"], base_tf["score_positions"])
    valid_gap = finite_gap(base_ppl=base_ppl, source_ppl=source_ppl, eps=args.gap_eps)

    for layer_idx in layers_to_run:
        for head_idx in heads_to_run:
            key = cma_record_key(sample["sample_id"], task, layer_idx, head_idx)
            if key in completed:
                continue
            patched_ppl = float("nan")
            mf = None
            if valid_gap:
                with patch_single_head_activation(
                    model=model,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    source_cache=source_store.cache,
                    prompt_positions=prompt_positions,
                ):
                    patched_ppl = teacher_forced_perplexity(model, base_tf["inputs"], base_tf["score_positions"])
                mf = mediation_fraction(source_ppl, base_ppl, patched_ppl, eps=args.gap_eps)
            record = {
                "cma_key": key,
                "model": MODEL_NAME,
                "sample_id": sample["sample_id"],
                "image_stem": sample["image_stem"],
                "task": task,
                "patch_positions": "prompt_tokens",
                "layer": layer_idx,
                "head": head_idx,
                "source_ppl": source_ppl,
                "base_ppl": base_ppl,
                "patched_ppl": patched_ppl,
                "valid_gap": valid_gap,
                "mediation_fraction": mf,
                "prompt": prompt,
                "target_text": target_text,
            }
            append_jsonl(args.cma_jsonl, record)
            completed.add(key)


def run_cma_stage(model: Any, processor: Any, args: argparse.Namespace, image_token_id: int) -> None:
    curated = load_or_filter_curated(model, processor, args, image_token_id)
    tasks = parse_task_list(args.tasks)
    layers = find_decoder_layers(model)
    num_layers = len(layers)
    first_attn = find_self_attention(layers[0])
    num_heads = infer_num_heads(model, first_attn)
    layers_to_run = parse_int_csv(args.layers, upper=num_layers) or list(range(num_layers))
    heads_to_run = parse_int_csv(args.heads, upper=num_heads) or list(range(num_heads))

    if args.overwrite and args.cma_jsonl.exists():
        args.cma_jsonl.unlink()
    completed = set() if args.no_resume or args.overwrite else completed_cma_keys(args.cma_jsonl)

    total_units = len(curated) * len(tasks)
    unit_idx = 0
    for sample in curated:
        for task in tasks:
            unit_idx += 1
            if task == "localization" and "target_bbox" not in sample:
                print(f"[cma] skip localization for {sample['sample_id']}: no target_bbox")
                continue
            print(f"[cma] {unit_idx}/{total_units} sample={sample['sample_id']} task={task}")
            run_cma_for_sample_task(
                model=model,
                processor=processor,
                sample=sample,
                task=task,
                image_token_id=image_token_id,
                layers_to_run=layers_to_run,
                heads_to_run=heads_to_run,
                args=args,
                completed=completed,
            )

    records = load_jsonl(args.cma_jsonl)
    summary = summarize_cma_records(records, num_layers=num_layers, num_heads=num_heads)
    summary["model"] = MODEL_NAME
    summary["args"] = path_to_str_dict(args)
    summary["num_records"] = len(records)
    summary["num_curated_samples"] = len(curated)
    write_json(args.summary_json, summary)
    write_cma_summary_csv(summary, args.summary_csv)
    print(f"[cma] records: {args.cma_jsonl} ({len(records)} rows)")
    print(f"[cma] summary: {args.summary_json}")


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)

    if args.stage == "prepare":
        prepare_stage(args)
        return

    model, processor = load_model(args)
    image_token_id = llava_image_token_id(model, processor)

    if args.stage == "filter":
        run_filter_stage(model, processor, args, image_token_id)
    elif args.stage == "cma":
        run_cma_stage(model, processor, args, image_token_id)
    elif args.stage == "all":
        prepare_stage(args)
        run_cma_stage(model, processor, args, image_token_id)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
