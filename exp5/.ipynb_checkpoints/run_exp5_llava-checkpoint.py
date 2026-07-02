#!/usr/bin/env python3
"""Experiment 5 LLM-decoder attention knockout for LLaVA-1.5-7B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp5_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_LAYER_GRANULARITIES,
    AttentionKnockoutController,
    add_exp5_fields,
    append_jsonl,
    args_to_jsonable,
    attention_knockout_condition_specs,
    build_layer_group_specs,
    build_localization_prompt,
    build_relation_prompt,
    build_result_record,
    completed_record_keys,
    condition_token_mask,
    find_decoder_layers,
    load_exp1_samples,
    load_jsonl,
    mask_to_flat,
    parse_csv_list,
    parse_int_csv,
    patched_decoder_attention,
    select_samples,
    summarize_exp5_records,
    write_json,
)


PROJECT_ROOT = Path("/Users/liuxiwei/Downloads/conference/AAAI27/Exp/whatsup")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp5" / "results_llava"
MODEL_NAME = "llava-1.5-7b-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 5 LLaVA LLM-decoder attention knockout.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm01", "norm1000"], default="auto")
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--max-new-tokens-relation", type=int, default=8)
    parser.add_argument("--layer-granularities", default=",".join(DEFAULT_LAYER_GRANULARITIES))
    parser.add_argument("--conditions", default=None, help="Optional comma-separated condition names to run.")
    parser.add_argument("--random-seeds", default="0")
    parser.add_argument(
        "--random-match-scope",
        choices=["target", "reference", "target_reference"],
        default="target_reference",
        help="Object mask whose per-sample token count is matched by random knockout.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
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


def get_model_device(model: Any) -> Any:
    import torch

    for param in model.parameters():
        if getattr(param.device, "type", None) != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_llava_model(args: argparse.Namespace) -> tuple[Any, Any]:
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


def llava_token_grid(model: Any) -> tuple[int, int]:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is None:
        raise AttributeError("LLaVA model config has no vision_config")
    image_size = int(getattr(vision_config, "image_size"))
    patch_size = int(getattr(vision_config, "patch_size"))
    grid = image_size // patch_size
    return grid, grid


def llava_image_token_id(model: Any, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_index", None)
    if token_id is not None:
        return int(token_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
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


def build_llava_inputs(processor: Any, image_path: str, prompt: str, model: Any) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text = format_llava_prompt(processor, prompt)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return move_inputs_to_model_device(dict(inputs), model)


def generate_llava(
    model: Any,
    processor: Any,
    controller: AttentionKnockoutController,
    image_token_id: int,
    mask_flat: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = build_llava_inputs(processor, image_path, prompt, model)
    controller.set_from_input_ids(inputs["input_ids"], image_token_id=image_token_id, mask_flat=mask_flat)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)
    finally:
        controller.clear()
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def run_condition_sample(
    model: Any,
    processor: Any,
    controller: AttentionKnockoutController,
    sample: dict[str, Any],
    condition: dict[str, Any],
    layer_group: dict[str, Any],
    grid_hw: tuple[int, int],
    image_token_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask, token_stats = condition_token_mask(sample, args.data_root, grid_hw, condition)
    mask_flat = mask_to_flat(mask)
    target_prompt = build_localization_prompt(sample["target_name"])
    reference_prompt = build_localization_prompt(sample["reference_name"])
    relation_prompt = build_relation_prompt(sample)

    pred_target_text = generate_llava(
        model,
        processor,
        controller,
        image_token_id,
        mask_flat,
        sample["image_path"],
        target_prompt,
        args.max_new_tokens_bbox,
    )
    pred_reference_text = generate_llava(
        model,
        processor,
        controller,
        image_token_id,
        mask_flat,
        sample["image_path"],
        reference_prompt,
        args.max_new_tokens_bbox,
    )
    pred_relation_text = generate_llava(
        model,
        processor,
        controller,
        image_token_id,
        mask_flat,
        sample["image_path"],
        relation_prompt,
        args.max_new_tokens_relation,
    )

    record = build_result_record(
        sample=sample,
        pred_target_text=pred_target_text,
        pred_reference_text=pred_reference_text,
        pred_relation_text=pred_relation_text,
        coord_mode=args.coord_mode,
    )
    record["prompts"] = {
        "target_localization": target_prompt,
        "reference_localization": reference_prompt,
        "relation": relation_prompt,
    }
    return add_exp5_fields(record, condition, layer_group, token_stats, model_name=MODEL_NAME)


def maybe_filter_conditions(conditions: list[dict[str, Any]], condition_text: str | None) -> list[dict[str, Any]]:
    if not condition_text:
        return conditions
    allowed = set(parse_csv_list(condition_text))
    return [condition for condition in conditions if condition["condition"] in allowed]


def run_records(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    layer_group: dict[str, Any],
    grid_hw: tuple[int, int],
    image_token_id: int,
    output_jsonl: Path,
    completed: set[str],
    args: argparse.Namespace,
) -> None:
    controller = AttentionKnockoutController()
    for cidx, condition in enumerate(conditions, start=1):
        print(f"[llava] {layer_group['layer_group']} condition {cidx}/{len(conditions)}: {condition['condition']}")
        for sidx, sample in enumerate(samples, start=1):
            key = f"{sample['sample_id']}::{condition['condition']}::{layer_group['layer_group']}"
            if key in completed:
                continue
            record = run_condition_sample(
                model=model,
                processor=processor,
                controller=controller,
                sample=sample,
                condition=condition,
                layer_group=layer_group,
                grid_hw=grid_hw,
                image_token_id=image_token_id,
                args=args,
            )
            append_jsonl(output_jsonl, record)
            completed.add(record["exp5_record_key"])
            if args.print_every > 0 and sidx % args.print_every == 0:
                print(f"[llava] {layer_group['layer_group']} {condition['condition']} processed {sidx}/{len(samples)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "llava_exp5_results.jsonl")
    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    all_samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(all_samples, args.start_index, args.max_samples)
    completed = set() if args.no_resume else completed_record_keys(output_jsonl)

    model, processor = load_llava_model(args)
    grid_hw = llava_token_grid(model)
    image_token_id = llava_image_token_id(model, processor)
    num_layers = len(find_decoder_layers(model))
    granularities = parse_csv_list(args.layer_granularities)
    layer_groups = build_layer_group_specs(num_layers, granularities)
    random_seeds = parse_int_csv(args.random_seeds)
    conditions = maybe_filter_conditions(
        attention_knockout_condition_specs(random_seeds, args.random_match_scope),
        args.conditions,
    )
    baseline_conditions = [] if args.skip_baseline else [condition for condition in conditions if condition["kind"] == "baseline"]
    knockout_conditions = [condition for condition in conditions if condition["kind"] != "baseline"]

    metadata = {
        "model": MODEL_NAME,
        "args": args_to_jsonable(args),
        "grid_hw": list(grid_hw),
        "image_token_id": image_token_id,
        "num_decoder_layers": num_layers,
        "num_samples": len(samples),
        "conditions": conditions,
        "layer_groups": layer_groups,
    }
    write_json(args.output_dir / "llava_exp5_run_metadata.json", metadata)
    print(
        f"[llava] samples={len(samples)} grid={grid_hw} image_token_id={image_token_id} "
        f"layers={num_layers} layer_groups={len(layer_groups)} knockout_conditions={len(knockout_conditions)}"
    )

    if baseline_conditions:
        baseline_group = {"layer_granularity": "none", "layer_group": "baseline", "layers": []}
        run_records(
            model,
            processor,
            samples,
            baseline_conditions,
            baseline_group,
            grid_hw,
            image_token_id,
            output_jsonl,
            completed,
            args,
        )

    for gidx, layer_group in enumerate(layer_groups, start=1):
        print(f"[llava] layer group {gidx}/{len(layer_groups)}: {layer_group['layer_group']} layers={layer_group['layers']}")
        controller = AttentionKnockoutController()
        with patched_decoder_attention(model, layer_group["layers"], controller):
            for cidx, condition in enumerate(knockout_conditions, start=1):
                print(f"[llava] {layer_group['layer_group']} condition {cidx}/{len(knockout_conditions)}: {condition['condition']}")
                for sidx, sample in enumerate(samples, start=1):
                    key = f"{sample['sample_id']}::{condition['condition']}::{layer_group['layer_group']}"
                    if key in completed:
                        continue
                    record = run_condition_sample(
                        model=model,
                        processor=processor,
                        controller=controller,
                        sample=sample,
                        condition=condition,
                        layer_group=layer_group,
                        grid_hw=grid_hw,
                        image_token_id=image_token_id,
                        args=args,
                    )
                    append_jsonl(output_jsonl, record)
                    completed.add(record["exp5_record_key"])
                    if args.print_every > 0 and sidx % args.print_every == 0:
                        print(
                            f"[llava] {layer_group['layer_group']} {condition['condition']} "
                            f"processed {sidx}/{len(samples)}"
                        )

    records = load_jsonl(output_jsonl)
    summarize_exp5_records(records, args.output_dir, model_name="llava")
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / 'llava_exp5_summary.json'}")


if __name__ == "__main__":
    main()
