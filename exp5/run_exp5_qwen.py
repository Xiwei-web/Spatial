#!/usr/bin/env python3
"""Experiment 5 LLM-decoder attention knockout for Qwen2.5-VL-7B-Instruct."""

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


PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp5" / "results_qwen"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 5 Qwen LLM-decoder attention knockout.")
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
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--resized-height", type=int, default=448)
    parser.add_argument("--resized-width", type=int, default=448)
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


def load_qwen_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
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
        f"(version={version}). Install a newer transformers release that includes "
        "Qwen2_5_VLForConditionalGeneration. Loader attempts:\n"
        f"{details}"
    )


def get_visual_module(model: Any) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen visual module")


def get_spatial_merge_size(model: Any) -> int:
    visual = get_visual_module(model)
    if hasattr(visual, "spatial_merge_size"):
        return int(visual.spatial_merge_size)
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "spatial_merge_size"):
        return int(vision_config.spatial_merge_size)
    return 2


def qwen_image_token_id(model: Any, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_id", None)
    if token_id is not None:
        return int(token_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for token in ("<|image_pad|>", "<image>"):
            converted = tokenizer.convert_tokens_to_ids(token)
            if converted is not None and converted != tokenizer.unk_token_id:
                return int(converted)
    raise AttributeError("Could not infer Qwen image token id")


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


def force_left_padding(inputs: Any, processor: Any) -> Any:
    """Normalize text tensors to left padding after Qwen's multimodal processor.

    Some processor versions ignore tokenizer.padding_side during multimodal
    batching. Generation checks the last input column for pad tokens, so this
    post-processes input_ids/attention_mask directly.
    """

    import torch

    if "input_ids" not in inputs or "attention_mask" not in inputs:
        return inputs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.ndim != 2 or input_ids.shape[0] <= 1:
        return inputs

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        return inputs

    batch_size, seq_len = input_ids.shape
    new_input_ids = torch.full_like(input_ids, int(pad_token_id))
    new_attention_mask = torch.zeros_like(attention_mask)
    for batch_idx in range(batch_size):
        valid = attention_mask[batch_idx].to(dtype=torch.bool)
        tokens = input_ids[batch_idx, valid]
        length = int(tokens.numel())
        if length == 0:
            continue
        new_input_ids[batch_idx, seq_len - length :] = tokens
        new_attention_mask[batch_idx, seq_len - length :] = 1

    inputs["input_ids"] = new_input_ids
    inputs["attention_mask"] = new_attention_mask
    return inputs


def build_qwen_batch_inputs(
    processor: Any,
    image_paths: list[str],
    prompts: list[str],
    model: Any,
    args: argparse.Namespace,
) -> Any:
    messages = [build_qwen_messages(image_path, prompt, args) for image_path, prompt in zip(image_paths, prompts)]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    except Exception:
        from PIL import Image

        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            if args.resized_height is not None and args.resized_width is not None:
                image = image.resize((args.resized_width, args.resized_height))
            images.append(image)
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = force_left_padding(inputs, processor)
    return move_inputs_to_model_device(inputs, model)


def build_qwen_inputs(processor: Any, image_path: str, prompt: str, model: Any, args: argparse.Namespace) -> Any:
    return build_qwen_batch_inputs(processor, [image_path], [prompt], model, args)


def qwen_token_grid_from_inputs(model: Any, inputs: Any) -> tuple[int, int]:
    grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    if len(grid) != 3:
        raise ValueError(f"Expected image_grid_thw row to have 3 values, got {grid}")
    temporal, raw_h, raw_w = [int(v) for v in grid]
    if temporal != 1:
        raise ValueError(f"This experiment expects single-image inputs, got temporal={temporal}")
    merge = get_spatial_merge_size(model)
    return raw_h // merge, raw_w // merge


def sample_qwen_grid(model: Any, processor: Any, sample: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    inputs = build_qwen_inputs(processor, sample["image_path"], "Describe the image.", model, args)
    return qwen_token_grid_from_inputs(model, inputs)


def generate_qwen(
    model: Any,
    processor: Any,
    controller: AttentionKnockoutController,
    image_token_id: int,
    mask_flat: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    args: argparse.Namespace,
) -> str:
    import torch

    inputs = build_qwen_inputs(processor, image_path, prompt, model, args)
    controller.set_from_input_ids(inputs["input_ids"], image_token_id=image_token_id, mask_flat=mask_flat)
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        controller.clear()
    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated_ids)]
    decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def generate_qwen_batch(
    model: Any,
    processor: Any,
    controller: AttentionKnockoutController,
    image_token_id: int,
    mask_flats: list[Any],
    image_paths: list[str],
    prompts: list[str],
    max_new_tokens: int,
    args: argparse.Namespace,
) -> list[str]:
    import torch

    inputs = build_qwen_batch_inputs(processor, image_paths, prompts, model, args)
    controller.set_from_input_ids(inputs["input_ids"], image_token_id=image_token_id, mask_flat=mask_flats)
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        controller.clear()
    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated_ids)]
    return [
        text.strip()
        for text in processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


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

    pred_target_text = generate_qwen(
        model,
        processor,
        controller,
        image_token_id,
        mask_flat,
        sample["image_path"],
        target_prompt,
        args.max_new_tokens_bbox,
        args,
    )
    pred_reference_text = generate_qwen(
        model,
        processor,
        controller,
        image_token_id,
        mask_flat,
        sample["image_path"],
        reference_prompt,
        args.max_new_tokens_bbox,
        args,
    )
    pred_relation_text = generate_qwen(
        model,
        processor,
        controller,
        image_token_id,
        mask_flat,
        sample["image_path"],
        relation_prompt,
        args.max_new_tokens_relation,
        args,
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


def run_condition_batch(
    model: Any,
    processor: Any,
    controller: AttentionKnockoutController,
    samples: list[dict[str, Any]],
    condition: dict[str, Any],
    layer_group: dict[str, Any],
    grid_hw: tuple[int, int],
    image_token_id: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    masks_and_stats = [condition_token_mask(sample, args.data_root, grid_hw, condition) for sample in samples]
    mask_flats = [mask_to_flat(mask) for mask, _ in masks_and_stats]
    image_paths = [sample["image_path"] for sample in samples]
    target_prompts = [build_localization_prompt(sample["target_name"]) for sample in samples]
    reference_prompts = [build_localization_prompt(sample["reference_name"]) for sample in samples]
    relation_prompts = [build_relation_prompt(sample) for sample in samples]

    pred_target_texts = generate_qwen_batch(
        model,
        processor,
        controller,
        image_token_id,
        mask_flats,
        image_paths,
        target_prompts,
        args.max_new_tokens_bbox,
        args,
    )
    pred_reference_texts = generate_qwen_batch(
        model,
        processor,
        controller,
        image_token_id,
        mask_flats,
        image_paths,
        reference_prompts,
        args.max_new_tokens_bbox,
        args,
    )
    pred_relation_texts = generate_qwen_batch(
        model,
        processor,
        controller,
        image_token_id,
        mask_flats,
        image_paths,
        relation_prompts,
        args.max_new_tokens_relation,
        args,
    )

    records = []
    for idx, sample in enumerate(samples):
        record = build_result_record(
            sample=sample,
            pred_target_text=pred_target_texts[idx],
            pred_reference_text=pred_reference_texts[idx],
            pred_relation_text=pred_relation_texts[idx],
            coord_mode=args.coord_mode,
        )
        record["prompts"] = {
            "target_localization": target_prompts[idx],
            "reference_localization": reference_prompts[idx],
            "relation": relation_prompts[idx],
        }
        records.append(
            add_exp5_fields(
                record,
                condition,
                layer_group,
                masks_and_stats[idx][1],
                model_name=MODEL_NAME,
            )
        )
    return records


def maybe_filter_conditions(conditions: list[dict[str, Any]], condition_text: str | None) -> list[dict[str, Any]]:
    if not condition_text:
        return conditions
    allowed = set(parse_csv_list(condition_text))
    return [condition for condition in conditions if condition["condition"] in allowed]


def run_baseline_records(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    baseline_conditions: list[dict[str, Any]],
    grid_hw: tuple[int, int],
    image_token_id: int,
    output_jsonl: Path,
    completed: set[str],
    args: argparse.Namespace,
) -> None:
    if not baseline_conditions:
        return
    controller = AttentionKnockoutController()
    baseline_group = {"layer_granularity": "none", "layer_group": "baseline", "layers": []}
    for condition in baseline_conditions:
        print(f"[qwen] baseline condition: {condition['condition']}")
        pending = [
            sample
            for sample in samples
            if f"{sample['sample_id']}::{condition['condition']}::{baseline_group['layer_group']}" not in completed
        ]
        for start in range(0, len(pending), args.eval_batch_size):
            batch_samples = pending[start : start + args.eval_batch_size]
            if not batch_samples:
                continue
            records = run_condition_batch(
                model,
                processor,
                controller,
                batch_samples,
                condition,
                baseline_group,
                grid_hw,
                image_token_id,
                args,
            )
            for record in records:
                append_jsonl(output_jsonl, record)
                completed.add(record["exp5_record_key"])
            done = min(start + len(batch_samples), len(pending))
            if args.print_every > 0 and done % args.print_every == 0:
                print(f"[qwen] baseline processed {done}/{len(pending)} pending samples")


def run_knockout_condition_records(
    model: Any,
    processor: Any,
    controller: AttentionKnockoutController,
    samples: list[dict[str, Any]],
    condition: dict[str, Any],
    layer_group: dict[str, Any],
    grid_hw: tuple[int, int],
    image_token_id: int,
    output_jsonl: Path,
    completed: set[str],
    args: argparse.Namespace,
) -> None:
    pending = [
        sample
        for sample in samples
        if f"{sample['sample_id']}::{condition['condition']}::{layer_group['layer_group']}" not in completed
    ]
    for start in range(0, len(pending), args.eval_batch_size):
        batch_samples = pending[start : start + args.eval_batch_size]
        if not batch_samples:
            continue
        records = run_condition_batch(
            model=model,
            processor=processor,
            controller=controller,
            samples=batch_samples,
            condition=condition,
            layer_group=layer_group,
            grid_hw=grid_hw,
            image_token_id=image_token_id,
            args=args,
        )
        for record in records:
            append_jsonl(output_jsonl, record)
            completed.add(record["exp5_record_key"])
        done = min(start + len(batch_samples), len(pending))
        if args.print_every > 0 and done % args.print_every == 0:
            print(
                f"[qwen] {layer_group['layer_group']} {condition['condition']} "
                f"processed {done}/{len(pending)} pending samples"
            )


def main() -> None:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "qwen_exp5_results.jsonl")
    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    all_samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(all_samples, args.start_index, args.max_samples)
    completed = set() if args.no_resume else completed_record_keys(output_jsonl)

    model, processor = load_qwen_model(args)
    grid_hw = sample_qwen_grid(model, processor, samples[0], args)
    image_token_id = qwen_image_token_id(model, processor)
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
    write_json(args.output_dir / "qwen_exp5_run_metadata.json", metadata)
    print(
        f"[qwen] samples={len(samples)} grid={grid_hw} image_token_id={image_token_id} "
        f"resize=({args.resized_height}, {args.resized_width}) layers={num_layers} "
        f"layer_groups={len(layer_groups)} knockout_conditions={len(knockout_conditions)}"
    )

    run_baseline_records(
        model,
        processor,
        samples,
        baseline_conditions,
        grid_hw,
        image_token_id,
        output_jsonl,
        completed,
        args,
    )

    for gidx, layer_group in enumerate(layer_groups, start=1):
        print(f"[qwen] layer group {gidx}/{len(layer_groups)}: {layer_group['layer_group']} layers={layer_group['layers']}")
        controller = AttentionKnockoutController()
        with patched_decoder_attention(model, layer_group["layers"], controller):
            for cidx, condition in enumerate(knockout_conditions, start=1):
                print(f"[qwen] {layer_group['layer_group']} condition {cidx}/{len(knockout_conditions)}: {condition['condition']}")
                run_knockout_condition_records(
                    model=model,
                    processor=processor,
                    controller=controller,
                    samples=samples,
                    condition=condition,
                    layer_group=layer_group,
                    grid_hw=grid_hw,
                    image_token_id=image_token_id,
                    output_jsonl=output_jsonl,
                    completed=completed,
                    args=args,
                )

    records = load_jsonl(output_jsonl)
    summarize_exp5_records(records, args.output_dir, model_name="qwen")
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / 'qwen_exp5_summary.json'}")


if __name__ == "__main__":
    main()
