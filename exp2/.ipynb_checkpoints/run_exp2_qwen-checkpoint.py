#!/usr/bin/env python3
"""Experiment 2 object-token ablation for Qwen2.5-VL-7B-Instruct."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp2_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    add_exp2_fields,
    append_jsonl,
    build_localization_prompt,
    build_relation_prompt,
    build_result_record,
    completed_record_keys,
    make_record_key,
    mask_to_flat,
    object_condition_specs,
    object_mask_for_scope,
    random_mask_for_proportion,
    removed_token_stats,
    select_samples,
    summarize_records,
    unique_object_proportions,
    load_exp1_samples,
    load_jsonl,
)


PROJECT_ROOT = Path("/Users/liuxiwei/Downloads/conference/AAAI27/Exp/whatsup")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp2" / "results_qwen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 Qwen object-token ablation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--global-mean-path", type=Path, default=None)
    parser.add_argument("--recompute-global-mean", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm01", "norm1000"], default="auto")
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--max-new-tokens-relation", type=int, default=8)
    parser.add_argument("--resized-height", type=int, default=None)
    parser.add_argument("--resized-width", type=int, default=None)
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--skip-random", action="store_true")
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

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def get_visual_module(model: Any) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen visual module at model.visual or model.model.visual")


def get_model_device(model: Any) -> Any:
    return next(model.parameters()).device


def get_spatial_merge_size(model: Any) -> int:
    visual = get_visual_module(model)
    if hasattr(visual, "spatial_merge_size"):
        return int(visual.spatial_merge_size)
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "spatial_merge_size"):
        return int(vision_config.spatial_merge_size)
    return 2


def build_qwen_messages(image_path: str, prompt: str, resized_height: int | None, resized_width: int | None) -> list[dict[str, Any]]:
    image_item: dict[str, Any] = {"type": "image", "image": image_path}
    if resized_height is not None:
        image_item["resized_height"] = resized_height
    if resized_width is not None:
        image_item["resized_width"] = resized_width
    return [{"role": "user", "content": [image_item, {"type": "text", "text": prompt}]}]


def build_qwen_inputs(processor: Any, image_path: str, prompt: str, args: argparse.Namespace) -> Any:
    from PIL import Image

    messages = build_qwen_messages(image_path, prompt, args.resized_height, args.resized_width)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    except Exception:
        image = Image.open(image_path).convert("RGB")
        return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def unwrap_visual_output(visual_output: Any) -> Any:
    import torch

    if torch.is_tensor(visual_output):
        return visual_output
    if hasattr(visual_output, "pooler_output") and visual_output.pooler_output is not None:
        return visual_output.pooler_output
    if hasattr(visual_output, "last_hidden_state") and visual_output.last_hidden_state is not None:
        return visual_output.last_hidden_state
    if hasattr(visual_output, "hidden_states") and visual_output.hidden_states is not None:
        return visual_output.hidden_states[-1]
    if isinstance(visual_output, (tuple, list)) and visual_output:
        return visual_output[0]
    raise TypeError(f"Unsupported visual output type: {type(visual_output)!r}")


def qwen_token_grid_from_inputs(model: Any, inputs: Any) -> tuple[int, int]:
    grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    if len(grid) != 3:
        raise ValueError(f"Expected image_grid_thw row to have 3 values, got {grid}")
    temporal, raw_h, raw_w = [int(v) for v in grid]
    if temporal != 1:
        raise ValueError(f"This experiment expects single-image inputs, got temporal={temporal}")
    merge = get_spatial_merge_size(model)
    return raw_h // merge, raw_w // merge


def qwen_visual_embeddings(model: Any, inputs: Any) -> Any:
    import torch

    visual = get_visual_module(model)
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    visual_dtype = getattr(visual, "dtype", next(visual.parameters()).dtype)
    pixel_values = pixel_values.to(device=get_model_device(model), dtype=visual_dtype)
    image_grid_thw = image_grid_thw.to(get_model_device(model))
    with torch.inference_mode():
        return unwrap_visual_output(visual(pixel_values, grid_thw=image_grid_thw))


def compute_global_mean_embedding(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    cache_path: Path,
) -> Any:
    import torch

    if cache_path.is_file() and not args.recompute_global_mean:
        print(f"Loading cached global mean: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    total_sum = None
    total_tokens = 0
    prompt = "Describe the image."
    for idx, sample in enumerate(samples, start=1):
        inputs = build_qwen_inputs(processor, sample["image_path"], prompt, args)
        inputs = inputs.to(get_model_device(model))
        embeds = qwen_visual_embeddings(model, inputs).detach().float().cpu()
        if embeds.ndim == 3:
            embeds = embeds.reshape(-1, embeds.shape[-1])
        if total_sum is None:
            total_sum = embeds.sum(dim=0)
        else:
            total_sum += embeds.sum(dim=0)
        total_tokens += int(embeds.shape[0])
        if args.print_every > 0 and idx % args.print_every == 0:
            print(f"[qwen] global mean images {idx}/{len(samples)}")
    if total_sum is None or total_tokens == 0:
        raise RuntimeError("Could not compute global visual embedding mean")
    mean = total_sum / total_tokens
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, cache_path)
    print(f"Saved global mean: {cache_path}")
    return mean


class QwenVisualAblator:
    def __init__(self, global_mean: Any):
        self.global_mean = global_mean
        self.mask_flat: np.ndarray | None = None

    def set_mask(self, mask_flat: np.ndarray | None) -> None:
        self.mask_flat = mask_flat

    def apply(self, visual_output: Any) -> Any:
        import torch

        if self.mask_flat is None or not self.mask_flat.any():
            return visual_output
        if not torch.is_tensor(visual_output):
            return visual_output

        output = visual_output.clone()
        mask = torch.as_tensor(self.mask_flat, dtype=torch.bool, device=output.device)
        replacement = self.global_mean.to(device=output.device, dtype=output.dtype)
        if output.ndim == 2:
            if output.shape[0] != mask.numel():
                raise ValueError(f"Ablation mask has {mask.numel()} tokens but visual output has shape {tuple(output.shape)}")
            output[mask] = replacement
        elif output.ndim == 3:
            if output.shape[0] != 1 or output.shape[1] != mask.numel():
                raise ValueError(f"Expected [1, {mask.numel()}, D] visual output, got {tuple(output.shape)}")
            output[:, mask, :] = replacement
        else:
            raise ValueError(f"Unsupported visual output shape for ablation: {tuple(output.shape)}")
        return output


@contextlib.contextmanager
def patched_qwen_visual(model: Any, ablator: QwenVisualAblator):
    visual = get_visual_module(model)
    original_forward = visual.forward

    def wrapped_forward(*args, **kwargs):
        return ablator.apply(original_forward(*args, **kwargs))

    visual.forward = wrapped_forward
    try:
        yield
    finally:
        visual.forward = original_forward


def generate_qwen(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    args: argparse.Namespace,
) -> str:
    import torch

    inputs = build_qwen_inputs(processor, image_path, prompt, args)
    inputs = inputs.to(get_model_device(model))
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def sample_grid(model: Any, processor: Any, sample: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    inputs = build_qwen_inputs(processor, sample["image_path"], "Describe the image.", args)
    return qwen_token_grid_from_inputs(model, inputs)


def condition_mask(
    sample: dict[str, Any],
    data_root: Path,
    grid_hw: tuple[int, int],
    condition: dict[str, Any],
) -> np.ndarray:
    if condition["kind"] == "baseline":
        return np.zeros(grid_hw, dtype=bool)
    if condition["kind"] == "object":
        return object_mask_for_scope(sample, data_root, grid_hw, condition["scope"], condition["padding"])
    if condition["kind"] == "random":
        return random_mask_for_proportion(
            grid_hw=grid_hw,
            proportion=condition["random_target_proportion"],
            seed=condition["random_seed"],
            sample_id=sample["sample_id"],
            condition=condition["condition"],
        )
    raise ValueError(f"Unsupported condition kind: {condition['kind']}")


def compute_average_object_proportions(
    samples: list[dict[str, Any]],
    data_root: Path,
    grid_by_sample_id: dict[str, tuple[int, int]],
) -> list[float]:
    records = []
    for condition in object_condition_specs():
        if condition["kind"] != "object":
            continue
        for sample in samples:
            mask = condition_mask(sample, data_root, grid_by_sample_id[sample["sample_id"]], condition)
            stats = removed_token_stats(mask)
            records.append({"condition": condition["condition"], "condition_kind": "object", **stats})
    return unique_object_proportions(records)


def random_condition_specs(proportions: list[float], random_seeds: list[int]) -> list[dict[str, Any]]:
    specs = []
    for proportion in proportions:
        for seed in random_seeds:
            specs.append(
                {
                    "condition": f"random_p_{proportion:.8f}_seed_{seed}",
                    "kind": "random",
                    "scope": "random",
                    "padding": None,
                    "random_seed": seed,
                    "random_target_proportion": proportion,
                }
            )
    return specs


def run_condition_sample(
    model: Any,
    processor: Any,
    ablator: QwenVisualAblator,
    sample: dict[str, Any],
    condition: dict[str, Any],
    grid_hw: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask = condition_mask(sample, args.data_root, grid_hw, condition)
    token_stats = removed_token_stats(mask)
    ablator.set_mask(mask_to_flat(mask))
    target_prompt = build_localization_prompt(sample["target_name"])
    reference_prompt = build_localization_prompt(sample["reference_name"])
    relation_prompt = build_relation_prompt(sample)
    pred_target_text = generate_qwen(model, processor, sample["image_path"], target_prompt, args.max_new_tokens_bbox, args)
    pred_reference_text = generate_qwen(
        model, processor, sample["image_path"], reference_prompt, args.max_new_tokens_bbox, args
    )
    pred_relation_text = generate_qwen(
        model, processor, sample["image_path"], relation_prompt, args.max_new_tokens_relation, args
    )
    ablator.set_mask(None)
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
    return add_exp2_fields(record, condition, token_stats, model_name="Qwen2.5-VL-7B-Instruct")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "qwen_exp2_results.jsonl")
    global_mean_path = args.global_mean_path or (args.output_dir / "qwen_global_average_visual_embedding.pt")
    random_seeds = [int(item.strip()) for item in args.random_seeds.split(",") if item.strip()]

    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    all_samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(all_samples, args.start_index, args.max_samples)
    completed = set() if args.no_resume else completed_record_keys(output_jsonl)

    model, processor = load_qwen_model(args)
    global_mean = compute_global_mean_embedding(model, processor, all_samples, args, global_mean_path)
    ablator = QwenVisualAblator(global_mean)

    print("[qwen] building token grids")
    grid_by_sample_id = {sample["sample_id"]: sample_grid(model, processor, sample, args) for sample in samples}
    proportions = [] if args.skip_random else compute_average_object_proportions(samples, args.data_root, grid_by_sample_id)
    conditions = object_condition_specs() + random_condition_specs(proportions, random_seeds)
    print(f"[qwen] samples={len(samples)} conditions={len(conditions)} random_proportions={proportions}")

    with patched_qwen_visual(model, ablator):
        for cidx, condition in enumerate(conditions, start=1):
            print(f"[qwen] condition {cidx}/{len(conditions)}: {condition['condition']}")
            for sidx, sample in enumerate(samples, start=1):
                key = make_record_key(sample["sample_id"], condition["condition"])
                if key in completed:
                    continue
                record = run_condition_sample(
                    model=model,
                    processor=processor,
                    ablator=ablator,
                    sample=sample,
                    condition=condition,
                    grid_hw=grid_by_sample_id[sample["sample_id"]],
                    args=args,
                )
                append_jsonl(output_jsonl, record)
                completed.add(key)
                if args.print_every > 0 and sidx % args.print_every == 0:
                    print(f"[qwen] {condition['condition']} processed {sidx}/{len(samples)}")

    records = load_jsonl(output_jsonl)
    summarize_records(records, args.output_dir, model_name="qwen")
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / 'qwen_exp2_summary.json'}")


if __name__ == "__main__":
    main()
