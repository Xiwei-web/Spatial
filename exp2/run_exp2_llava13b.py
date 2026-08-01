#!/usr/bin/env python3
"""Experiment 2 object-token ablation for LLaVA-v1.5-13B in the official LLaVA format."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp2_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
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
from run_exp2_llava import LlavaImageFeatureAblator  # noqa: E402


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "llava-1.5-13b"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_llava13b"
SUMMARY_MODEL_NAME = "llava13b"
RECORD_MODEL_NAME = "llava-v1.5-13b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 LLaVA-v1.5-13B object-token ablation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--llava-repo",
        type=Path,
        default=None,
        help="Path to a cloned official LLaVA repo. If set, it is prepended to sys.path before importing llava.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--global-mean-path", type=Path, default=None)
    parser.add_argument("--recompute-global-mean", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm01", "norm1000"], default="auto")
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--max-new-tokens-relation", type=int, default=8)
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument(
        "--show-transformers-warnings",
        action="store_true",
        help="Show verbose Transformers generation warnings. By default these are suppressed.",
    )
    return parser.parse_args()


def configure_transformers_logging(show_warnings: bool) -> None:
    if show_warnings:
        return
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


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


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def model_dtype(model: Any) -> Any:
    return getattr(model, "dtype", next(model.parameters()).dtype)


def get_model_device(model: Any) -> Any:
    return next(model.parameters()).device


def load_llava13b_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"--model-path does not exist or is not a directory: {model_path}\n"
            "Pass the local checkpoint directory with --model-path; see paths.example.sh."
        )
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"--model-path is missing config.json: {model_path}")

    if args.llava_repo is not None:
        repo_path = args.llava_repo.expanduser().resolve()
        if not (repo_path / "llava").is_dir():
            raise FileNotFoundError(f"--llava-repo does not contain a llava/ package directory: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
    except ImportError as exc:
        raise ImportError(
            "This script expects the official LLaVA package to be installed on AutoDL. "
            "Either install it without dependencies, or pass --llava-repo /path/to/LLaVA."
        ) from exc

    model_name = args.model_name or get_model_name_from_path(str(args.model_path))
    load_kwargs = {
        "model_path": str(model_path),
        "model_base": str(args.model_base) if args.model_base else None,
        "model_name": model_name,
        "load_8bit": args.load_8bit,
        "load_4bit": args.load_4bit,
        "device_map": args.device_map,
        "device": args.device,
    }
    if not args.load_8bit and not args.load_4bit:
        load_kwargs["torch_dtype"] = resolve_torch_dtype(args.dtype)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        **supported_kwargs(load_pretrained_model, load_kwargs)
    )
    model.eval()
    torch.set_grad_enabled(False)
    return tokenizer, model, image_processor


def format_llava13b_prompt(model: Any, prompt: str, conv_mode: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from llava.conversation import conv_templates

    if conv_mode not in conv_templates:
        raise KeyError(f"Unknown LLaVA conversation template: {conv_mode}. Available: {sorted(conv_templates)}")

    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(getattr(model, "config", None), "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], f"{image_token}\n{prompt}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_llava13b_inputs(
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    image_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    image = Image.open(image_path).convert("RGB")
    text = format_llava13b_prompt(model, prompt, args.conv_mode)
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(get_model_device(model))

    images = process_images([image], image_processor, model.config)
    dtype = model_dtype(model)
    device = get_model_device(model)
    if isinstance(images, list):
        images = [image_tensor.to(device=device, dtype=dtype) for image_tensor in images]
    else:
        images = images.to(device=device, dtype=dtype)

    return {"input_ids": input_ids, "images": images, "image_sizes": [image.size]}


def encode_images(model: Any, images: Any) -> Any:
    import torch

    with torch.inference_mode():
        if isinstance(images, list):
            return model.encode_images(torch.cat(images, dim=0))
        return model.encode_images(images)


def image_features_to_2d(features: Any) -> Any:
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(f"This script expects batch size 1, got image features {tuple(features.shape)}")
        return features[0]
    if features.ndim == 2:
        return features
    raise ValueError(f"Unsupported LLaVA image feature shape: {tuple(features.shape)}")


def compute_global_mean_embedding(
    tokenizer: Any,
    model: Any,
    image_processor: Any,
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
        inputs = build_llava13b_inputs(tokenizer, model, image_processor, sample["image_path"], prompt, args)
        features = image_features_to_2d(encode_images(model, inputs["images"])).detach().float().cpu()
        if total_sum is None:
            total_sum = features.sum(dim=0)
        else:
            total_sum += features.sum(dim=0)
        total_tokens += int(features.shape[0])
        if args.print_every > 0 and idx % args.print_every == 0:
            print(f"[llava13b] global mean images {idx}/{len(samples)}")

    if total_sum is None or total_tokens == 0:
        raise RuntimeError("Could not compute global visual embedding mean")
    mean = total_sum / total_tokens
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, cache_path)
    print(f"Saved global mean: {cache_path}")
    return mean


def llava13b_token_grid(model: Any, image_processor: Any) -> tuple[int, int]:
    vision_tower = None
    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
    elif hasattr(model, "get_model") and hasattr(model.get_model(), "get_vision_tower"):
        vision_tower = model.get_model().get_vision_tower()

    vision_config = getattr(vision_tower, "config", None)
    image_size = getattr(vision_config, "image_size", None)
    patch_size = getattr(vision_config, "patch_size", None)

    if image_size is None:
        crop_size = getattr(image_processor, "crop_size", None)
        if isinstance(crop_size, dict):
            image_size = crop_size.get("height") or crop_size.get("width")
    if image_size is None:
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            image_size = size.get("shortest_edge") or size.get("height") or size.get("width")
    if patch_size is None:
        patch_size = 14
    if image_size is None:
        raise AttributeError("Could not infer LLaVA-13B token grid from vision tower or image processor")

    grid = int(image_size) // int(patch_size)
    return grid, grid


@contextlib.contextmanager
def patched_encode_images(model: Any, ablator: LlavaImageFeatureAblator):
    original_encode_images = model.encode_images

    def wrapped_encode_images(images):
        return ablator.apply(original_encode_images(images))

    model.encode_images = wrapped_encode_images
    try:
        yield
    finally:
        model.encode_images = original_encode_images


def generate_llava13b(
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    ablator: LlavaImageFeatureAblator,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    args: argparse.Namespace,
) -> str:
    import torch

    inputs = build_llava13b_inputs(tokenizer, model, image_processor, image_path, prompt, args)
    tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True}
    if tokenizer_pad_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer_pad_id

    with torch.inference_mode(), patched_encode_images(model, ablator):
        multimodal_kwargs = {"images": inputs["images"]}
        try:
            generate_params = inspect.signature(model.generate).parameters
            if "image_sizes" in generate_params:
                multimodal_kwargs["image_sizes"] = inputs["image_sizes"]
        except (TypeError, ValueError):
            pass
        output_ids = model.generate(inputs["input_ids"], **multimodal_kwargs, **generate_kwargs)

    output_seq = output_ids[0]
    input_len = inputs["input_ids"].shape[-1]
    if output_seq.shape[-1] > input_len:
        input_seq = inputs["input_ids"][0].to(output_seq.device)
        try:
            has_prompt_prefix = bool((output_seq[:input_len] == input_seq).all().item())
        except RuntimeError:
            has_prompt_prefix = False
        generated_ids = output_seq[input_len:] if has_prompt_prefix else output_seq
    else:
        # Official LLaVA generate() routes through inputs_embeds and may return
        # only newly generated ids, so slicing by the text prompt length would
        # discard the entire answer.
        generated_ids = output_seq
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


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
    grid_hw: tuple[int, int],
) -> list[float]:
    records = []
    for condition in object_condition_specs():
        if condition["kind"] != "object":
            continue
        for sample in samples:
            mask = condition_mask(sample, data_root, grid_hw, condition)
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
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    ablator: LlavaImageFeatureAblator,
    sample: dict[str, Any],
    condition: dict[str, Any],
    grid_hw: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask = condition_mask(sample, args.data_root, grid_hw, condition)
    token_stats = removed_token_stats(mask)
    ablator.set_mask(mask_to_flat(mask))
    ablator.reset_diagnostics()
    target_prompt = build_localization_prompt(sample["target_name"])
    reference_prompt = build_localization_prompt(sample["reference_name"])
    relation_prompt = build_relation_prompt(sample)

    pred_target_text = generate_llava13b(
        tokenizer, model, image_processor, ablator, sample["image_path"], target_prompt, args.max_new_tokens_bbox, args
    )
    pred_reference_text = generate_llava13b(
        tokenizer, model, image_processor, ablator, sample["image_path"], reference_prompt, args.max_new_tokens_bbox, args
    )
    pred_relation_text = generate_llava13b(
        tokenizer, model, image_processor, ablator, sample["image_path"], relation_prompt, args.max_new_tokens_relation, args
    )

    diagnostics = ablator.diagnostics()
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
    record = add_exp2_fields(record, condition, token_stats, model_name=RECORD_MODEL_NAME)
    record.update(diagnostics)
    return record


def main() -> None:
    args = parse_args()
    configure_transformers_logging(args.show_transformers_warnings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "llava13b_exp2_results.jsonl")
    global_mean_path = args.global_mean_path or (args.output_dir / "llava13b_global_average_visual_embedding.pt")
    random_seeds = [int(item.strip()) for item in args.random_seeds.split(",") if item.strip()]

    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    all_samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(all_samples, args.start_index, args.max_samples)
    completed = set() if args.no_resume else completed_record_keys(output_jsonl)

    tokenizer, model, image_processor = load_llava13b_model(args)
    grid_hw = llava13b_token_grid(model, image_processor)
    print(f"[llava13b] token grid: {grid_hw}")
    global_mean = compute_global_mean_embedding(tokenizer, model, image_processor, all_samples, args, global_mean_path)
    ablator = LlavaImageFeatureAblator(global_mean)

    proportions = [] if args.skip_random else compute_average_object_proportions(samples, args.data_root, grid_hw)
    conditions = object_condition_specs() + random_condition_specs(proportions, random_seeds)
    print(f"[llava13b] samples={len(samples)} conditions={len(conditions)} random_proportions={proportions}")

    for cidx, condition in enumerate(conditions, start=1):
        print(f"[llava13b] condition {cidx}/{len(conditions)}: {condition['condition']}")
        for sidx, sample in enumerate(samples, start=1):
            key = make_record_key(sample["sample_id"], condition["condition"])
            if key in completed:
                continue
            record = run_condition_sample(
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                ablator=ablator,
                sample=sample,
                condition=condition,
                grid_hw=grid_hw,
                args=args,
            )
            append_jsonl(output_jsonl, record)
            completed.add(key)
            if args.print_every > 0 and sidx % args.print_every == 0:
                print(f"[llava13b] {condition['condition']} processed {sidx}/{len(samples)}")

    records = load_jsonl(output_jsonl)
    summarize_records(records, args.output_dir, model_name=SUMMARY_MODEL_NAME)
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / f'{SUMMARY_MODEL_NAME}_exp2_summary.json'}")


if __name__ == "__main__":
    main()
