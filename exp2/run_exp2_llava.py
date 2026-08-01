#!/usr/bin/env python3
"""Experiment 2 object-token ablation for LLaVA-1.5-7B."""

from __future__ import annotations

import argparse
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


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_llava"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 LLaVA object-token ablation.")
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
    parser.add_argument("--coord-mode", choices=["auto", "pixel", "norm01", "norm1000"], default="auto")
    parser.add_argument("--max-new-tokens-bbox", type=int, default=64)
    parser.add_argument("--max-new-tokens-relation", type=int, default=8)
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


def load_llava_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=resolve_torch_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def get_model_device(model: Any) -> Any:
    return next(model.parameters()).device


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


def llava_token_grid(model: Any) -> tuple[int, int]:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is None:
        raise AttributeError("LLaVA model config has no vision_config")
    image_size = getattr(vision_config, "image_size", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if image_size is None or patch_size is None:
        raise AttributeError("Could not infer LLaVA image token grid from vision_config.image_size/patch_size")
    grid = int(image_size) // int(patch_size)
    return grid, grid


def call_get_image_features(model: Any, inputs: dict[str, Any]) -> Any:
    if not hasattr(model, "get_image_features"):
        raise AttributeError("This script expects LlavaForConditionalGeneration.get_image_features")
    method = model.get_image_features
    try:
        signature = inspect.signature(method)
        params = signature.parameters
        kwargs: dict[str, Any] = {}
        if "pixel_values" in params:
            kwargs["pixel_values"] = inputs["pixel_values"]
        if "vision_feature_layer" in params:
            kwargs["vision_feature_layer"] = get_vision_feature_layer(model)
        if "vision_feature_select_strategy" in params:
            kwargs["vision_feature_select_strategy"] = get_vision_feature_select_strategy(model)
        if "image_sizes" in params and "image_sizes" in inputs:
            kwargs["image_sizes"] = inputs["image_sizes"]
        if "output_hidden_states" in params:
            kwargs["output_hidden_states"] = True
        if kwargs:
            return method(**kwargs)
    except (TypeError, ValueError):
        pass
    return method(inputs["pixel_values"])


def get_vision_feature_layer(model: Any) -> Any:
    layer = getattr(model.config, "vision_feature_layer", None)
    return -2 if layer is None else layer


def get_vision_feature_select_strategy(model: Any) -> str:
    return getattr(model.config, "vision_feature_select_strategy", "default") or "default"


def get_attr_by_path(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def get_multi_modal_projector(model: Any, sample_features: Any | None = None) -> Any:
    direct_paths = (
        "multi_modal_projector",
        "multimodal_projector",
        "mm_projector",
        "vision_projector",
        "visual_projector",
        "projector",
        "model.multi_modal_projector",
        "model.multimodal_projector",
        "model.mm_projector",
        "model.vision_projector",
        "model.visual_projector",
        "base_model.multi_modal_projector",
        "base_model.mm_projector",
        "base_model.model.multi_modal_projector",
        "base_model.model.mm_projector",
    )
    for path in direct_paths:
        projector = get_attr_by_path(model, path)
        if projector is not None:
            return projector

    candidates = []
    if hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            lowered = name.lower()
            last = lowered.rsplit(".", maxsplit=1)[-1]
            if (
                "multi_modal_projector" in lowered
                or "multimodal_projector" in lowered
                or "mm_projector" in lowered
                or "vision_projector" in lowered
                or "visual_projector" in lowered
                or last == "projector"
            ):
                candidates.append((name, module))

    if len(candidates) == 1:
        return candidates[0][1]

    if sample_features is not None and candidates:
        working = select_projector_by_trial(candidates, sample_features)
        if working is not None:
            return working

    candidate_names = [name for name, _ in candidates[:50]]
    top_children = list(getattr(model, "_modules", {}).keys())
    raise AttributeError(
        "Could not find LLaVA multimodal projector on the model. "
        f"Top-level children: {top_children}. Projector-like candidates: {candidate_names}."
    )


def select_projector_by_trial(candidates: list[tuple[str, Any]], sample_features: Any) -> Any | None:
    import torch

    if not torch.is_tensor(sample_features) or sample_features.ndim < 3:
        return None
    probe = sample_features[:, : min(2, sample_features.shape[1]), :]
    for _, module in candidates:
        if not callable(module):
            continue
        try:
            with torch.inference_mode():
                output = module(probe)
            if torch.is_tensor(output) and output.ndim == probe.ndim and output.shape[:2] == probe.shape[:2]:
                return module
        except Exception:
            continue
    return None


def select_vision_hidden_state(model: Any, image_outputs: Any) -> Any:
    import torch

    hidden_states = getattr(image_outputs, "hidden_states", None)
    layer = get_vision_feature_layer(model)

    if hidden_states is not None:
        if isinstance(layer, (list, tuple)):
            selected = [hidden_states[int(idx)] for idx in layer]
            if not selected:
                raise ValueError("vision_feature_layer is an empty list")
            image_features = selected[0] if len(selected) == 1 else torch.cat(selected, dim=-1)
        else:
            image_features = hidden_states[int(layer)]
    elif getattr(image_outputs, "last_hidden_state", None) is not None:
        image_features = image_outputs.last_hidden_state
    else:
        raise TypeError(
            "LLaVA image output is not a tensor and has neither hidden_states nor last_hidden_state. "
            f"Got {type(image_outputs)!r}."
        )

    return apply_feature_select_strategy(image_features, model)


def apply_feature_select_strategy(image_features: Any, model: Any) -> Any:
    strategy = get_vision_feature_select_strategy(model)
    if strategy == "default":
        return image_features[:, 1:]
    if strategy == "full":
        return image_features
    raise ValueError(f"Unsupported LLaVA vision_feature_select_strategy: {strategy}")


def normalize_image_features(model: Any, features: Any) -> Any:
    import torch

    if torch.is_tensor(features):
        return features
    if isinstance(features, (tuple, list)) and features and torch.is_tensor(features[0]):
        return features[0]
    if hasattr(features, "hidden_states") or hasattr(features, "last_hidden_state"):
        selected = select_vision_hidden_state(model, features)
        projector = get_multi_modal_projector(model, selected)
        return projector(selected)
    raise TypeError(f"Unsupported LLaVA image feature type: {type(features)!r}")


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
        inputs = build_llava_inputs(processor, sample["image_path"], prompt, model)
        with torch.inference_mode():
            features = normalize_image_features(model, call_get_image_features(model, inputs)).detach().float().cpu()
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        if total_sum is None:
            total_sum = features.sum(dim=0)
        else:
            total_sum += features.sum(dim=0)
        total_tokens += int(features.shape[0])
        if args.print_every > 0 and idx % args.print_every == 0:
            print(f"[llava] global mean images {idx}/{len(samples)}")
    if total_sum is None or total_tokens == 0:
        raise RuntimeError("Could not compute global visual embedding mean")
    mean = total_sum / total_tokens
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, cache_path)
    print(f"Saved global mean: {cache_path}")
    return mean


class LlavaImageFeatureAblator:
    def __init__(self, global_mean: Any):
        self.global_mean = global_mean
        self.mask_flat: np.ndarray | None = None
        self.reset_diagnostics()

    def set_mask(self, mask_flat: np.ndarray | None) -> None:
        self.mask_flat = mask_flat

    def reset_diagnostics(self) -> None:
        self.apply_calls = 0
        self.changed_calls = 0
        self.delta_l2_sum = 0.0
        self.delta_l2_max = 0.0
        self.last_feature_shape: tuple[int, ...] | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "ablation_apply_calls": self.apply_calls,
            "ablation_changed_calls": self.changed_calls,
            "ablation_delta_l2_sum": self.delta_l2_sum,
            "ablation_delta_l2_max": self.delta_l2_max,
            "ablation_last_feature_shape": list(self.last_feature_shape) if self.last_feature_shape else None,
        }

    def _record_delta(self, before: Any, after: Any) -> None:
        import torch

        if before.numel() == 0:
            return
        delta = torch.linalg.vector_norm((after.detach().float() - before.detach().float()).reshape(-1))
        value = float(delta.cpu())
        self.changed_calls += 1
        self.delta_l2_sum += value
        self.delta_l2_max = max(self.delta_l2_max, value)

    def apply_tensor(self, features: Any) -> Any:
        import torch

        self.apply_calls += 1
        self.last_feature_shape = tuple(int(v) for v in features.shape)
        if self.mask_flat is None or not self.mask_flat.any():
            return features
        output = features.clone()
        mask = torch.as_tensor(self.mask_flat, dtype=torch.bool, device=output.device)
        replacement = self.global_mean.to(device=output.device, dtype=output.dtype)

        if output.ndim == 2:
            if output.shape[0] == mask.numel() + 1:
                mask = torch.cat([torch.zeros(1, dtype=torch.bool, device=output.device), mask], dim=0)
            if output.shape[0] != mask.numel():
                raise ValueError(f"Ablation mask has {mask.numel()} tokens but image features have shape {tuple(output.shape)}")
            before = output[mask].clone()
            output[mask] = replacement
            self._record_delta(before, output[mask])
        elif output.ndim == 3:
            if output.shape[0] != 1:
                raise ValueError(f"This script expects batch size 1, got image features {tuple(output.shape)}")
            if output.shape[1] == mask.numel() + 1:
                mask = torch.cat([torch.zeros(1, dtype=torch.bool, device=output.device), mask], dim=0)
            if output.shape[1] != mask.numel():
                raise ValueError(f"Ablation mask has {mask.numel()} tokens but image features have shape {tuple(output.shape)}")
            before = output[:, mask, :].clone()
            output[:, mask, :] = replacement
            self._record_delta(before, output[:, mask, :])
        else:
            raise ValueError(f"Unsupported LLaVA image feature shape for ablation: {tuple(output.shape)}")
        return output

    def apply(self, features: Any) -> Any:
        import torch

        if torch.is_tensor(features):
            return self.apply_tensor(features)
        if isinstance(features, tuple) and features and torch.is_tensor(features[0]):
            return (self.apply_tensor(features[0]), *features[1:])
        if isinstance(features, list) and features and torch.is_tensor(features[0]):
            return [self.apply_tensor(features[0]), *features[1:]]
        return features


def llava_image_token_id(model: Any, processor: Any) -> int:
    config = getattr(model, "config", None)
    for attr in ("image_token_index", "image_token_id"):
        token_id = getattr(config, attr, None)
        if token_id is not None:
            return int(token_id)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        token_id = tokenizer.convert_tokens_to_ids("<image>")
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    raise AttributeError("Could not infer LLaVA image token id from model config or tokenizer")


def text_embedding_module(model: Any) -> Any:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    for path in (
        "language_model.model.embed_tokens",
        "language_model.embed_tokens",
        "model.embed_tokens",
    ):
        current = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None:
            return current
    raise AttributeError("Could not find LLaVA text embedding module")


def llava_inputs_embeds_with_ablation(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    ablator: LlavaImageFeatureAblator,
) -> Any:
    import torch

    input_ids = inputs["input_ids"]
    inputs_embeds = text_embedding_module(model)(input_ids).clone()
    image_features = normalize_image_features(model, call_get_image_features(model, inputs))
    image_features = ablator.apply(image_features)
    if image_features.ndim == 3:
        if image_features.shape[0] != 1:
            raise ValueError(f"This script expects batch size 1, got image features {tuple(image_features.shape)}")
        image_features = image_features[0]
    if image_features.ndim != 2:
        raise ValueError(f"Expected 2D LLaVA image features after flattening, got {tuple(image_features.shape)}")

    image_token_id = llava_image_token_id(model, processor)
    image_token_mask = input_ids == image_token_id
    image_token_count = int(image_token_mask.sum().item())
    if image_token_count != int(image_features.shape[0]):
        raise ValueError(
            "LLaVA image token count does not match visual features: "
            f"input_ids have {image_token_count} image tokens, vision path returned {image_features.shape[0]}."
        )
    image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    inputs_embeds[image_token_mask] = image_features
    return inputs_embeds


def llava_generation_inputs(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    ablator: LlavaImageFeatureAblator,
) -> dict[str, Any]:
    inputs_embeds = llava_inputs_embeds_with_ablation(model, processor, inputs, ablator)
    keep_keys = ("input_ids", "attention_mask")
    generation_inputs = {key: inputs[key] for key in keep_keys if key in inputs}
    generation_inputs["inputs_embeds"] = inputs_embeds
    return generation_inputs


def generate_llava(
    model: Any,
    processor: Any,
    ablator: LlavaImageFeatureAblator,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = build_llava_inputs(processor, image_path, prompt, model)
    generation_inputs = llava_generation_inputs(model, processor, inputs, ablator)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    with torch.inference_mode():
        output_ids = model.generate(**generation_inputs, **generate_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


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
    model: Any,
    processor: Any,
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
    pred_target_text = generate_llava(
        model, processor, ablator, sample["image_path"], target_prompt, args.max_new_tokens_bbox
    )
    pred_reference_text = generate_llava(
        model, processor, ablator, sample["image_path"], reference_prompt, args.max_new_tokens_bbox
    )
    pred_relation_text = generate_llava(
        model, processor, ablator, sample["image_path"], relation_prompt, args.max_new_tokens_relation
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
    record = add_exp2_fields(record, condition, token_stats, model_name="llava-1.5-7b-hf")
    record.update(diagnostics)
    return record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (args.output_dir / "llava_exp2_results.jsonl")
    global_mean_path = args.global_mean_path or (args.output_dir / "llava_global_average_visual_embedding.pt")
    random_seeds = [int(item.strip()) for item in args.random_seeds.split(",") if item.strip()]

    if output_jsonl.exists() and args.overwrite:
        output_jsonl.unlink()

    all_samples = load_exp1_samples(data_root=args.data_root)
    samples = select_samples(all_samples, args.start_index, args.max_samples)
    completed = set() if args.no_resume else completed_record_keys(output_jsonl)

    model, processor = load_llava_model(args)
    grid_hw = llava_token_grid(model)
    print(f"[llava] token grid: {grid_hw}")
    global_mean = compute_global_mean_embedding(model, processor, all_samples, args, global_mean_path)
    ablator = LlavaImageFeatureAblator(global_mean)

    proportions = [] if args.skip_random else compute_average_object_proportions(samples, args.data_root, grid_hw)
    conditions = object_condition_specs() + random_condition_specs(proportions, random_seeds)
    print(f"[llava] samples={len(samples)} conditions={len(conditions)} random_proportions={proportions}")

    for cidx, condition in enumerate(conditions, start=1):
        print(f"[llava] condition {cidx}/{len(conditions)}: {condition['condition']}")
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
                grid_hw=grid_hw,
                args=args,
            )
            append_jsonl(output_jsonl, record)
            completed.add(key)
            if args.print_every > 0 and sidx % args.print_every == 0:
                print(f"[llava] {condition['condition']} processed {sidx}/{len(samples)}")

    records = load_jsonl(output_jsonl)
    summarize_records(records, args.output_dir, model_name="llava")
    print(f"Done. Results: {output_jsonl}")
    print(f"Summary: {args.output_dir / 'llava_exp2_summary.json'}")


if __name__ == "__main__":
    main()
