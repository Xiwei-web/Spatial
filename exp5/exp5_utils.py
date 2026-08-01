#!/usr/bin/env python3
"""Shared utilities for Experiment 5 LLM-decoder attention knockout."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_DIR = PROJECT_ROOT / "exp1"
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

from exp1_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    append_jsonl,
    build_localization_prompt,
    build_relation_prompt,
    build_result_record,
    load_exp1_samples,
    load_jsonl,
    safe_mean,
    write_json,
)


IOU_THRESHOLDS = (0.5, 0.7, 0.9)
OBJECT_SCOPES = ("target", "reference", "target_reference")
PADDING_VARIANTS = (("object", 0), ("plus_1_padding", 1))
DEFAULT_LAYER_GRANULARITIES = ("group4", "single", "group6", "all")


def args_to_jsonable(args: Any) -> dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def select_samples(samples: list[dict[str, Any]], start_index: int, max_samples: int | None) -> list[dict[str, Any]]:
    selected = samples[start_index:]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def condition_name(scope: str, variant: str) -> str:
    return f"{scope}_{variant}"


def attention_knockout_condition_specs(random_seeds: list[int], random_match_scope: str) -> list[dict[str, Any]]:
    if random_match_scope not in OBJECT_SCOPES:
        raise ValueError(f"Unsupported random_match_scope={random_match_scope!r}; choose one of {OBJECT_SCOPES}")

    specs = [{"condition": "baseline", "kind": "baseline", "scope": None, "padding": None, "padding_variant": None}]
    for variant, padding in PADDING_VARIANTS:
        for scope in OBJECT_SCOPES:
            specs.append(
                {
                    "condition": condition_name(scope, variant),
                    "kind": "object",
                    "scope": scope,
                    "padding": padding,
                    "padding_variant": variant,
                }
            )
        for seed in random_seeds:
            specs.append(
                {
                    "condition": f"random_{variant}_seed_{seed}",
                    "kind": "random",
                    "scope": "random",
                    "padding": padding,
                    "padding_variant": variant,
                    "random_seed": seed,
                    "random_match_scope": random_match_scope,
                }
            )
    return specs


def load_mask_grid(mask_path: Path | None, grid_hw: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    grid_h, grid_w = grid_hw
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Invalid token grid size: {grid_hw}")
    if mask_path is None or not mask_path.is_file():
        return np.zeros((grid_h, grid_w), dtype=bool)

    mask = Image.open(mask_path).convert("L")
    try:
        resampling = Image.Resampling.BOX
    except AttributeError:
        resampling = Image.BOX
    pooled = mask.resize((grid_w, grid_h), resampling)
    return np.asarray(pooled) > 0


def dilate_mask(mask: Any, radius: int) -> Any:
    import numpy as np

    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dr in range(size):
        for dc in range(size):
            output |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return output


def erode_mask(mask: Any, radius: int) -> Any:
    import numpy as np

    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.ones_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dr in range(size):
        for dc in range(size):
            output &= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return output


def apply_token_padding(mask: Any, padding: int | None) -> Any:
    if padding is None or padding == 0:
        return mask.copy()
    if padding > 0:
        return dilate_mask(mask, padding)
    return erode_mask(mask, abs(padding))


def object_mask_for_scope(
    sample: dict[str, Any],
    data_root: Path,
    grid_hw: tuple[int, int],
    scope: str,
    padding: int | None,
) -> Any:
    import numpy as np

    masks = []
    if scope in {"target", "target_reference"}:
        target_mask_path = sample.get("target_mask_path")
        target_mask = load_mask_grid(data_root / target_mask_path if target_mask_path else None, grid_hw)
        masks.append(apply_token_padding(target_mask, padding))
    if scope in {"reference", "target_reference"}:
        reference_mask_path = sample.get("reference_mask_path")
        reference_mask = load_mask_grid(data_root / reference_mask_path if reference_mask_path else None, grid_hw)
        masks.append(apply_token_padding(reference_mask, padding))
    if not masks:
        return np.zeros(grid_hw, dtype=bool)
    return np.logical_or.reduce(masks)


def random_mask_for_count(
    grid_hw: tuple[int, int],
    k: int,
    seed: int,
    sample_id: str,
    condition: str,
) -> Any:
    import numpy as np

    grid_h, grid_w = grid_hw
    total = grid_h * grid_w
    k = max(0, min(int(k), total))
    mask = np.zeros(total, dtype=bool)
    if k == 0:
        return mask.reshape(grid_hw)

    digest = hashlib.sha256(f"{seed}:{sample_id}:{condition}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(local_seed)
    selected = rng.choice(total, size=k, replace=False)
    mask[selected] = True
    return mask.reshape(grid_hw)


def mask_to_flat(mask: Any) -> Any:
    return mask.reshape(-1).astype(bool)


def removed_token_stats(mask: Any, matched_object_count: int | None = None) -> dict[str, Any]:
    total = int(mask.size)
    removed = int(mask.sum())
    return {
        "removed_token_count": removed,
        "total_image_tokens": total,
        "removed_token_proportion": removed / total if total else 0.0,
        "matched_object_token_count": matched_object_count,
    }


def condition_token_mask(
    sample: dict[str, Any],
    data_root: Path,
    grid_hw: tuple[int, int],
    condition: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if condition["kind"] == "baseline":
        mask = np.zeros(grid_hw, dtype=bool)
        return mask, removed_token_stats(mask)
    if condition["kind"] == "object":
        mask = object_mask_for_scope(sample, data_root, grid_hw, condition["scope"], condition["padding"])
        return mask, removed_token_stats(mask)
    if condition["kind"] == "random":
        match_scope = condition.get("random_match_scope", "target_reference")
        base_mask = object_mask_for_scope(sample, data_root, grid_hw, match_scope, condition["padding"])
        k = int(base_mask.sum())
        mask = random_mask_for_count(
            grid_hw=grid_hw,
            k=k,
            seed=int(condition.get("random_seed", 0)),
            sample_id=sample["sample_id"],
            condition=condition["condition"],
        )
        return mask, removed_token_stats(mask, matched_object_count=k)
    raise ValueError(f"Unsupported condition kind: {condition['kind']}")


def get_attr_by_path(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_decoder_layers(model: Any) -> list[Any]:
    paths = (
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "model.language_model.layers",
        "model.text_model.layers",
        "model.layers",
        "model.model.layers",
        "base_model.model.layers",
        "base_model.model.model.layers",
        "base_model.language_model.model.layers",
        "base_model.model.language_model.model.layers",
        "base_model.model.language_model.layers",
    )
    for path in paths:
        layers = get_attr_by_path(model, path)
        if layers is not None:
            return list(layers)
    candidates = []
    if hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            lowered = name.lower()
            if lowered.endswith(".layers") and hasattr(module, "__len__"):
                try:
                    if len(module) > 0:
                        candidates.append((name, module))
                except TypeError:
                    pass
    decoder_candidates = [
        (name, module)
        for name, module in candidates
        if any(token in name.lower() for token in ("language_model", "text_model", "llm"))
    ]
    if len(decoder_candidates) == 1:
        return list(decoder_candidates[0][1])
    if len(candidates) == 1:
        return list(candidates[0][1])
    raise AttributeError(f"Could not locate decoder layers. Layer-like candidates: {[name for name, _ in candidates[:20]]}")


def find_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "self_attention", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    candidates = []
    if hasattr(layer, "named_modules"):
        for name, module in layer.named_modules():
            last = name.rsplit(".", maxsplit=1)[-1].lower()
            if last in {"self_attn", "self_attention", "attention", "attn"}:
                candidates.append((name, module))
    if len(candidates) == 1:
        return candidates[0][1]
    raise AttributeError(f"Could not locate self-attention module in layer {layer!r}")


def contiguous_layer_groups(num_layers: int, size: int, label: str) -> list[dict[str, Any]]:
    groups = []
    for start in range(0, num_layers, size):
        end = min(start + size, num_layers)
        layers = list(range(start, end))
        groups.append(
            {
                "layer_granularity": label,
                "layer_group": f"{label}_{start:02d}_{end - 1:02d}",
                "layers": layers,
            }
        )
    return groups


def build_layer_group_specs(num_layers: int, granularities: list[str]) -> list[dict[str, Any]]:
    specs = []
    for granularity in granularities:
        if granularity == "single":
            specs.extend(
                {
                    "layer_granularity": "single",
                    "layer_group": f"layer_{idx:02d}",
                    "layers": [idx],
                }
                for idx in range(num_layers)
            )
        elif granularity == "group4":
            specs.extend(contiguous_layer_groups(num_layers, 4, "group4"))
        elif granularity == "group6":
            specs.extend(contiguous_layer_groups(num_layers, 6, "group6"))
        elif granularity == "all":
            specs.append({"layer_granularity": "all", "layer_group": "all_layers", "layers": list(range(num_layers))})
        else:
            raise ValueError(
                f"Unsupported layer granularity {granularity!r}; "
                "use comma-separated values from group4,single,group6,all"
            )
    return specs


class AttentionKnockoutController:
    """Stores per-generation object-token key positions for patched decoder attention."""

    def __init__(self) -> None:
        self.blocked_key_positions_by_batch: list[Any] | None = None
        self.last_image_positions: list[int | None] | None = None

    def clear(self) -> None:
        self.blocked_key_positions_by_batch = None
        self.last_image_positions = None

    def set_from_input_ids(self, input_ids: Any, image_token_id: int, mask_flat: Any) -> None:
        import torch

        batch_size = int(input_ids.shape[0])
        if isinstance(mask_flat, (list, tuple)):
            mask_flats = list(mask_flat)
        else:
            mask_flats = [mask_flat]
        if len(mask_flats) != batch_size:
            raise ValueError(f"Got {len(mask_flats)} knockout masks for batch size {batch_size}")

        blocked_by_batch = []
        last_positions = []
        any_blocked = False
        input_ids_cpu = input_ids.detach().cpu()
        for batch_idx in range(batch_size):
            image_positions = torch.nonzero(
                input_ids_cpu[batch_idx] == int(image_token_id),
                as_tuple=False,
            ).flatten()
            mask_tensor = torch.as_tensor(mask_flats[batch_idx], dtype=torch.bool).flatten().cpu()
            if int(image_positions.numel()) != int(mask_tensor.numel()):
                raise ValueError(
                    f"Batch {batch_idx}: mask has {int(mask_tensor.numel())} image tokens, but input_ids contains "
                    f"{int(image_positions.numel())} image tokens for image_token_id={image_token_id}"
                )
            blocked = image_positions[mask_tensor].to(torch.long)
            blocked_by_batch.append(blocked)
            last_positions.append(int(image_positions.max().item()) if image_positions.numel() else None)
            any_blocked = any_blocked or bool(blocked.numel())

        if not any_blocked:
            self.clear()
            return

        self.blocked_key_positions_by_batch = blocked_by_batch
        self.last_image_positions = last_positions

    def apply(self, hidden_states: Any, attention_mask: Any | None, cache_position: Any | None = None) -> Any:
        import torch

        if self.blocked_key_positions_by_batch is None or self.last_image_positions is None:
            return attention_mask
        if hidden_states.ndim < 3:
            return attention_mask

        batch_size, q_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        if len(self.blocked_key_positions_by_batch) != batch_size:
            raise ValueError(
                f"Knockout controller has {len(self.blocked_key_positions_by_batch)} masks, "
                f"but attention batch size is {batch_size}"
            )

        key_len = infer_key_len(attention_mask, q_len)
        if key_len <= 0:
            return attention_mask

        device = hidden_states.device
        query_start = max(0, key_len - q_len)
        query_positions = torch.arange(query_start, query_start + q_len, device=device, dtype=torch.long)
        dtype = attention_mask.dtype if attention_mask is not None and torch.is_floating_point(attention_mask) else hidden_states.dtype
        if not torch.is_floating_point(torch.empty((), dtype=dtype, device=device)):
            dtype = hidden_states.dtype
        knockout_mask = torch.zeros((batch_size, 1, q_len, key_len), device=device, dtype=dtype)
        mask_value = attention_mask_fill_value(dtype)

        any_applied = False
        for batch_idx, blocked_keys_cpu in enumerate(self.blocked_key_positions_by_batch):
            last_image_position = self.last_image_positions[batch_idx]
            if last_image_position is None:
                continue
            post_image_queries = query_positions > int(last_image_position)
            if not bool(post_image_queries.any()):
                continue
            blocked_keys = blocked_keys_cpu.to(device=device)
            blocked_keys = blocked_keys[(blocked_keys >= 0) & (blocked_keys < key_len)]
            if not bool(blocked_keys.numel()):
                continue
            query_indices = torch.nonzero(post_image_queries, as_tuple=False).flatten()
            knockout_mask[batch_idx : batch_idx + 1, :, query_indices[:, None], blocked_keys[None, :]] = mask_value
            any_applied = True

        if not any_applied:
            return attention_mask

        if attention_mask is None:
            return knockout_mask
        if attention_mask.dtype == torch.bool:
            expanded = attention_mask.clone()
            if expanded.ndim == 2:
                expanded = expanded[:, None, None, :].expand(batch_size, 1, q_len, key_len).clone()
            expanded = expanded & (knockout_mask == 0)
            return expanded
        return attention_mask + knockout_mask.to(dtype=attention_mask.dtype)


def infer_key_len(attention_mask: Any | None, q_len: int) -> int:
    if attention_mask is not None and hasattr(attention_mask, "shape") and len(attention_mask.shape) >= 1:
        return int(attention_mask.shape[-1])
    return q_len


def attention_mask_fill_value(dtype: Any) -> float:
    import torch

    if dtype in {torch.float16, torch.bfloat16}:
        return -1.0e4
    return -1.0e9


def _wrapper_for_attention_forward(original_forward: Any, controller: AttentionKnockoutController) -> Any:
    signature = inspect.signature(original_forward)
    param_names = list(signature.parameters)
    hidden_idx = param_names.index("hidden_states") if "hidden_states" in param_names else 0
    mask_idx = param_names.index("attention_mask") if "attention_mask" in param_names else None
    cache_idx = param_names.index("cache_position") if "cache_position" in param_names else None

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > hidden_idx:
            hidden_states = args[hidden_idx]

        attention_mask = kwargs.get("attention_mask")
        positional_mask = False
        if "attention_mask" not in kwargs and mask_idx is not None and len(args) > mask_idx:
            attention_mask = args[mask_idx]
            positional_mask = True

        cache_position = kwargs.get("cache_position")
        if cache_position is None and cache_idx is not None and len(args) > cache_idx:
            cache_position = args[cache_idx]

        if hidden_states is not None:
            attention_mask = controller.apply(hidden_states, attention_mask, cache_position=cache_position)
            if positional_mask:
                args_list = list(args)
                args_list[mask_idx] = attention_mask
                args = tuple(args_list)
            else:
                kwargs["attention_mask"] = attention_mask
        return original_forward(*args, **kwargs)

    return wrapped_forward


@contextmanager
def patched_decoder_attention(model: Any, layer_indices: list[int], controller: AttentionKnockoutController):
    layers = find_decoder_layers(model)
    originals = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer index {layer_idx} out of range for {len(layers)} decoder layers")
        attn = find_self_attention(layers[layer_idx])
        original_forward = attn.forward
        attn.forward = _wrapper_for_attention_forward(original_forward, controller)
        originals.append((attn, original_forward))
    try:
        yield
    finally:
        controller.clear()
        for attn, original_forward in originals:
            attn.forward = original_forward


def localization_score(iou: float | None) -> float:
    if iou is None:
        return 0.0
    return sum(1.0 for threshold in IOU_THRESHOLDS if iou >= threshold) / len(IOU_THRESHOLDS)


def threshold_fields(prefix: str, iou: float | None) -> dict[str, int]:
    fields = {}
    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        fields[f"{prefix}_iou_at_{suffix}"] = int(iou is not None and iou >= threshold)
    return fields


def make_exp5_record_key(sample_id: str, condition: str, layer_group: str) -> str:
    return f"{sample_id}::{condition}::{layer_group}"


def completed_record_keys(path: Path) -> set[str]:
    return {row["exp5_record_key"] for row in load_jsonl(path) if row.get("exp5_record_key")}


def add_exp5_fields(
    record: dict[str, Any],
    condition: dict[str, Any],
    layer_group: dict[str, Any],
    token_stats: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    target_score = localization_score(record.get("target_iou"))
    reference_score = localization_score(record.get("reference_iou"))
    record["model"] = model_name
    record["condition"] = condition["condition"]
    record["condition_kind"] = condition["kind"]
    record["knockout_scope"] = condition.get("scope")
    record["padding"] = condition.get("padding")
    record["padding_variant"] = condition.get("padding_variant")
    record["random_seed"] = condition.get("random_seed")
    record["random_match_scope"] = condition.get("random_match_scope")
    record["layer_granularity"] = layer_group["layer_granularity"]
    record["layer_group"] = layer_group["layer_group"]
    record["knocked_layers"] = layer_group["layers"]
    record["exp5_record_key"] = make_exp5_record_key(record["sample_id"], record["condition"], record["layer_group"])
    record.update(token_stats)
    record["target_localization_score"] = target_score
    record["reference_localization_score"] = reference_score
    record["mean_localization_score"] = (target_score + reference_score) / 2.0
    record["spatial_relation_correct"] = bool(record.get("rel_correct"))
    record.update(threshold_fields("target", record.get("target_iou")))
    record.update(threshold_fields("reference", record.get("reference_iou")))
    return record


def metric_stats(values: list[float | None]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "sem": None, "n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    sem = std / math.sqrt(n) if n > 0 else None
    return {"mean": mean, "std": std, "sem": sem, "n": n}


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    out = {
        "condition": first.get("condition"),
        "condition_kind": first.get("condition_kind"),
        "knockout_scope": first.get("knockout_scope"),
        "padding": first.get("padding"),
        "padding_variant": first.get("padding_variant"),
        "random_seed": first.get("random_seed"),
        "random_match_scope": first.get("random_match_scope"),
        "layer_granularity": first.get("layer_granularity"),
        "layer_group": first.get("layer_group"),
        "knocked_layers": first.get("knocked_layers"),
        "count": len(rows),
        "avg_removed_token_proportion": safe_mean([row.get("removed_token_proportion") for row in rows]),
        "avg_removed_token_count": safe_mean([row.get("removed_token_count") for row in rows]),
    }
    metrics = {
        "target_localization_accuracy": [row.get("target_localization_score") for row in rows],
        "reference_localization_accuracy": [row.get("reference_localization_score") for row in rows],
        "mean_localization_accuracy": [row.get("mean_localization_score") for row in rows],
        "spatial_relation_accuracy": [1.0 if row.get("rel_correct") else 0.0 for row in rows],
        "mean_target_iou": [row.get("target_iou") for row in rows],
        "mean_reference_iou": [row.get("reference_iou") for row in rows],
    }
    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        metrics[f"target_iou_at_{suffix}"] = [row.get(f"target_iou_at_{suffix}") for row in rows]
        metrics[f"reference_iou_at_{suffix}"] = [row.get(f"reference_iou_at_{suffix}") for row in rows]

    for metric, values in metrics.items():
        stats = metric_stats(values)
        out[metric] = stats["mean"]
        out[f"{metric}_std"] = stats["std"]
        out[f"{metric}_sem"] = stats["sem"]
        out[f"{metric}_n"] = stats["n"]
    return out


def summarize_exp5_records(records: list[dict[str, Any]], output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[(row["condition"], row["layer_group"])].append(row)

    group_rows = [summarize_group(rows) for _, rows in sorted(by_group.items())]
    summary = {
        "model_name": model_name,
        "num_records": len(records),
        "iou_thresholds": IOU_THRESHOLDS,
        "group_summaries": group_rows,
    }
    write_json(output_dir / f"{model_name}_exp5_summary.json", summary)
    write_summary_csv(group_rows, output_dir / f"{model_name}_exp5_group_summary.csv")


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("knocked_layers"), list):
                out["knocked_layers"] = ";".join(str(v) for v in out["knocked_layers"])
            writer.writerow(out)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LAYER_GRANULARITIES",
    "IOU_THRESHOLDS",
    "AttentionKnockoutController",
    "add_exp5_fields",
    "append_jsonl",
    "args_to_jsonable",
    "attention_knockout_condition_specs",
    "build_layer_group_specs",
    "build_localization_prompt",
    "build_relation_prompt",
    "build_result_record",
    "completed_record_keys",
    "condition_token_mask",
    "find_decoder_layers",
    "load_exp1_samples",
    "load_jsonl",
    "mask_to_flat",
    "parse_csv_list",
    "parse_int_csv",
    "patched_decoder_attention",
    "select_samples",
    "summarize_exp5_records",
    "write_json",
]
