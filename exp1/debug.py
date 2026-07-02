#!/usr/bin/env python3
"""
Diagnose whether Qwen2.5-VL visual RoPE is really disabled.

This script runs a small visual-tower forward pass and writes a JSON report with:
  - discovered RoPE/position-embedding paths in the visual module;
  - runtime call counts for rot_pos_emb, rotary_pos_emb, apply_rotary_pos_emb_vision,
    rotary_pos_emb forward args, and position_embeddings forward args;
  - embedding differences among baseline, legacy rot_pos_emb-only disabling, and
    the full no-RoPE patch used by visualize_qwen_direction_vectors_pca.py.

No generation is performed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import importlib
import inspect
import json
import platform
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import visualize_qwen_direction_vectors_pca as viz  # noqa: E402


DEFAULT_OUTPUT_PATH = viz.DEFAULT_OUTPUT_DIR / "qwen_visual_rope_diagnostics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Qwen2.5-VL visual RoPE disabling.")
    parser.add_argument("--model-dir", type=Path, default=viz.DEFAULT_MODEL_DIR)
    parser.add_argument("--data-root", type=Path, default=viz.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional image path. If omitted, one sample is selected from the WhatIsUp data.",
    )
    parser.add_argument("--source-dataset", default="B")
    parser.add_argument("--pair-target", default=None)
    parser.add_argument("--pair-reference", default=None)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--relation", choices=viz.RELATION_ORDER, default="left_of")
    parser.add_argument("--sample-index", type=int, default=0)
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
    parser.add_argument("--resized-height", type=int, default=None)
    parser.add_argument("--resized-width", type=int, default=None)
    parser.add_argument(
        "--max-call-details",
        type=int,
        default=8,
        help="Maximum detailed call records to keep per hook. Counts are still complete.",
    )
    parser.add_argument(
        "--max-discovered-modules",
        type=int,
        default=120,
        help="Maximum discovered modules/parameters to store in the report.",
    )
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def tensor_stats(tensor: torch.Tensor, sample_values: int = 6) -> dict[str, Any]:
    detached = tensor.detach()
    stats: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }
    if detached.numel() == 0:
        return stats

    finite = torch.isfinite(detached) if torch.is_floating_point(detached) or torch.is_complex(detached) else None
    if finite is not None:
        stats["finite"] = bool(finite.all().item())

    if torch.is_floating_point(detached) or torch.is_complex(detached):
        x = detached.float().cpu()
        stats.update(
            {
                "min": float(x.min().item()),
                "max": float(x.max().item()),
                "mean": float(x.mean().item()),
                "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
                "abs_max": float(x.abs().max().item()),
                "l2_norm": float(torch.linalg.vector_norm(x).item()),
            }
        )
        stats["sample"] = [float(v) for v in x.flatten()[:sample_values].tolist()]
    else:
        x = detached.cpu()
        stats.update(
            {
                "min": int(x.min().item()),
                "max": int(x.max().item()),
                "sample": [int(v) for v in x.flatten()[:sample_values].tolist()],
            }
        )
    return stats


def value_stats(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return {"type": type(value).__name__, "repr": repr(value)[:200]}
    if torch.is_tensor(value):
        return tensor_stats(value)
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "len": len(value),
            "items": [value_stats(item, depth + 1) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "len": len(value),
            "items": [value_stats(item, depth + 1) for item in value[:8]],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
            "items": {str(k): value_stats(v, depth + 1) for k, v in list(value.items())[:8]},
        }
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def no_rope_value(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, tuple) and len(value) == 2 and all(torch.is_tensor(item) for item in value):
        cos, sin = value
        return torch.ones_like(cos), torch.zeros_like(sin)
    if isinstance(value, list) and len(value) == 2 and all(torch.is_tensor(item) for item in value):
        cos, sin = value
        return [torch.ones_like(cos), torch.zeros_like(sin)]
    return value


def diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    a = left.detach().float().cpu()
    b = right.detach().float().cpu()
    delta = a - b
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    delta_norm = torch.linalg.vector_norm(delta)
    denom = torch.clamp(a_norm * b_norm, min=1e-12)
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "max_abs_diff": float(delta.abs().max().item()),
        "mean_abs_diff": float(delta.abs().mean().item()),
        "l2_diff": float(delta_norm.item()),
        "relative_l2_diff_vs_left": float((delta_norm / torch.clamp(a_norm, min=1e-12)).item()),
        "left_l2_norm": float(a_norm.item()),
        "right_l2_norm": float(b_norm.item()),
        "cosine": float((torch.sum(a.flatten() * b.flatten()) / denom).item()),
    }


class Recorder:
    def __init__(self, max_call_details: int):
        self.max_call_details = max_call_details
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.details: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self.patches: list[dict[str, Any]] = []

    def record_patch(self, kind: str, target: str) -> None:
        self.patches.append({"kind": kind, "target": target})

    def record_call(self, key: str, detail: dict[str, Any] | None = None) -> None:
        self.counts[key] += 1
        if detail is not None and len(self.details[key]) < self.max_call_details:
            self.details[key].append(detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "details": dict(sorted(self.details.items())),
            "patches": self.patches,
            "patch_count": len(self.patches),
        }


def patch_attr(restorers: list[Callable[[], None]], obj: Any, name: str, value: Any) -> None:
    had_instance_attr = hasattr(obj, "__dict__") and name in obj.__dict__
    instance_original = obj.__dict__.get(name) if had_instance_attr else None
    original = getattr(obj, name)
    setattr(obj, name, value)

    def restore() -> None:
        if had_instance_attr:
            setattr(obj, name, instance_original)
            return
        try:
            delattr(obj, name)
        except AttributeError:
            setattr(obj, name, original)

    restorers.append(restore)


def module_label(module: Any) -> str:
    name = module.__class__.__name__
    full = module.__class__.__module__
    return f"{full}.{name}"


def patch_rot_pos_emb_only(visual: Any, recorder: Recorder, restorers: list[Callable[[], None]]) -> None:
    if not hasattr(visual, "rot_pos_emb") or not callable(visual.rot_pos_emb):
        recorder.record_call("legacy_rot_pos_emb_only.missing")
        return

    original = visual.rot_pos_emb

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        original_output = original(*args, **kwargs)
        patched_output = no_rope_value(original_output)
        recorder.record_call(
            "legacy_rot_pos_emb_only.rot_pos_emb",
            {"original_output": value_stats(original_output), "patched_output": value_stats(patched_output)},
        )
        return patched_output

    patch_attr(restorers, visual, "rot_pos_emb", wrapped)
    recorder.record_patch("zero_output", "visual.rot_pos_emb")


def patch_visual_rope_sources(visual: Any, recorder: Recorder, restorers: list[Callable[[], None]]) -> None:
    if hasattr(visual, "rot_pos_emb") and callable(visual.rot_pos_emb):
        original = visual.rot_pos_emb

        @functools.wraps(original)
        def wrapped_rot_pos_emb(*args, **kwargs):
            original_output = original(*args, **kwargs)
            patched_output = no_rope_value(original_output)
            recorder.record_call(
                "full.rot_pos_emb",
                {"original_output": value_stats(original_output), "patched_output": value_stats(patched_output)},
            )
            return patched_output

        patch_attr(restorers, visual, "rot_pos_emb", wrapped_rot_pos_emb)
        recorder.record_patch("zero_output", "visual.rot_pos_emb")

    rotary_pos_emb = getattr(visual, "rotary_pos_emb", None)
    if rotary_pos_emb is not None:
        if hasattr(rotary_pos_emb, "forward") and callable(rotary_pos_emb.forward):
            original_forward = rotary_pos_emb.forward

            @functools.wraps(original_forward)
            def wrapped_rotary_forward(*args, **kwargs):
                original_output = original_forward(*args, **kwargs)
                patched_output = no_rope_value(original_output)
                recorder.record_call(
                    "full.visual.rotary_pos_emb.forward",
                    {"original_output": value_stats(original_output), "patched_output": value_stats(patched_output)},
                )
                return patched_output

            patch_attr(restorers, rotary_pos_emb, "forward", wrapped_rotary_forward)
            recorder.record_patch("zero_output", "visual.rotary_pos_emb.forward")
        elif callable(rotary_pos_emb):
            original_rotary = rotary_pos_emb

            @functools.wraps(original_rotary)
            def wrapped_rotary(*args, **kwargs):
                original_output = original_rotary(*args, **kwargs)
                patched_output = no_rope_value(original_output)
                recorder.record_call(
                    "full.visual.rotary_pos_emb",
                    {"original_output": value_stats(original_output), "patched_output": value_stats(patched_output)},
                )
                return patched_output

            patch_attr(restorers, visual, "rotary_pos_emb", wrapped_rotary)
            recorder.record_patch("zero_output", "visual.rotary_pos_emb")


def patch_apply_rotary_pos_emb_vision(recorder: Recorder, restorers: list[Callable[[], None]]) -> None:
    def identity_apply_rotary_pos_emb_vision(q, k, *args, **kwargs):
        recorder.record_call(
            "full.apply_rotary_pos_emb_vision",
            {
                "q": value_stats(q),
                "k": value_stats(k),
                "extra_arg_types": [type(arg).__name__ for arg in args],
                "kwarg_keys": sorted(kwargs.keys()),
            },
        )
        return q, k

    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        module_name = getattr(module, "__name__", "")
        lowered = module_name.lower()
        if "qwen" not in lowered or "vl" not in lowered:
            continue
        if not hasattr(module, "apply_rotary_pos_emb_vision"):
            continue
        patch_attr(restorers, module, "apply_rotary_pos_emb_vision", identity_apply_rotary_pos_emb_vision)
        module_file = getattr(module, "__file__", None)
        recorder.record_patch("identity_qk", f"{module_name}.apply_rotary_pos_emb_vision ({module_file})")


def patch_forward_rope_args(visual: Any, recorder: Recorder, restorers: list[Callable[[], None]]) -> None:
    rope_arg_names = {"rotary_pos_emb", "position_embeddings"}
    seen_module_ids: set[int] = set()

    for module_name, module in visual.named_modules():
        module_id = id(module)
        if module_id in seen_module_ids or not hasattr(module, "forward"):
            continue
        seen_module_ids.add(module_id)
        forward = module.forward
        if not callable(forward):
            continue
        try:
            signature = inspect.signature(forward)
        except (TypeError, ValueError):
            continue

        param_names = list(signature.parameters)
        found = sorted(rope_arg_names.intersection(param_names))
        if not found:
            continue

        @functools.wraps(forward)
        def wrapped_forward(
            *args,
            _forward=forward,
            _param_names=param_names,
            _module_name=module_name,
            _module_label=module_label(module),
            **kwargs,
        ):
            args_list = list(args)
            details: dict[str, Any] = {
                "module": _module_name,
                "class": _module_label,
                "sanitized": {},
            }
            for arg_name in rope_arg_names:
                if arg_name in kwargs:
                    before = kwargs[arg_name]
                    after = no_rope_value(before)
                    kwargs[arg_name] = after
                    details["sanitized"][arg_name] = {
                        "source": "kwargs",
                        "before": value_stats(before),
                        "after": value_stats(after),
                    }
                if arg_name in _param_names:
                    arg_index = _param_names.index(arg_name)
                    if arg_index < len(args_list):
                        before = args_list[arg_index]
                        after = no_rope_value(before)
                        args_list[arg_index] = after
                        details["sanitized"][arg_name] = {
                            "source": "args",
                            "before": value_stats(before),
                            "after": value_stats(after),
                        }

            if details["sanitized"]:
                recorder.record_call(f"full.forward_args.{_module_name or '<root>'}", details)
            return _forward(*args_list, **kwargs)

        patch_attr(restorers, module, "forward", wrapped_forward)
        recorder.record_patch("sanitize_forward_args", f"{module_name or '<root>'}.{module.__class__.__name__}{signature}")


@contextlib.contextmanager
def diagnostic_patch_context(visual: Any, mode: str, recorder: Recorder):
    restorers: list[Callable[[], None]] = []
    if mode == "baseline":
        yield
        return
    if mode == "legacy_rot_pos_emb_only":
        patch_rot_pos_emb_only(visual, recorder, restorers)
    elif mode == "full_disable_rope":
        patch_visual_rope_sources(visual, recorder, restorers)
        patch_apply_rotary_pos_emb_vision(recorder, restorers)
        patch_forward_rope_args(visual, recorder, restorers)
    else:
        raise ValueError(f"Unsupported diagnostic patch mode: {mode}")

    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def import_version(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def environment_report() -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": import_version("torch"),
        "transformers": import_version("transformers"),
        "qwen_vl_utils": import_version("qwen_vl_utils"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def discovered_apply_rotary_modules() -> list[dict[str, Any]]:
    found = []
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        module_name = getattr(module, "__name__", "")
        lowered = module_name.lower()
        if "qwen" not in lowered or "vl" not in lowered:
            continue
        if hasattr(module, "apply_rotary_pos_emb_vision"):
            func = getattr(module, "apply_rotary_pos_emb_vision")
            try:
                signature = str(inspect.signature(func))
            except (TypeError, ValueError):
                signature = None
            found.append(
                {
                    "module": module_name,
                    "file": getattr(module, "__file__", None),
                    "function_repr": repr(func),
                    "signature": signature,
                }
            )
    return found


def discover_visual_paths(visual: Any, max_items: int) -> dict[str, Any]:
    module_forward_args = []
    interesting_attrs = []
    interesting_params = []

    attr_names = [
        "rot_pos_emb",
        "rotary_pos_emb",
        "position_embedding",
        "position_embeddings",
        "pos_embed",
        "spatial_merge_size",
        "window_size",
        "fullatt_block_indexes",
    ]
    for attr_name in attr_names:
        if hasattr(visual, attr_name):
            attr = getattr(visual, attr_name)
            interesting_attrs.append(
                {
                    "owner": "<visual>",
                    "attr": attr_name,
                    "type": type(attr).__name__,
                    "repr": repr(attr)[:300],
                    "callable": callable(attr),
                }
            )

    for module_name, module in visual.named_modules():
        if len(module_forward_args) >= max_items and len(interesting_attrs) >= max_items:
            break
        for attr_name in attr_names:
            if len(interesting_attrs) >= max_items:
                break
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                interesting_attrs.append(
                    {
                        "owner": module_name or "<root>",
                        "owner_class": module_label(module),
                        "attr": attr_name,
                        "type": type(attr).__name__,
                        "repr": repr(attr)[:300],
                        "callable": callable(attr),
                    }
                )
        if hasattr(module, "forward") and callable(module.forward):
            try:
                signature = inspect.signature(module.forward)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                params = set(signature.parameters)
                if {"rotary_pos_emb", "position_embeddings"}.intersection(params):
                    module_forward_args.append(
                        {
                            "module": module_name or "<root>",
                            "class": module_label(module),
                            "signature": str(signature),
                        }
                    )

    for name, param in visual.named_parameters():
        lowered = name.lower()
        if any(key in lowered for key in ("pos", "rope", "rotary")):
            interesting_params.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": bool(param.requires_grad),
                }
            )
        if len(interesting_params) >= max_items:
            break

    return {
        "class": module_label(visual),
        "repr": repr(visual)[:1000],
        "module_count": sum(1 for _ in visual.modules()),
        "interesting_attrs": interesting_attrs[:max_items],
        "forward_args_accepting_rope": module_forward_args[:max_items],
        "named_parameters_with_pos_rope_rotary": interesting_params[:max_items],
        "apply_rotary_pos_emb_vision_modules": discovered_apply_rotary_modules(),
    }


def select_image(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.image_path:
        return args.image_path, {"source": "arg", "image_path": args.image_path}

    grouped, selected_pair = viz.load_complete_pair_samples(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        pair_target=args.pair_target,
        pair_reference=args.pair_reference,
        pair_index=args.pair_index,
        list_only=False,
    )
    relation_samples = grouped[args.relation]
    if not relation_samples:
        raise ValueError(f"No samples found for relation {args.relation}")
    sample_index = args.sample_index % len(relation_samples)
    sample = relation_samples[sample_index]
    return sample["image_path"], {
        "source": "dataset",
        "selected_pair": selected_pair,
        "relation": args.relation,
        "sample_index": sample_index,
        "sample_id": sample.get("sample_id"),
        "image_path": sample["image_path"],
        "target_name": sample.get("target_name"),
        "reference_name": sample.get("reference_name"),
        "gt_relation": sample.get("gt_relation"),
        "source_dataset": sample.get("source_dataset"),
    }


def build_visual_inputs(model: Any, processor: Any, image_path: str, args: argparse.Namespace) -> dict[str, Any]:
    visual = viz.get_visual_module(model)
    device = viz.get_model_device(model)
    inputs = viz.build_qwen_inputs(processor, image_path, args.resized_height, args.resized_width)
    inputs = inputs.to(device)
    if "pixel_values" not in inputs:
        raise KeyError("Processor output does not contain pixel_values")

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    visual_dtype = getattr(visual, "dtype", next(visual.parameters()).dtype)
    pixel_values = pixel_values.to(dtype=visual_dtype)
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "input_ids_shape": list(inputs["input_ids"].shape) if "input_ids" in inputs else None,
        "pixel_values_stats": tensor_stats(pixel_values),
        "image_grid_thw_stats": tensor_stats(image_grid_thw),
    }


def run_visual_mode(
    model: Any,
    visual_inputs: dict[str, Any],
    mode: str,
    max_call_details: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    visual = viz.get_visual_module(model)
    recorder = Recorder(max_call_details=max_call_details)
    with diagnostic_patch_context(visual, mode, recorder):
        with torch.inference_mode():
            visual_output = visual(visual_inputs["pixel_values"], grid_thw=visual_inputs["image_grid_thw"])
            image_embeds = viz.unwrap_visual_output(visual_output).detach().float().cpu()

    mode_report = {
        "mode": mode,
        "embedding_stats": tensor_stats(image_embeds),
        "recorder": recorder.as_dict(),
    }
    return image_embeds, mode_report


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "args": vars(args),
        "environment": environment_report(),
        "status": "started",
    }

    try:
        image_path, sample_info = select_image(args)
        report["sample"] = sample_info

        model, processor = viz.load_qwen_model(args)
        visual = viz.get_visual_module(model)
        report["visual_discovery"] = discover_visual_paths(visual, args.max_discovered_modules)

        visual_inputs = build_visual_inputs(model, processor, image_path, args)
        report["visual_inputs"] = {
            "input_ids_shape": visual_inputs["input_ids_shape"],
            "pixel_values_stats": visual_inputs["pixel_values_stats"],
            "image_grid_thw_stats": visual_inputs["image_grid_thw_stats"],
            "image_grid_thw_values": visual_inputs["image_grid_thw"].detach().cpu().tolist(),
        }

        mode_outputs: dict[str, torch.Tensor] = {}
        mode_reports: dict[str, Any] = {}
        for mode in ("baseline", "legacy_rot_pos_emb_only", "full_disable_rope"):
            embeds, mode_report = run_visual_mode(
                model=model,
                visual_inputs=visual_inputs,
                mode=mode,
                max_call_details=args.max_call_details,
            )
            mode_outputs[mode] = embeds
            mode_reports[mode] = mode_report

        report["modes"] = mode_reports
        report["embedding_diffs"] = {
            "baseline_vs_legacy_rot_pos_emb_only": diff_stats(
                mode_outputs["baseline"], mode_outputs["legacy_rot_pos_emb_only"]
            ),
            "baseline_vs_full_disable_rope": diff_stats(
                mode_outputs["baseline"], mode_outputs["full_disable_rope"]
            ),
            "legacy_rot_pos_emb_only_vs_full_disable_rope": diff_stats(
                mode_outputs["legacy_rot_pos_emb_only"], mode_outputs["full_disable_rope"]
            ),
        }
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        write_report(report, args.output_path)
        print(f"Wrote diagnostics: {args.output_path}")


if __name__ == "__main__":
    main()
