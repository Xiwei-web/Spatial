#!/usr/bin/env python3
"""Shared utilities for Experiment 6 causal mediation analysis.

The experiment follows the source/base/patched setup from "Mechanisms of
Object Localization in VLMs" Sec. 3.3.2. The source image contains the target
object, the base image is the target-removed/inpainted image, and each patched
run copies one source attention-head activation into the base run.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP6_DIR = PROJECT_ROOT / "exp6"

DEFAULT_QA_PATH = PROJECT_ROOT / "whatsUP" / "qa.json"
DEFAULT_BBOX_MASK_XYXY_PATH = PROJECT_ROOT / "whatsUP" / "bbox_mask_xyxy.json"
DEFAULT_SOURCE_IMAGE_ROOT = Path("autodl-tmp/whatsUP/images")
DEFAULT_REMOVED_IMAGE_ROOT = Path("autodl-tmp/whatsUP_lama_504/output_dilate20")
DEFAULT_RANDOM_SEED = 20260616
DEFAULT_CURATED_SIZE = 50

RELATION_PHRASES = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "in_front_of": "in front of",
    "behind": "behind",
    "under": "under",
    "on": "on",
}

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp", ".bmp")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonical_relation_from_text(text: str) -> str | None:
    text_l = text.lower().replace("-", " ")
    if "left" in text_l:
        return "left_of"
    if "right" in text_l:
        return "right_of"
    if "in front" in text_l or "front of" in text_l:
        return "in_front_of"
    if "behind" in text_l:
        return "behind"
    if "under" in text_l or "below" in text_l:
        return "under"
    if " on " in f" {text_l} " or "on top" in text_l or "above" in text_l:
        return "on"
    return None


def canonical_relation_from_qa(qa_item: dict[str, Any]) -> str | None:
    answer = qa_item.get("answer")
    options = qa_item.get("options", {})
    if answer in options:
        relation = canonical_relation_from_text(options[answer])
        if relation:
            return relation

    image_stem = Path(qa_item.get("image_path", "")).stem
    target = qa_item.get("target", "")
    reference = qa_item.get("reference", "")
    prefix = f"{target}_"
    suffix = f"_{reference}"
    if image_stem.startswith(prefix) and image_stem.endswith(suffix):
        return image_stem[len(prefix) : -len(suffix)]
    return None


def relation_phrase(relation: str) -> str:
    if relation not in RELATION_PHRASES:
        raise ValueError(f"Unsupported relation {relation!r}. Known relations: {sorted(RELATION_PHRASES)}")
    return RELATION_PHRASES[relation]


def build_binary_relation_prompt(target: str, relation: str, reference: str) -> str:
    phrase = relation_phrase(relation)
    return f"Is the {target} {phrase} the {reference}? Answer yes or no."


def build_localization_prompt(object_name: str) -> str:
    return (
        f"Please provide the bounding box coordinates of the {object_name}.\n"
        "Format: [x1, y1, x2, y2]."
    )


def bbox_answer_text(box: list[float] | tuple[float, float, float, float]) -> str:
    vals = []
    for value in box:
        as_float = float(value)
        vals.append(str(int(round(as_float))) if abs(as_float - round(as_float)) < 1e-4 else f"{as_float:.2f}")
    return f" [{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}]"


def convert_qa_to_binary(qa_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    binary = []
    for item in qa_data:
        relation = canonical_relation_from_qa(item)
        if relation is None:
            continue
        new_item = dict(item)
        new_item["original_question"] = item.get("question")
        new_item["original_options"] = item.get("options")
        new_item["original_answer"] = item.get("answer")
        new_item["gt_relation"] = relation
        new_item["question"] = build_binary_relation_prompt(item["target"], relation, item["reference"])
        new_item["answer"] = "yes"
        new_item["answer_type"] = "binary_yes_no"
        binary.append(new_item)
    return binary


def write_binary_qa(qa_path: Path, output_path: Path) -> list[dict[str, Any]]:
    binary = convert_qa_to_binary(load_json(qa_path))
    write_json(output_path, binary)
    return binary


def image_stem_from_qa(item: dict[str, Any]) -> str:
    return Path(item["image_path"]).stem


def build_bbox_lookup(bbox_mask_path: Path | None) -> dict[str, dict[str, Any]]:
    if bbox_mask_path is None or not bbox_mask_path.is_file():
        return {}
    coco = load_json(bbox_mask_path)
    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    lookup = {}
    for image_id, anns in annotations_by_image.items():
        image_info = images_by_id.get(image_id)
        if not image_info:
            continue
        target_anns = [ann for ann in anns if int(ann.get("category_id", -1)) == 1]
        reference_anns = [ann for ann in anns if int(ann.get("category_id", -1)) == 2]
        if len(target_anns) != 1 or len(reference_anns) != 1:
            continue
        key = Path(image_info["file_name"]).stem
        lookup[key] = {
            "image_width": image_info.get("width"),
            "image_height": image_info.get("height"),
            "target_bbox": [float(v) for v in target_anns[0]["bbox"]],
            "reference_bbox": [float(v) for v in reference_anns[0]["bbox"]],
            "target_annotation_id": target_anns[0].get("id"),
            "reference_annotation_id": reference_anns[0].get("id"),
            "target_mask_path": target_anns[0].get("attributes", {}).get("mask_path"),
            "reference_mask_path": reference_anns[0].get("attributes", {}).get("mask_path"),
        }
    return lookup


def find_source_image(source_image_root: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = source_image_root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    matches = sorted(source_image_root.glob(f"{stem}.*"))
    return matches[0] if matches else None


def build_candidate_pairs(
    binary_qa: list[dict[str, Any]],
    source_image_root: Path,
    removed_image_root: Path,
    bbox_mask_path: Path | None = None,
    require_existing_files: bool = True,
) -> list[dict[str, Any]]:
    """Build source/base pairs by matching removed image stems to QA entries."""

    qa_by_stem = {image_stem_from_qa(item): item for item in binary_qa}
    bbox_lookup = build_bbox_lookup(bbox_mask_path)

    if require_existing_files:
        removed_paths = sorted(path for path in removed_image_root.iterdir() if path.is_file())
    else:
        removed_paths = [removed_image_root / f"{stem}.png" for stem in sorted(qa_by_stem)]

    pairs = []
    for removed_path in removed_paths:
        stem = removed_path.stem
        qa_item = qa_by_stem.get(stem)
        if qa_item is None:
            continue
        source_path = find_source_image(source_image_root, stem) if require_existing_files else source_image_root / Path(qa_item["image_path"]).name
        if source_path is None:
            continue
        qa_id = qa_item.get("id", len(pairs) + 1)
        source_dataset = qa_item.get("source_dataset", "unknown")
        sample = {
            "sample_id": f"whatsup_{source_dataset}_{int(qa_id):06d}",
            "qa_id": qa_id,
            "source_dataset": source_dataset,
            "image_stem": stem,
            "source_image_path": str(source_path),
            "base_image_path": str(removed_path),
            "removed_image_path": str(removed_path),
            "target_name": qa_item["target"],
            "reference_name": qa_item["reference"],
            "gt_relation": qa_item["gt_relation"],
            "binary_question": qa_item["question"],
            "binary_answer": "yes",
            "original_image_path": qa_item.get("image_path"),
            "original_question": qa_item.get("original_question"),
            "original_options": qa_item.get("original_options"),
            "original_answer": qa_item.get("original_answer"),
        }
        sample.update(bbox_lookup.get(stem, {}))
        pairs.append(sample)
    return pairs


def parse_yes_no(text: str) -> str | None:
    clean = text.strip().lower()
    clean = re.sub(r"^[^a-z]+", "", clean)
    if clean.startswith("yes"):
        return "yes"
    if clean.startswith("no"):
        return "no"
    if re.search(r"\byes\b", clean) and not re.search(r"\bno\b", clean):
        return "yes"
    if re.search(r"\bno\b", clean) and not re.search(r"\byes\b", clean):
        return "no"
    return None


def is_source_correct_base_wrong_or_degraded(record: dict[str, Any], min_ppl_gap: float) -> bool:
    source_correct = record.get("source_prediction") == "yes"
    base_wrong = record.get("base_prediction") != "yes"
    source_ppl = record.get("source_yes_ppl")
    base_ppl = record.get("base_yes_ppl")
    degraded = (
        isinstance(source_ppl, (int, float))
        and isinstance(base_ppl, (int, float))
        and math.isfinite(source_ppl)
        and math.isfinite(base_ppl)
        and (float(base_ppl) - float(source_ppl)) > float(min_ppl_gap)
    )
    return bool(source_correct and (base_wrong or degraded))


def select_curated_subset(
    behavior_records: list[dict[str, Any]],
    max_samples: int,
    seed: int,
    min_ppl_gap: float,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in behavior_records
        if is_source_correct_base_wrong_or_degraded(row, min_ppl_gap=min_ppl_gap)
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[: min(max_samples, len(eligible))]
    return [row["sample"] for row in selected]


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


def find_output_projection(attn: Any) -> Any:
    for name in ("o_proj", "out_proj", "dense", "c_proj", "proj"):
        module = getattr(attn, name, None)
        if module is not None and callable(module):
            return module
    candidates = []
    if hasattr(attn, "named_modules"):
        for name, module in attn.named_modules():
            last = name.rsplit(".", maxsplit=1)[-1].lower()
            if last in {"o_proj", "out_proj", "dense", "c_proj", "proj"} and callable(module):
                candidates.append((name, module))
    if len(candidates) == 1:
        return candidates[0][1]
    raise AttributeError(f"Could not locate attention output projection in {attn!r}")


def infer_num_heads(model: Any, attn: Any) -> int:
    for attr in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    config = getattr(model, "config", None)
    for path in (
        "text_config.num_attention_heads",
        "language_config.num_attention_heads",
        "llm_config.num_attention_heads",
        "num_attention_heads",
    ):
        value = get_attr_by_path(config, path) if config is not None else None
        if value is not None:
            return int(value)
    raise AttributeError("Could not infer number of attention heads")


def parse_int_csv(text: str | None, upper: int | None = None) -> list[int] | None:
    if text is None or text.strip().lower() in {"", "all"}:
        return None
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if upper is not None:
        for value in values:
            if value < 0 or value >= upper:
                raise ValueError(f"Index {value} out of range 0..{upper - 1}")
    return values


def normalize_batch_to_dict(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return dict(inputs)
    if hasattr(inputs, "data") and isinstance(inputs.data, dict):
        return dict(inputs.data)
    return dict(inputs)


def clone_inputs_without_labels(inputs: Any) -> dict[str, Any]:
    out = normalize_batch_to_dict(inputs)
    out.pop("labels", None)
    return out


def teacher_forced_perplexity(
    model: Any,
    inputs: Any,
    score_positions: list[int],
) -> float:
    import torch

    if not score_positions:
        return float("nan")
    model_inputs = clone_inputs_without_labels(inputs)
    model_inputs["use_cache"] = False
    input_ids = model_inputs["input_ids"]
    score_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    valid_positions = [pos for pos in score_positions if 0 <= pos < input_ids.shape[1]]
    if not valid_positions:
        return float("nan")
    score_mask[:, valid_positions] = True

    with torch.inference_mode():
        try:
            outputs = model(**model_inputs)
        except TypeError as exc:
            if "use_cache" not in str(exc):
                raise
            model_inputs.pop("use_cache", None)
            outputs = model(**model_inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = score_mask[:, 1:]
    if not bool(shift_mask.any()):
        return float("nan")

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    nll = -token_log_probs[shift_mask]
    mean_nll = nll.mean()
    return float(torch.exp(mean_nll).detach().cpu().item())


def score_positions_for_target(
    tokenizer: Any,
    full_input_ids: Any,
    prompt_len: int,
    task: str,
) -> list[int]:
    import torch

    if task not in {"relation", "localization"}:
        raise ValueError(f"Unsupported task: {task}")

    ids = full_input_ids[0].detach().cpu().tolist() if torch.is_tensor(full_input_ids) else list(full_input_ids[0])
    positions = list(range(prompt_len, len(ids)))
    if task == "relation":
        return positions

    digit_positions = []
    for pos in positions:
        piece = tokenizer.decode([ids[pos]], skip_special_tokens=False)
        if re.search(r"\d", piece):
            digit_positions.append(pos)
    return digit_positions or positions


def finite_gap(base_ppl: float, source_ppl: float, eps: float) -> bool:
    return (
        math.isfinite(base_ppl)
        and math.isfinite(source_ppl)
        and abs(float(base_ppl) - float(source_ppl)) >= float(eps)
    )


def mediation_fraction(source_ppl: float, base_ppl: float, patched_ppl: float, eps: float) -> float | None:
    if not finite_gap(base_ppl=base_ppl, source_ppl=source_ppl, eps=eps):
        return None
    return (float(base_ppl) - float(patched_ppl)) / (float(base_ppl) - float(source_ppl))


def make_task_payload(sample: dict[str, Any], task: str) -> tuple[str, str]:
    if task == "relation":
        return sample["binary_question"], " yes"
    if task == "localization":
        if "target_bbox" not in sample:
            raise KeyError(f"Sample {sample.get('sample_id')} has no target_bbox for localization CMA")
        return build_localization_prompt(sample["target_name"]), bbox_answer_text(sample["target_bbox"])
    raise ValueError(f"Unsupported task: {task}")


class HeadActivationStore:
    """Caches o_proj input activations for prompt token positions."""

    def __init__(self, prompt_positions: list[int]) -> None:
        self.prompt_positions = [int(pos) for pos in prompt_positions]
        self.cache: dict[int, Any] = {}

    def capture(self, layer_idx: int, tensor: Any) -> None:
        import torch

        if tensor.ndim != 3:
            raise ValueError(f"Expected o_proj input [B, S, D], got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != 1:
            raise ValueError("Experiment 6 currently expects batch size 1")
        positions = [pos for pos in self.prompt_positions if 0 <= pos < tensor.shape[1]]
        if not positions:
            raise ValueError("No valid prompt positions to cache")
        index = torch.as_tensor(positions, device=tensor.device, dtype=torch.long)
        self.cache[layer_idx] = tensor.index_select(dim=1, index=index).detach().clone()


@contextmanager
def cache_head_activations(model: Any, prompt_positions: list[int]):
    layers = find_decoder_layers(model)
    store = HeadActivationStore(prompt_positions)
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise RuntimeError("o_proj pre-hook received no input")
            store.capture(layer_idx, args[0])

        return hook

    for layer_idx, layer in enumerate(layers):
        attn = find_self_attention(layer)
        o_proj = find_output_projection(attn)
        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_single_head_activation(
    model: Any,
    layer_idx: int,
    head_idx: int,
    source_cache: dict[int, Any],
    prompt_positions: list[int],
):
    import torch

    layers = find_decoder_layers(model)
    layer = layers[layer_idx]
    attn = find_self_attention(layer)
    o_proj = find_output_projection(attn)
    num_heads = infer_num_heads(model, attn)
    if head_idx < 0 or head_idx >= num_heads:
        raise IndexError(f"Head {head_idx} out of range for layer {layer_idx} with {num_heads} heads")
    if layer_idx not in source_cache:
        raise KeyError(f"No source activation cached for layer {layer_idx}")

    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            raise RuntimeError("o_proj pre-hook received no input")
        tensor = args[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected o_proj input [B, S, D], got shape {tuple(tensor.shape)}")
        hidden_size = int(tensor.shape[-1])
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} is not divisible by num_heads={num_heads}")
        head_dim = hidden_size // num_heads
        start = head_idx * head_dim
        end = start + head_dim

        positions = [pos for pos in prompt_positions if 0 <= pos < tensor.shape[1]]
        if not positions:
            return args
        source = source_cache[layer_idx].to(device=tensor.device, dtype=tensor.dtype)
        if source.shape[1] != len(positions):
            raise ValueError(
                f"Cached source positions ({source.shape[1]}) do not match valid patch positions ({len(positions)})"
            )
        index = torch.as_tensor(positions, device=tensor.device, dtype=torch.long)
        patched = tensor.clone()
        patched[:, index, start:end] = source[:, :, start:end]
        return (patched,) + tuple(args[1:])

    handle = o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def summarize_cma_records(records: list[dict[str, Any]], num_layers: int, num_heads: int) -> dict[str, Any]:
    by_head: dict[tuple[str, int, int], list[float]] = {}
    valid_by_head: dict[tuple[str, int, int], int] = {}
    for row in records:
        task = row["task"]
        key = (task, int(row["layer"]), int(row["head"]))
        mf = row.get("mediation_fraction")
        if mf is not None and isinstance(mf, (int, float)) and math.isfinite(float(mf)):
            by_head.setdefault(key, []).append(float(mf))
        valid_by_head[key] = valid_by_head.get(key, 0) + int(row.get("valid_gap", False))

    tasks = sorted({row["task"] for row in records})
    out = {"num_layers": num_layers, "num_heads": num_heads, "tasks": {}}
    for task in tasks:
        matrix = []
        counts = []
        for layer in range(num_layers):
            matrix_row = []
            count_row = []
            for head in range(num_heads):
                values = by_head.get((task, layer, head), [])
                matrix_row.append(sum(values) / len(values) if values else None)
                count_row.append(valid_by_head.get((task, layer, head), 0))
            matrix.append(matrix_row)
            counts.append(count_row)
        out["tasks"][task] = {"mean_mediation_fraction": matrix, "valid_counts": counts}
    return out


def write_cma_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for task, task_data in summary.get("tasks", {}).items():
        matrix = task_data["mean_mediation_fraction"]
        counts = task_data["valid_counts"]
        for layer_idx, row in enumerate(matrix):
            for head_idx, mf in enumerate(row):
                rows.append(
                    {
                        "task": task,
                        "layer": layer_idx,
                        "head": head_idx,
                        "mean_mediation_fraction": mf,
                        "valid_count": counts[layer_idx][head_idx],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "layer", "head", "mean_mediation_fraction", "valid_count"])
        writer.writeheader()
        writer.writerows(rows)


def completed_behavior_ids(path: Path) -> set[str]:
    return {row["sample_id"] for row in load_jsonl(path) if row.get("sample_id")}


def completed_cma_keys(path: Path) -> set[str]:
    return {row["cma_key"] for row in load_jsonl(path) if row.get("cma_key")}


def cma_record_key(sample_id: str, task: str, layer: int, head: int) -> str:
    return f"{sample_id}::{task}::L{layer:03d}::H{head:03d}"


def parse_task_list(text: str) -> list[str]:
    tasks = [item.strip() for item in text.split(",") if item.strip()]
    for task in tasks:
        if task not in {"relation", "localization"}:
            raise ValueError("Tasks must be comma-separated values from relation,localization")
    return tasks


def path_to_str_dict(args: Any) -> dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output
