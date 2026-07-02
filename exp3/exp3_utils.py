#!/usr/bin/env python3
"""Shared utilities for Experiment 3 position and relation probes."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
EXP1_DIR = PROJECT_ROOT / "exp1"
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

from exp1_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_relation_prompt,
    load_exp1_samples,
    write_json,
)


DEFAULT_IMAGENET_ROOT = Path("imagenet")
RELATION_LABELS = ("left_of", "right_of", "in_front_of", "behind")
RELATION_TO_ID = {name: idx for idx, name in enumerate(RELATION_LABELS)}
OPTION_LABELS = ("A", "B", "C", "D")
OPTION_TO_ID = {name: idx for idx, name in enumerate(OPTION_LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_imagenet_val_images(imagenet_root: Path, start_index: int = 0, max_images: int | None = None) -> list[str]:
    """Return ImageNet validation JPEG paths.

    The intended HPC layout is:
    /root/autodl-tmp/imagenet/val/ILSVRC2012_val_00000001.JPEG
    """

    val_dir = imagenet_root / "val" if (imagenet_root / "val").is_dir() else imagenet_root
    images = sorted(str(path) for path in val_dir.glob("*.JPEG"))
    if not images:
        raise FileNotFoundError(
            f"No ImageNet validation JPEGs found under {val_dir}. "
            "Expected files like ILSVRC2012_val_00000001.JPEG."
        )
    images = images[start_index:]
    if max_images is not None:
        images = images[:max_images]
    return images


def load_relation_samples(
    data_root: Path = DEFAULT_DATA_ROOT,
    start_index: int = 0,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    samples = []
    for sample in load_exp1_samples(data_root=data_root):
        relation = sample.get("gt_relation")
        if relation not in RELATION_TO_ID:
            continue
        sample = dict(sample)
        sample["relation_label"] = relation
        sample["relation_id"] = RELATION_TO_ID[relation]
        sample["relation_prompt"] = build_relation_prompt(sample)
        samples.append(sample)
    samples = samples[start_index:]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def split_train_eval(
    items: list[Any],
    train_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--relation-train-fraction must be between 0 and 1")
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    split = max(1, min(len(indices) - 1, int(round(len(indices) * train_fraction))))
    train_idx = set(indices[:split])
    train_items = [item for idx, item in enumerate(items) if idx in train_idx]
    eval_items = [item for idx, item in enumerate(items) if idx not in train_idx]
    return train_items, eval_items


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def make_position_labels(grid_hw: tuple[int, int], device: Any) -> tuple[Any, Any]:
    import torch

    grid_h, grid_w = grid_hw
    xs = torch.arange(grid_w, dtype=torch.long, device=device).repeat(grid_h)
    ys = torch.arange(grid_h, dtype=torch.long, device=device).repeat_interleave(grid_w)
    return xs, ys


def image_token_positions_from_input_ids(
    input_ids: Any,
    image_token_id: int,
    expected_tokens: int,
) -> list[Any]:
    import torch

    positions = []
    for row in input_ids:
        pos = torch.nonzero(row == image_token_id, as_tuple=False).flatten()
        if int(pos.numel()) != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} image tokens for image_token_id={image_token_id}, "
                f"found {int(pos.numel())}. Check processor image size and token id."
            )
        positions.append(pos)
    return positions


def last_nonpad_positions(attention_mask: Any) -> Any:
    import torch

    if attention_mask is None:
        raise ValueError("attention_mask is required to identify the final prompt token")
    return attention_mask.long().sum(dim=1).sub(1).clamp_min(0).to(torch.long)


def select_decoder_hidden_states(outputs: Any, expected_layers: int | None = None, include_embedding_layer: bool = False) -> list[Any]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and hasattr(outputs, "language_model_outputs"):
        hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
    if hidden_states is None:
        raise AttributeError("Model output did not include decoder hidden_states")

    states = list(hidden_states)
    if not include_embedding_layer:
        if expected_layers is None or len(states) == expected_layers + 1:
            states = states[1:]
    if expected_layers is not None and len(states) != expected_layers:
        print(f"[warn] expected {expected_layers} decoder layers, got {len(states)} hidden-state entries")
    return states


def gather_image_hidden(layer_hidden: Any, image_positions: list[Any]) -> Any:
    import torch

    gathered = [layer_hidden[batch_idx, pos.to(layer_hidden.device), :] for batch_idx, pos in enumerate(image_positions)]
    return torch.stack(gathered, dim=0)


def gather_final_token_hidden(layer_hidden: Any, final_positions: Any) -> Any:
    import torch

    batch_indices = torch.arange(layer_hidden.shape[0], device=layer_hidden.device)
    final_positions = final_positions.to(layer_hidden.device)
    return layer_hidden[batch_indices, final_positions, :]


class PositionProbeBank:
    def __init__(self, num_layers: int, hidden_size: int, grid_hw: tuple[int, int], device: Any):
        import torch

        self.torch = torch
        self.grid_h, self.grid_w = grid_hw
        self.x_probes = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, self.grid_w) for _ in range(num_layers)]
        ).to(device)
        self.y_probes = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, self.grid_h) for _ in range(num_layers)]
        ).to(device)

    def parameters(self) -> Any:
        return list(self.x_probes.parameters()) + list(self.y_probes.parameters())

    def train(self) -> None:
        self.x_probes.train()
        self.y_probes.train()

    def eval(self) -> None:
        self.x_probes.eval()
        self.y_probes.eval()

    def state_dict(self) -> dict[str, Any]:
        return {"x_probes": self.x_probes.state_dict(), "y_probes": self.y_probes.state_dict()}


class RelationProbeBank:
    def __init__(self, num_layers: int, hidden_size: int, num_relations: int, device: Any):
        import torch

        self.torch = torch
        self.probes = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, num_relations) for _ in range(num_layers)]
        ).to(device)

    def parameters(self) -> Any:
        return self.probes.parameters()

    def train(self) -> None:
        self.probes.train()

    def eval(self) -> None:
        self.probes.eval()

    def state_dict(self) -> dict[str, Any]:
        return {"relation_probes": self.probes.state_dict()}


def get_attr_by_path(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_lm_head(model: Any) -> Any:
    for path in (
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "base_model.lm_head",
        "base_model.model.lm_head",
    ):
        module = get_attr_by_path(model, path)
        if module is not None:
            return module
    raise AttributeError("Could not locate lm_head for layer-wise logit lens")


def find_final_norm(model: Any) -> Any | None:
    for path in (
        "model.norm",
        "language_model.model.norm",
        "language_model.norm",
        "base_model.model.norm",
        "base_model.language_model.model.norm",
        "norm",
    ):
        module = get_attr_by_path(model, path)
        if module is not None:
            return module
    return None


def module_device(module: Any, fallback: Any) -> Any:
    import torch

    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)

    try:
        device = next(module.parameters()).device
        if getattr(device, "type", None) == "meta":
            return fallback
        return device
    except StopIteration:
        return fallback


def module_dtype(module: Any, fallback: Any) -> Any:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


def build_option_token_id_map(tokenizer: Any, option_labels: tuple[str, ...] = OPTION_LABELS) -> dict[str, list[int]]:
    """Map each option letter to plausible single-token next-token ids."""

    mapping: dict[str, list[int]] = {}
    for label in option_labels:
        ids: set[int] = set()
        for variant in (label, f" {label}", f"\n{label}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))
        if not ids:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"Could not tokenize option label {label!r}")
            ids.add(int(token_ids[-1]))
        mapping[label] = sorted(ids)
    return mapping


def _maybe_apply_final_norm(hidden: Any, final_norm: Any | None, layer_idx: int, num_layers: int, norm_mode: str) -> Any:
    if final_norm is None or norm_mode == "none":
        return hidden
    if norm_mode == "all_but_last" and layer_idx == num_layers - 1:
        return hidden
    if norm_mode not in {"all", "all_but_last", "none"}:
        raise ValueError(f"Unsupported norm_mode: {norm_mode}")
    norm_device = module_device(final_norm, hidden.device)
    norm_dtype = module_dtype(final_norm, hidden.dtype)
    return final_norm(hidden.to(device=norm_device, dtype=norm_dtype))


def answer_logit_lens_batch_rows(
    model: Any,
    layer_states: list[Any],
    final_positions: Any,
    labels: Any,
    option_token_ids: dict[str, list[int]],
    norm_mode: str = "all_but_last",
) -> list[dict[str, float]]:
    """Score A/B/C/D directly from each layer's final-token hidden state.

    This is a no-training early-exit/logit-lens evaluation.
    """

    import torch

    lm_head = find_lm_head(model)
    final_norm = find_final_norm(model)
    lm_head_device = module_device(lm_head, final_positions.device)
    lm_head_dtype = module_dtype(lm_head, torch.float32)
    labels = labels.to(lm_head_device)
    rows = []
    num_layers = len(layer_states)

    for layer_idx, hidden in enumerate(layer_states):
        final_hidden = gather_final_token_hidden(hidden, final_positions).detach()
        final_hidden = _maybe_apply_final_norm(final_hidden, final_norm, layer_idx, num_layers, norm_mode)
        final_hidden = final_hidden.to(device=lm_head_device, dtype=lm_head_dtype)
        logits = lm_head(final_hidden)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        option_log_probs = []
        for label in OPTION_LABELS:
            ids = torch.tensor(option_token_ids[label], dtype=torch.long, device=log_probs.device)
            option_log_probs.append(torch.logsumexp(log_probs.index_select(dim=-1, index=ids), dim=-1))
        option_log_probs_t = torch.stack(option_log_probs, dim=-1)
        option_probs_t = option_log_probs_t.exp()

        pred = option_log_probs_t.argmax(dim=-1)
        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        correct_logprob = option_log_probs_t[batch_indices, labels]
        correct_prob = option_probs_t[batch_indices, labels]
        top2 = torch.topk(option_log_probs_t, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        rows.append(
            {
                "layer": layer_idx,
                "answer_acc": (pred == labels).float().mean().item(),
                "correct_option_prob": correct_prob.mean().item(),
                "correct_option_logprob": correct_logprob.mean().item(),
                "option_logprob_margin": margin.mean().item(),
            }
        )
    return rows


def infer_hidden_size_from_states(layer_states: list[Any]) -> int:
    if not layer_states:
        raise ValueError("No hidden states found")
    return int(layer_states[0].shape[-1])


def position_probe_loss_and_stats(
    probes: PositionProbeBank,
    layer_states: list[Any],
    image_positions: list[Any],
    x_labels: Any,
    y_labels: Any,
) -> tuple[Any, list[dict[str, float]]]:
    import torch
    import torch.nn.functional as F

    total_loss = None
    rows = []
    for layer_idx, hidden in enumerate(layer_states):
        probe_device = probes.x_probes[layer_idx].weight.device
        image_hidden = gather_image_hidden(hidden, image_positions).detach().to(device=probe_device, dtype=torch.float32)
        batch_size, num_tokens, hidden_size = image_hidden.shape
        flat = image_hidden.reshape(batch_size * num_tokens, hidden_size)
        x_target = x_labels.to(probe_device).repeat(batch_size)
        y_target = y_labels.to(probe_device).repeat(batch_size)
        x_logits = probes.x_probes[layer_idx](flat)
        y_logits = probes.y_probes[layer_idx](flat)
        loss = F.cross_entropy(x_logits, x_target) + F.cross_entropy(y_logits, y_target)
        total_loss = loss if total_loss is None else total_loss + loss

        with torch.no_grad():
            x_pred = x_logits.argmax(dim=-1)
            y_pred = y_logits.argmax(dim=-1)
            x_acc = (x_pred == x_target).float().mean().item()
            y_acc = (y_pred == y_target).float().mean().item()
            xy_acc = ((x_pred == x_target) & (y_pred == y_target)).float().mean().item()
        rows.append({"layer": layer_idx, "x_acc": x_acc, "y_acc": y_acc, "xy_acc": xy_acc})
    if total_loss is None:
        raise ValueError("No layers were available for position probe loss")
    return total_loss / len(layer_states), rows


def relation_probe_loss_and_stats(
    probes: RelationProbeBank,
    layer_states: list[Any],
    final_positions: Any,
    labels: Any,
) -> tuple[Any, list[dict[str, float]]]:
    import torch
    import torch.nn.functional as F

    total_loss = None
    rows = []
    for layer_idx, hidden in enumerate(layer_states):
        probe_device = probes.probes[layer_idx].weight.device
        final_hidden = gather_final_token_hidden(hidden, final_positions).detach().to(device=probe_device, dtype=torch.float32)
        labels = labels.to(probe_device)
        logits = probes.probes[layer_idx](final_hidden)
        loss = F.cross_entropy(logits, labels)
        total_loss = loss if total_loss is None else total_loss + loss
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
        rows.append({"layer": layer_idx, "relation_acc": acc})
    if total_loss is None:
        raise ValueError("No layers were available for relation probe loss")
    return total_loss / len(layer_states), rows


def _mean_std_sem(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    count = len(values)
    mean = sum(values) / count
    if count <= 1:
        return float(mean), 0.0, 0.0, count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    std = variance ** 0.5
    sem = std / (count ** 0.5)
    return float(mean), float(std), float(sem), count


def average_layer_rows(rows: list[dict[str, float]], include_uncertainty: bool = True) -> list[dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        layer = int(row["layer"])
        for key, value in row.items():
            if key != "layer":
                grouped[layer][key].append(float(value))
    output = []
    for layer in sorted(grouped):
        item: dict[str, float] = {"layer": layer}
        for key, values in grouped[layer].items():
            mean, std, sem, count = _mean_std_sem(values)
            item[key] = mean
            if include_uncertainty:
                item[f"{key}_std"] = std
                item[f"{key}_sem"] = sem
                item[f"{key}_n"] = count
        output.append(item)
    return output


def merge_position_train_eval(train_rows: list[dict[str, float]], eval_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    eval_by_layer = {int(row["layer"]): row for row in eval_rows}
    output = []
    for row in train_rows:
        layer = int(row["layer"])
        eval_row = eval_by_layer.get(layer, {})
        merged: dict[str, Any] = {"layer": layer}
        for phase, source in (("train", row), ("eval", eval_row)):
            for key in ("x_acc", "y_acc", "xy_acc"):
                out_key = f"{phase}_{key}"
                merged[out_key] = source.get(key)
                merged[f"{out_key}_std"] = source.get(f"{key}_std")
                merged[f"{out_key}_sem"] = source.get(f"{key}_sem")
                merged[f"{out_key}_n"] = source.get(f"{key}_n")
        output.append(merged)
    return output


def merge_relation_train_eval(train_rows: list[dict[str, float]], eval_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    eval_by_layer = {int(row["layer"]): row for row in eval_rows}
    output = []
    for row in train_rows:
        layer = int(row["layer"])
        eval_row = eval_by_layer.get(layer, {})
        output.append(
            {
                "layer": layer,
                "train_relation_acc": row.get("relation_acc"),
                "train_relation_acc_std": row.get("relation_acc_std"),
                "train_relation_acc_sem": row.get("relation_acc_sem"),
                "train_relation_acc_n": row.get("relation_acc_n"),
                "eval_relation_acc": eval_row.get("relation_acc"),
                "eval_relation_acc_std": eval_row.get("relation_acc_std"),
                "eval_relation_acc_sem": eval_row.get("relation_acc_sem"),
                "eval_relation_acc_n": eval_row.get("relation_acc_n"),
            }
        )
    return output


def format_answer_logit_lens_rows(eval_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    output = []
    for row in eval_rows:
        item: dict[str, Any] = {"layer": int(row["layer"])}
        for key in ("answer_acc", "correct_option_prob", "correct_option_logprob", "option_logprob_margin"):
            item[key] = row.get(key)
            item[f"{key}_std"] = row.get(f"{key}_std")
            item[f"{key}_sem"] = row.get(f"{key}_sem")
            item[f"{key}_n"] = row.get(f"{key}_n")
        output.append(item)
    return output


def save_probe_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    keys: list[str],
    error_suffix: str | None = "sem",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping plot {output_path}: {exc}")
        return

    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [row["layer"] for row in rows]
    plt.figure(figsize=(8, 4.8))
    for key in keys:
        values = [row.get(key) for row in rows]
        yerr = None
        if error_suffix:
            err_key = f"{key}_{error_suffix}"
            errors = [row.get(err_key) for row in rows]
            if any(error is not None for error in errors):
                yerr = [0.0 if error is None else error for error in errors]
        if yerr is None:
            plt.plot(layers, values, marker="o", linewidth=1.5, label=key)
        else:
            plt.errorbar(layers, values, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=key)
    plt.xlabel("LLM decoder layer")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.02)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def args_to_jsonable(args: Any) -> dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output
