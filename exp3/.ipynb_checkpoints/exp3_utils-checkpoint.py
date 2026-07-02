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

PROJECT_ROOT = Path("/Users/liuxiwei/Downloads/conference/AAAI27/Exp/whatsup")
EXP1_DIR = PROJECT_ROOT / "exp1"
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

from exp1_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    build_relation_prompt,
    load_exp1_samples,
    write_json,
)


DEFAULT_IMAGENET_ROOT = Path("/root/autodl-tmp/imagenet")
RELATION_LABELS = ("left_of", "right_of", "in_front_of", "behind")
RELATION_TO_ID = {name: idx for idx, name in enumerate(RELATION_LABELS)}


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


def average_layer_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
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
            item[key] = float(sum(values) / len(values)) if values else float("nan")
        output.append(item)
    return output


def merge_position_train_eval(train_rows: list[dict[str, float]], eval_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    eval_by_layer = {int(row["layer"]): row for row in eval_rows}
    output = []
    for row in train_rows:
        layer = int(row["layer"])
        eval_row = eval_by_layer.get(layer, {})
        output.append(
            {
                "layer": layer,
                "train_x_acc": row.get("x_acc"),
                "train_y_acc": row.get("y_acc"),
                "train_xy_acc": row.get("xy_acc"),
                "eval_x_acc": eval_row.get("x_acc"),
                "eval_y_acc": eval_row.get("y_acc"),
                "eval_xy_acc": eval_row.get("xy_acc"),
            }
        )
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
                "eval_relation_acc": eval_row.get("relation_acc"),
            }
        )
    return output


def save_probe_plot(rows: list[dict[str, Any]], output_path: Path, title: str, keys: list[str]) -> None:
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
        plt.plot(layers, values, marker="o", linewidth=1.5, label=key)
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
