#!/usr/bin/env python3
"""
Visualize the geometry of LLaVA-1.5-7B 1D absolute position embeddings.

This reproduces the LLaVA part of Figure 5 in:
"Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models".

The paper's procedure:
  1. Take the learnable 1D absolute position embedding from the ViT vision tower.
  2. LLaVA-1.5-7B has 577 position embeddings: 1 CLS token + 24*24 image patch tokens.
  3. Remove the CLS token.
  4. Run t-SNE from the original embedding dimension to 2D.
  5. Plot the 576 points twice, colored by row id and column id.

No image dataset is required. Only the local model weights are needed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
from sklearn.manifold import TSNE


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_MODEL_DIR = DEFAULT_PROJECT_ROOT / "models/llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "exp1" / "outputs"

POSITION_EMBEDDING_KEY_CANDIDATES = (
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "model.vision_tower.vision_model.embeddings.position_embedding.weight",
    "vision_model.embeddings.position_embedding.weight",
    "clip_vision_model.vision_model.embeddings.position_embedding.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw t-SNE geometry of LLaVA-1.5 1D absolute position embeddings."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="llava_1d_abs_pe_tsne.png")
    parser.add_argument("--embedding-key", default=None)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Patch grid size. If omitted, inferred from num_position_embeddings - 1.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw subplot titles.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the 2D t-SNE coordinates as .npy.",
    )
    return parser.parse_args()


def list_safetensor_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return files


def keys_in_safetensor(path: Path) -> list[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return list(f.keys())


def find_key_in_index(model_dir: Path, requested_key: str | None) -> tuple[str, Path] | None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    candidates: Iterable[str]
    if requested_key:
        candidates = (requested_key,)
    else:
        candidates = POSITION_EMBEDDING_KEY_CANDIDATES

    for key in candidates:
        if key in weight_map:
            return key, model_dir / weight_map[key]
    return None


def find_key_by_scanning(model_dir: Path, requested_key: str | None) -> tuple[str, Path]:
    files = list_safetensor_files(model_dir)

    if requested_key:
        for path in files:
            if requested_key in keys_in_safetensor(path):
                return requested_key, path
        raise KeyError(f"Requested key not found in safetensors: {requested_key}")

    found = []
    for path in files:
        keys = keys_in_safetensor(path)
        for key in keys:
            if key in POSITION_EMBEDDING_KEY_CANDIDATES:
                found.append((key, path))
            elif key.endswith("vision_model.embeddings.position_embedding.weight"):
                found.append((key, path))
            elif key.endswith("embeddings.position_embedding.weight") and "vision" in key:
                found.append((key, path))

    if not found:
        all_keys_preview = []
        for path in files[:2]:
            all_keys_preview.extend(keys_in_safetensor(path)[:20])
        raise KeyError(
            "Could not find a vision position embedding key. "
            f"Try --embedding-key explicitly. Key preview: {all_keys_preview}"
        )
    return found[0]


def load_position_embedding(model_dir: Path, requested_key: str | None = None) -> tuple[np.ndarray, str]:
    located = find_key_in_index(model_dir, requested_key)
    if located is None:
        located = find_key_by_scanning(model_dir, requested_key)

    key, shard_path = located
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard for {key} does not exist: {shard_path}")

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key).float().cpu().numpy()

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D position embedding tensor, got shape {tensor.shape}")
    return tensor, key


def infer_grid_size(num_positions: int, grid_size: int | None) -> int:
    if grid_size is not None:
        expected = 1 + grid_size * grid_size
        if num_positions != expected:
            raise ValueError(
                f"grid_size={grid_size} expects 1 + grid^2 = {expected} positions, "
                f"but embedding has {num_positions}."
            )
        return grid_size

    patch_positions = num_positions - 1
    inferred = int(math.isqrt(patch_positions))
    if inferred * inferred != patch_positions:
        raise ValueError(
            f"Cannot infer square patch grid from {num_positions} positions. "
            "Pass --grid-size explicitly."
        )
    return inferred


def run_tsne(embeddings: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": random_state,
        "init": "pca",
        "learning_rate": "auto",
    }

    signature = inspect.signature(TSNE)
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000

    return TSNE(**kwargs).fit_transform(embeddings)


def make_row_col_labels(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    row_ids = np.repeat(np.arange(grid_size), grid_size)
    col_ids = np.tile(np.arange(grid_size), grid_size)
    return row_ids, col_ids


def plot_tsne(
    coords: np.ndarray,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    grid_size: int,
    output_path: Path,
    point_size: float,
    dpi: int,
    no_title: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    panels = [
        (axes[0], row_ids, "(a) PE in LLaVA (row)", "row ID"),
        (axes[1], col_ids, "(b) PE in LLaVA (col)", "col ID"),
    ]

    for ax, labels, title, cbar_label in panels:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap="viridis",
            s=point_size,
            alpha=0.95,
            linewidths=0,
        )
        if not no_title:
            ax.set_title(title, fontsize=12)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.tick_params(labelsize=8)
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(cbar_label)
        colorbar.set_ticks(np.arange(0, grid_size, max(1, grid_size // 8)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    position_embedding, key = load_position_embedding(args.model_dir, args.embedding_key)
    grid_size = infer_grid_size(position_embedding.shape[0], args.grid_size)

    patch_position_embedding = position_embedding[1:]
    expected_num_patches = grid_size * grid_size
    if patch_position_embedding.shape[0] != expected_num_patches:
        raise ValueError(
            f"After removing CLS, expected {expected_num_patches} patch embeddings, "
            f"got {patch_position_embedding.shape[0]}"
        )

    coords = run_tsne(
        patch_position_embedding,
        perplexity=args.perplexity,
        random_state=args.random_state,
    )
    row_ids, col_ids = make_row_col_labels(grid_size)

    output_path = args.output_dir / args.output_name
    plot_tsne(
        coords=coords,
        row_ids=row_ids,
        col_ids=col_ids,
        grid_size=grid_size,
        output_path=output_path,
        point_size=args.point_size,
        dpi=args.dpi,
        no_title=args.no_title,
    )

    if args.save_npy:
        np.save(args.output_dir / "llava_1d_abs_pe_tsne_coords.npy", coords)

    print(f"Loaded position embedding key: {key}")
    print(f"Position embedding shape: {position_embedding.shape}")
    print(f"Patch grid: {grid_size}x{grid_size}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
