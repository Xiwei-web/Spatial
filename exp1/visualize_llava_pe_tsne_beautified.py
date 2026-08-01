#!/usr/bin/env python3
"""
Beautified t-SNE visualization for LLaVA-1.5-7B 1D absolute position embeddings.

Improvements over the original version:
  - Larger, cleaner scatter markers.
  - Optional marker edge stroke for readability.
  - Multiple colormap variants generated in one run.
  - Shared, consistent axes across the row/col panels.
  - Higher-resolution export with transparent/white-background options.
  - Optional coordinate caching so you can redraw palettes without rerunning t-SNE.
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


# Resolve paths from the repository, independent of the shell working directory.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = DEFAULT_PROJECT_ROOT / "models/llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "exp1" / "outputs"

POSITION_EMBEDDING_KEY_CANDIDATES = (
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "model.vision_tower.vision_model.embeddings.position_embedding.weight",
    "vision_model.embeddings.position_embedding.weight",
    "clip_vision_model.vision_model.embeddings.position_embedding.weight",
)

# Good perceptual choices for continuous row/column indices.
# DEFAULT_CMAPS = ("viridis", "plasma", "turbo", "magma", "cividis")
DEFAULT_CMAPS = ("magma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw beautified t-SNE geometry of LLaVA-1.5 1D absolute position embeddings."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="llava_1d_abs_pe_tsne")
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
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--alpha", type=float, default=0.92)
    parser.add_argument("--edge-width", type=float, default=0.15)
    parser.add_argument("--edge-color", default="white")
    parser.add_argument(
        "--cmaps",
        nargs="+",
        default=list(DEFAULT_CMAPS),
        help="Matplotlib colormaps to render. Example: --cmaps viridis plasma turbo",
    )
    parser.add_argument(
        "--figure-scale",
        type=float,
        default=1.0,
        help="Scale the default figure size. Use 1.15 or 1.25 for paper figures.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw subplot titles.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save transparent PNGs instead of white-background PNGs.",
    )
    parser.add_argument(
        "--coords-npy",
        type=Path,
        default=None,
        help="Optional existing t-SNE .npy coordinates to reuse instead of recomputing t-SNE.",
    )
    parser.add_argument(
        "--save-coords",
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


def beautify_axes(ax: plt.Axes) -> None:
    ax.grid(True, linewidth=0.45, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.75)
    ax.tick_params(labelsize=9, width=0.8, length=3.5)
    ax.set_aspect("equal", adjustable="datalim")


def plot_tsne(
    coords: np.ndarray,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    grid_size: int,
    output_path: Path,
    cmap: str,
    point_size: float,
    alpha: float,
    edge_width: float,
    edge_color: str,
    dpi: int,
    no_title: bool,
    transparent: bool,
    figure_scale: float,
) -> None:
    width = 10.8 * figure_scale
    height = 4.7 * figure_scale
    fig, axes = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)

    panels = [
        (axes[0], row_ids, "(a) PE in LLaVA (row)", "row ID"),
        (axes[1], col_ids, "(b) PE in LLaVA (col)", "col ID"),
    ]

    x_pad = 0.05 * (coords[:, 0].max() - coords[:, 0].min())
    y_pad = 0.05 * (coords[:, 1].max() - coords[:, 1].min())
    xlim = (coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ylim = (coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)

    for ax, labels, title, cbar_label in panels:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            linewidths=edge_width,
            edgecolors=edge_color if edge_width > 0 else "none",
        )
        if not no_title:
            ax.set_title(title, fontsize=14, pad=8)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        beautify_axes(ax)

        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label(cbar_label, fontsize=11)
        colorbar.ax.tick_params(labelsize=9)
        colorbar.set_ticks(np.arange(0, grid_size, max(1, grid_size // 8)))
        colorbar.outline.set_linewidth(0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="none" if transparent else "white",
        transparent=transparent,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.coords_npy is not None:
        coords = np.load(args.coords_npy)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected coords shape [N, 2], got {coords.shape}")
        patch_count = coords.shape[0]
        grid_size = args.grid_size or int(math.isqrt(patch_count))
        if grid_size * grid_size != patch_count:
            raise ValueError("When using --coords-npy, pass --grid-size if N is not a square.")
        key = "reused from --coords-npy"
        position_embedding_shape = f"not loaded; coords={coords.shape}"
    else:
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
        position_embedding_shape = str(position_embedding.shape)

    row_ids, col_ids = make_row_col_labels(grid_size)

    saved_paths = []
    for cmap in args.cmaps:
        output_path = args.output_dir / f"{args.output_prefix}_{cmap}.png"
        plot_tsne(
            coords=coords,
            row_ids=row_ids,
            col_ids=col_ids,
            grid_size=grid_size,
            output_path=output_path,
            cmap=cmap,
            point_size=args.point_size,
            alpha=args.alpha,
            edge_width=args.edge_width,
            edge_color=args.edge_color,
            dpi=args.dpi,
            no_title=args.no_title,
            transparent=args.transparent,
            figure_scale=args.figure_scale,
        )
        saved_paths.append(output_path)

    if args.save_coords:
        coords_path = args.output_dir / f"{args.output_prefix}_coords.npy"
        np.save(coords_path, coords)
        saved_paths.append(coords_path)

    print(f"Loaded position embedding key: {key}")
    print(f"Position embedding shape: {position_embedding_shape}")
    print(f"Patch grid: {grid_size}x{grid_size}")
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
