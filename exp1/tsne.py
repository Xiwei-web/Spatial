#!/usr/bin/env python3
"""
Combined visualization:

Panel 1-2:
  LLaVA-1.5-7B 1D absolute position embedding t-SNE:
    (a) row ID
    (b) col ID

Panel 3-4:
  Qwen2.5-VL direction-vector PCA:
    (c) With 2D RoPE
    (d) Without 2D RoPE

The final figure arranges all 4 panels in one row with:
  - equal axes box size
  - uniform spacing
  - consistent font sizes
  - per-panel colorbars
  - rotated 90-degree Qwen colorbar tick labels
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from sklearn.manifold import TSNE


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_DIR = PROJECT_ROOT / "exp1"

if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

from exp1_utils import DEFAULT_DATA_ROOT, load_exp1_samples  # noqa: E402


# ============================================================
# Defaults
# ============================================================

DEFAULT_LLAVA_MODEL_DIR = PROJECT_ROOT / "models/llava-1.5-7b-hf"
DEFAULT_QWEN_MODEL_DIR = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp1" / "outputs"

POSITION_EMBEDDING_KEY_CANDIDATES = (
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "model.vision_tower.vision_model.embeddings.position_embedding.weight",
    "vision_model.embeddings.position_embedding.weight",
    "clip_vision_model.vision_model.embeddings.position_embedding.weight",
)

RELATION_ORDER = ["left_of", "right_of", "behind", "in_front_of"]
RELATION_DISPLAY = {
    "left_of": "Left",
    "right_of": "Right",
    "behind": "Behind",
    "in_front_of": "In front of",
}


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw combined 4-panel LLaVA PE t-SNE and Qwen direction-vector PCA figure."
    )

    # Common
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="combined_llava_qwen_four_panels.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--transparent", action="store_true")

    # LLaVA t-SNE
    parser.add_argument("--llava-model-dir", type=Path, default=DEFAULT_LLAVA_MODEL_DIR)
    parser.add_argument("--embedding-key", default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-random-state", type=int, default=42)
    parser.add_argument("--tsne-alpha", type=float, default=0.92)
    parser.add_argument("--edge-width", type=float, default=0.15)
    parser.add_argument("--edge-color", default="white")
    parser.add_argument("--coords-npy", type=Path, default=None)
    parser.add_argument("--save-tsne-coords", action="store_true")

    # Qwen PCA
    parser.add_argument("--qwen-model-dir", type=Path, default=DEFAULT_QWEN_MODEL_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-dataset", default="B")
    parser.add_argument("--pair-target", default=None)
    parser.add_argument("--pair-reference", default=None)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--list-complete-pairs", action="store_true")
    parser.add_argument("--samples-per-relation", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--object-token-source",
        choices=["mask", "bbox"],
        default="mask",
    )
    parser.add_argument(
        "--pca-normalization",
        choices=["none", "center", "zscore", "l2", "center_l2"],
        default="zscore",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--resized-height", type=int, default=None)
    parser.add_argument("--resized-width", type=int, default=None)
    parser.add_argument("--save-npz", action="store_true")

    # Layout
    parser.add_argument("--fig-width", type=float, default=19.5)
    parser.add_argument("--fig-height", type=float, default=4.45)
    parser.add_argument("--wspace", type=float, default=0.16)

    return parser.parse_args()


# ============================================================
# LLaVA t-SNE functions
# ============================================================

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

    if requested_key:
        candidates: Iterable[str] = (requested_key,)
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
        raise KeyError("Could not find a vision position embedding key.")

    return found[0]


def load_position_embedding(model_dir: Path, requested_key: str | None = None) -> tuple[np.ndarray, str]:
    located = find_key_in_index(model_dir, requested_key)
    if located is None:
        located = find_key_by_scanning(model_dir, requested_key)

    key, shard_path = located

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key).float().cpu().numpy()

    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D position embedding tensor, got {tensor.shape}")

    return tensor, key


def infer_grid_size(num_positions: int, grid_size: int | None) -> int:
    if grid_size is not None:
        expected = 1 + grid_size * grid_size
        if num_positions != expected:
            raise ValueError(
                f"grid_size={grid_size} expects {expected} positions, got {num_positions}."
            )
        return grid_size

    patch_positions = num_positions - 1
    inferred = int(math.isqrt(patch_positions))

    if inferred * inferred != patch_positions:
        raise ValueError(
            f"Cannot infer square patch grid from {num_positions} positions."
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


def compute_llava_tsne(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
        position_embedding, key = load_position_embedding(
            args.llava_model_dir,
            args.embedding_key,
        )

        grid_size = infer_grid_size(position_embedding.shape[0], args.grid_size)
        patch_position_embedding = position_embedding[1:]

        expected_num_patches = grid_size * grid_size
        if patch_position_embedding.shape[0] != expected_num_patches:
            raise ValueError(
                f"Expected {expected_num_patches} patch embeddings, "
                f"got {patch_position_embedding.shape[0]}"
            )

        coords = run_tsne(
            patch_position_embedding,
            perplexity=args.perplexity,
            random_state=args.tsne_random_state,
        )

        position_embedding_shape = str(position_embedding.shape)

    row_ids, col_ids = make_row_col_labels(grid_size)

    print(f"Loaded LLaVA position embedding key: {key}")
    print(f"LLaVA position embedding shape: {position_embedding_shape}")
    print(f"LLaVA patch grid: {grid_size}x{grid_size}")

    if args.save_tsne_coords:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        coords_path = args.output_dir / "llava_1d_abs_pe_tsne_coords.npy"
        np.save(coords_path, coords)
        print(f"Saved t-SNE coords: {coords_path}")

    return coords, row_ids, col_ids, grid_size


# ============================================================
# Qwen model helpers
# ============================================================

def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
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
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.qwen_model_dir,
        local_files_only=True,
    )

    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
    }

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_dir,
        **model_kwargs,
    )

    model.eval()
    torch.set_grad_enabled(False)

    return model, processor


def get_visual_module(model: Any) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen visual module.")


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def get_spatial_merge_size(model: Any) -> int:
    visual = get_visual_module(model)

    if hasattr(visual, "spatial_merge_size"):
        return int(visual.spatial_merge_size)

    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "spatial_merge_size"):
        return int(vision_config.spatial_merge_size)

    return 2


def build_messages(
    image_path: str,
    prompt: str,
    resized_height: int | None,
    resized_width: int | None,
) -> list[dict[str, Any]]:
    image_item: dict[str, Any] = {
        "type": "image",
        "image": image_path,
    }

    if resized_height is not None:
        image_item["resized_height"] = resized_height
    if resized_width is not None:
        image_item["resized_width"] = resized_width

    return [
        {
            "role": "user",
            "content": [
                image_item,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_qwen_inputs(
    processor: Any,
    image_path: str,
    resized_height: int | None,
    resized_width: int | None,
) -> Any:
    prompt = "Describe the image."
    messages = build_messages(
        image_path,
        prompt,
        resized_height,
        resized_width,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)

        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    except Exception:
        image = Image.open(image_path).convert("RGB")

        return processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )


def _disable_rope_value(value: Any) -> Any:
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


def _make_zero_output_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return _disable_rope_value(original(*args, **kwargs))

    return wrapped


def _make_rope_sanitizing_forward(
    original: Callable[..., Any],
    signature: inspect.Signature,
) -> Callable[..., Any]:
    rope_arg_names = {"rotary_pos_emb", "position_embeddings"}
    param_names = list(signature.parameters)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        args_list = list(args)

        for name in rope_arg_names:
            if name in kwargs:
                kwargs[name] = _disable_rope_value(kwargs[name])

            if name in param_names:
                arg_index = param_names.index(name)
                if arg_index < len(args_list):
                    args_list[arg_index] = _disable_rope_value(args_list[arg_index])

        return original(*args_list, **kwargs)

    return wrapped


def _patch_attr(restorers: list[Callable[[], None]], obj: Any, name: str, value: Any) -> None:
    original = getattr(obj, name)
    setattr(obj, name, value)

    def restore() -> None:
        setattr(obj, name, original)

    restorers.append(restore)


def _patch_visual_rope_sources(visual: Any, restorers: list[Callable[[], None]]) -> int:
    patch_count = 0

    if hasattr(visual, "rot_pos_emb") and callable(visual.rot_pos_emb):
        _patch_attr(
            restorers,
            visual,
            "rot_pos_emb",
            _make_zero_output_wrapper(visual.rot_pos_emb),
        )
        patch_count += 1

    rotary_pos_emb = getattr(visual, "rotary_pos_emb", None)

    if rotary_pos_emb is not None:
        if hasattr(rotary_pos_emb, "forward") and callable(rotary_pos_emb.forward):
            _patch_attr(
                restorers,
                rotary_pos_emb,
                "forward",
                _make_zero_output_wrapper(rotary_pos_emb.forward),
            )
            patch_count += 1

        elif callable(rotary_pos_emb):
            _patch_attr(
                restorers,
                visual,
                "rotary_pos_emb",
                _make_zero_output_wrapper(rotary_pos_emb),
            )
            patch_count += 1

    return patch_count


def _patch_apply_rotary_pos_emb_vision(restorers: list[Callable[[], None]]) -> int:
    patch_count = 0

    def identity_apply_rotary_pos_emb_vision(q, k, *args, **kwargs):
        return q, k

    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue

        module_name = getattr(module, "__name__", "").lower()

        if "qwen" not in module_name or "vl" not in module_name:
            continue

        if not hasattr(module, "apply_rotary_pos_emb_vision"):
            continue

        _patch_attr(
            restorers,
            module,
            "apply_rotary_pos_emb_vision",
            identity_apply_rotary_pos_emb_vision,
        )

        patch_count += 1

    return patch_count


def _patch_visual_forward_rope_args(visual: Any, restorers: list[Callable[[], None]]) -> int:
    patch_count = 0
    seen_module_ids: set[int] = set()

    for module in visual.modules():
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

        if "rotary_pos_emb" not in signature.parameters and "position_embeddings" not in signature.parameters:
            continue

        _patch_attr(
            restorers,
            module,
            "forward",
            _make_rope_sanitizing_forward(forward, signature),
        )

        patch_count += 1

    return patch_count


@contextlib.contextmanager
def optionally_disable_visual_rope(model: Any, disable: bool):
    if not disable:
        yield
        return

    visual = get_visual_module(model)
    restorers: list[Callable[[], None]] = []
    patch_count = 0

    patch_count += _patch_visual_rope_sources(visual, restorers)
    patch_count += _patch_apply_rotary_pos_emb_vision(restorers)
    patch_count += _patch_visual_forward_rope_args(visual, restorers)

    if patch_count == 0:
        raise AttributeError("Could not find Qwen visual RoPE hooks to disable.")

    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def unwrap_visual_output(visual_output: Any) -> torch.Tensor:
    if torch.is_tensor(visual_output):
        image_embeds = visual_output

    elif hasattr(visual_output, "pooler_output") and visual_output.pooler_output is not None:
        image_embeds = visual_output.pooler_output

    elif hasattr(visual_output, "last_hidden_state") and visual_output.last_hidden_state is not None:
        image_embeds = visual_output.last_hidden_state

    elif hasattr(visual_output, "hidden_states") and visual_output.hidden_states is not None:
        image_embeds = visual_output.hidden_states[-1]

    elif isinstance(visual_output, (tuple, list)) and visual_output:
        image_embeds = visual_output[0]

    else:
        raise TypeError(f"Unsupported visual output type: {type(visual_output)!r}")

    if image_embeds.ndim == 3:
        if image_embeds.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(image_embeds.shape)}")
        image_embeds = image_embeds[0]

    if image_embeds.ndim != 2:
        raise ValueError(f"Expected [num_tokens, hidden_dim], got {tuple(image_embeds.shape)}")

    return image_embeds


def extract_visual_embeddings(
    model: Any,
    processor: Any,
    image_path: str,
    disable_rope: bool,
    resized_height: int | None,
    resized_width: int | None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    visual = get_visual_module(model)
    device = get_model_device(model)

    inputs = build_qwen_inputs(
        processor,
        image_path,
        resized_height,
        resized_width,
    )

    inputs = inputs.to(device)

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    visual_dtype = getattr(visual, "dtype", next(visual.parameters()).dtype)
    pixel_values = pixel_values.to(dtype=visual_dtype)

    with optionally_disable_visual_rope(model, disable=disable_rope):
        with torch.inference_mode():
            visual_output = visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = unwrap_visual_output(visual_output)

    grid = image_grid_thw[0].detach().cpu().tolist()
    temporal, grid_h, grid_w = [int(v) for v in grid]

    if temporal != 1:
        raise ValueError(f"Expected single-frame image, got temporal={temporal}")

    merge = get_spatial_merge_size(model)

    out_h = grid_h // merge
    out_w = grid_w // merge

    expected_pre_merge_tokens = grid_h * grid_w
    expected_post_merge_tokens = out_h * out_w

    if image_embeds.shape[0] == expected_post_merge_tokens + 1:
        image_embeds = image_embeds[1:]

    if image_embeds.shape[0] == expected_pre_merge_tokens + 1:
        image_embeds = image_embeds[1:]

    if image_embeds.shape[0] == expected_post_merge_tokens:
        token_grid_hw = (out_h, out_w)

    elif image_embeds.shape[0] == expected_pre_merge_tokens:
        token_grid_hw = (grid_h, grid_w)

    else:
        raise ValueError(
            f"Visual embedding count mismatch: got {image_embeds.shape[0]}, "
            f"expected {expected_post_merge_tokens} or {expected_pre_merge_tokens}."
        )

    return image_embeds.detach().float().cpu(), token_grid_hw


def bbox_to_token_indices(
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
    token_grid_hw: tuple[int, int],
) -> list[int]:
    grid_h, grid_w = token_grid_hw
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0.0, min(float(image_width), float(x1)))
    x2 = max(0.0, min(float(image_width), float(x2)))
    y1 = max(0.0, min(float(image_height), float(y1)))
    y2 = max(0.0, min(float(image_height), float(y2)))

    if x2 <= x1 or y2 <= y1:
        return []

    col_start = int(np.floor(x1 / image_width * grid_w))
    col_end = int(np.ceil(x2 / image_width * grid_w)) - 1
    row_start = int(np.floor(y1 / image_height * grid_h))
    row_end = int(np.ceil(y2 / image_height * grid_h)) - 1

    col_start = max(0, min(grid_w - 1, col_start))
    col_end = max(0, min(grid_w - 1, col_end))
    row_start = max(0, min(grid_h - 1, row_start))
    row_end = max(0, min(grid_h - 1, row_end))

    return [
        r * grid_w + c
        for r in range(row_start, row_end + 1)
        for c in range(col_start, col_end + 1)
    ]


def mask_to_token_indices(mask_path: Path, token_grid_hw: tuple[int, int]) -> list[int]:
    if not mask_path.is_file():
        return []

    grid_h, grid_w = token_grid_hw
    mask = Image.open(mask_path).convert("L")

    try:
        resampling = Image.Resampling.BOX
    except AttributeError:
        resampling = Image.BOX

    pooled = mask.resize((grid_w, grid_h), resampling)
    arr = np.asarray(pooled)

    rows, cols = np.where(arr > 0)

    return [
        int(r) * grid_w + int(c)
        for r, c in zip(rows, cols)
    ]


def object_token_indices(
    sample: dict[str, Any],
    role: str,
    data_root: Path,
    token_grid_hw: tuple[int, int],
    source: str,
) -> list[int]:
    if source == "mask":
        mask_rel = sample.get(f"{role}_mask_path")
        if mask_rel:
            indices = mask_to_token_indices(data_root / mask_rel, token_grid_hw)
            if indices:
                return indices

    return bbox_to_token_indices(
        sample[f"{role}_bbox"],
        image_width=sample["image_width"],
        image_height=sample["image_height"],
        token_grid_hw=token_grid_hw,
    )


def sample_object_mean(
    embeddings: torch.Tensor,
    token_indices: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    if not token_indices:
        return None

    if len(token_indices) <= 20:
        selected = token_indices
    else:
        n = max(1, len(token_indices) // 2)
        selected = rng.sample(token_indices, n)

    return embeddings[selected].mean(dim=0).numpy()


def normalize_source_dataset_arg(source_dataset: str | None) -> str | None:
    if source_dataset is None:
        return None

    if source_dataset.strip() == "" or source_dataset.strip().lower() == "none":
        return None

    return source_dataset


def get_relation(sample: dict[str, Any]) -> str | None:
    relation = sample.get("gt_relation")

    if isinstance(relation, str):
        return relation.replace("-", "_")

    return relation


def load_complete_pair_samples(
    data_root: Path,
    source_dataset: str | None,
    pair_target: str | None,
    pair_reference: str | None,
    pair_index: int,
    list_only: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], tuple[str, str]]:
    samples = load_exp1_samples(data_root=data_root)
    source_dataset = normalize_source_dataset_arg(source_dataset)

    pair_to_relation_samples: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for sample in samples:
        if source_dataset and sample["source_dataset"] != source_dataset:
            continue

        relation = get_relation(sample)

        if relation in RELATION_ORDER:
            pair = (sample["target_name"], sample["reference_name"])
            pair_to_relation_samples[pair][relation].append(sample)

    complete_pairs = sorted(
        pair
        for pair, relation_samples in pair_to_relation_samples.items()
        if all(relation_samples.get(relation) for relation in RELATION_ORDER)
    )

    if list_only:
        print(f"Complete pairs with all four relations: {len(complete_pairs)}")
        for idx, (target, reference) in enumerate(complete_pairs):
            print(f"{idx:04d}\t{target}\t{reference}")
        raise SystemExit(0)

    if pair_target is not None or pair_reference is not None:
        if not pair_target or not pair_reference:
            raise ValueError("Please provide both --pair-target and --pair-reference.")

        selected_pair = (pair_target, pair_reference)

        if selected_pair not in pair_to_relation_samples:
            raise ValueError(f"Pair not found: {selected_pair}")

        relation_samples = pair_to_relation_samples[selected_pair]

        missing = [
            relation
            for relation in RELATION_ORDER
            if not relation_samples.get(relation)
        ]

        if missing:
            raise ValueError(f"Pair {selected_pair} missing relations: {missing}")

    else:
        if not complete_pairs:
            raise ValueError("No complete target/reference pair has all four relations.")

        if pair_index < 0 or pair_index >= len(complete_pairs):
            raise ValueError(
                f"--pair-index {pair_index} out of range; "
                f"found {len(complete_pairs)} complete pairs."
            )

        selected_pair = complete_pairs[pair_index]
        relation_samples = pair_to_relation_samples[selected_pair]

    grouped = {
        relation: list(relation_samples[relation])
        for relation in RELATION_ORDER
    }

    return grouped, selected_pair


def collect_direction_vectors(
    model: Any,
    processor: Any,
    data_root: Path,
    grouped_samples: dict[str, list[dict[str, Any]]],
    samples_per_relation: int,
    seed: int,
    disable_rope: bool,
    resized_height: int | None,
    resized_width: int | None,
    object_token_source: str,
) -> tuple[np.ndarray, list[str]]:
    rng = random.Random(seed + (100000 if disable_rope else 0))

    vectors: list[np.ndarray] = []
    labels: list[str] = []

    embedding_cache: dict[tuple[str, bool], tuple[torch.Tensor, tuple[int, int]]] = {}

    for relation in RELATION_ORDER:
        relation_samples = grouped_samples[relation]
        attempts = 0

        while labels.count(relation) < samples_per_relation:
            attempts += 1

            if attempts > samples_per_relation * 50:
                raise RuntimeError(
                    f"Too many failed attempts while sampling relation {relation}."
                )

            sample = rng.choice(relation_samples)
            cache_key = (sample["image_path"], disable_rope)

            if cache_key not in embedding_cache:
                embedding_cache[cache_key] = extract_visual_embeddings(
                    model=model,
                    processor=processor,
                    image_path=sample["image_path"],
                    disable_rope=disable_rope,
                    resized_height=resized_height,
                    resized_width=resized_width,
                )

            embeddings, token_grid_hw = embedding_cache[cache_key]

            target_indices = object_token_indices(
                sample=sample,
                role="target",
                data_root=data_root,
                token_grid_hw=token_grid_hw,
                source=object_token_source,
            )

            reference_indices = object_token_indices(
                sample=sample,
                role="reference",
                data_root=data_root,
                token_grid_hw=token_grid_hw,
                source=object_token_source,
            )

            target_mean = sample_object_mean(embeddings, target_indices, rng)
            reference_mean = sample_object_mean(embeddings, reference_indices, rng)

            if target_mean is None or reference_mean is None:
                continue

            vectors.append(target_mean - reference_mean)
            labels.append(relation)

    return np.stack(vectors, axis=0), labels


def preprocess_vectors(vectors: np.ndarray, mode: str) -> np.ndarray:
    x = vectors.astype(np.float64)

    if mode == "none":
        return x

    if mode == "center":
        return x - x.mean(axis=0, keepdims=True)

    if mode == "zscore":
        x = x - x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        return x / np.maximum(std, 1e-8)

    if mode == "l2":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-8)

    if mode == "center_l2":
        x = x - x.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-8)

    raise ValueError(f"Unsupported PCA normalization mode: {mode}")


def pca_2d(vectors: np.ndarray, normalization: str) -> np.ndarray:
    x = preprocess_vectors(vectors, normalization)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def compute_qwen_pca(args: argparse.Namespace) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    grouped_samples, selected_pair = load_complete_pair_samples(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        pair_target=args.pair_target,
        pair_reference=args.pair_reference,
        pair_index=args.pair_index,
        list_only=args.list_complete_pairs,
    )

    print(f"Selected target/reference pair: {selected_pair[0]} / {selected_pair[1]}")

    for relation in RELATION_ORDER:
        print(f"  {RELATION_DISPLAY[relation]} samples: {len(grouped_samples[relation])}")

    model, processor = load_qwen_model(args)

    vectors_with_rope, labels_with_rope = collect_direction_vectors(
        model=model,
        processor=processor,
        data_root=args.data_root,
        grouped_samples=grouped_samples,
        samples_per_relation=args.samples_per_relation,
        seed=args.seed,
        disable_rope=False,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
        object_token_source=args.object_token_source,
    )

    vectors_without_rope, labels_without_rope = collect_direction_vectors(
        model=model,
        processor=processor,
        data_root=args.data_root,
        grouped_samples=grouped_samples,
        samples_per_relation=args.samples_per_relation,
        seed=args.seed,
        disable_rope=True,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
        object_token_source=args.object_token_source,
    )

    coords_with_rope = pca_2d(
        vectors_with_rope,
        normalization=args.pca_normalization,
    )

    coords_without_rope = pca_2d(
        vectors_without_rope,
        normalization=args.pca_normalization,
    )

    if args.save_npz:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_dir / "qwen_direction_vectors_pca_data.npz",
            vectors_with_rope=vectors_with_rope,
            labels_with_rope=np.array(labels_with_rope),
            coords_with_rope=coords_with_rope,
            vectors_without_rope=vectors_without_rope,
            labels_without_rope=np.array(labels_without_rope),
            coords_without_rope=coords_without_rope,
            selected_pair=np.array(selected_pair),
        )

    return coords_with_rope, labels_with_rope, coords_without_rope, labels_without_rope


# ============================================================
# Unified plotting
# ============================================================

def beautify_axes(ax: plt.Axes) -> None:
    ax.grid(True, linewidth=0.45, alpha=0.22)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.75)

    ax.tick_params(
        labelsize=9,
        width=0.8,
        length=3.5,
    )

    # 关键：让四个方框视觉大小一致
    ax.set_box_aspect(1)


def add_standard_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    scatter: Any,
    label: str | None = None,
    ticks: np.ndarray | list[int] | None = None,
    ticklabels: list[str] | None = None,
    rotate_ticklabels: bool = False,
) -> None:
    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.025,
    )

    if label is not None:
        colorbar.set_label(label, fontsize=10)

    if ticks is not None:
        colorbar.set_ticks(ticks)

    if ticklabels is not None:
        colorbar.set_ticklabels(ticklabels)

    colorbar.ax.tick_params(
        labelsize=8.5,
        length=3.5,
        width=0.8,
        pad=4 if rotate_ticklabels else 2,
    )

    if rotate_ticklabels:
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_rotation(90)
            tick_label.set_rotation_mode("anchor")
            tick_label.set_va("center")
            tick_label.set_ha("center")

    colorbar.outline.set_linewidth(0.7)


def plot_combined_four_panels(
    llava_coords: np.ndarray,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    grid_size: int,
    qwen_coords_with_rope: np.ndarray,
    qwen_labels_with_rope: list[str],
    qwen_coords_without_rope: np.ndarray,
    qwen_labels_without_rope: list[str],
    output_path: Path,
    cmap: str,
    point_size: float,
    tsne_alpha: float,
    edge_width: float,
    edge_color: str,
    dpi: int,
    transparent: bool,
    fig_width: float,
    fig_height: float,
    wspace: float,
) -> None:
    relation_to_id = {
        rel: idx
        for idx, rel in enumerate(RELATION_ORDER)
    }

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.16,
        top=0.88,
        wspace=wspace,
    )

    # --------------------------------------------------------
    # Shared t-SNE limits for LLaVA row/col panels
    # --------------------------------------------------------
    x_pad = 0.05 * (llava_coords[:, 0].max() - llava_coords[:, 0].min())
    y_pad = 0.05 * (llava_coords[:, 1].max() - llava_coords[:, 1].min())

    tsne_xlim = (
        llava_coords[:, 0].min() - x_pad,
        llava_coords[:, 0].max() + x_pad,
    )

    tsne_ylim = (
        llava_coords[:, 1].min() - y_pad,
        llava_coords[:, 1].max() + y_pad,
    )

    # --------------------------------------------------------
    # Panel 1: LLaVA row
    # --------------------------------------------------------
    ax = axes[0]

    scatter = ax.scatter(
        llava_coords[:, 0],
        llava_coords[:, 1],
        c=row_ids,
        cmap=cmap,
        s=point_size,
        alpha=tsne_alpha,
        linewidths=edge_width,
        edgecolors=edge_color if edge_width > 0 else "none",
    )

    ax.set_title("(a) PE in LLaVA (row)", fontsize=12, pad=8)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.set_xlim(*tsne_xlim)
    ax.set_ylim(*tsne_ylim)

    beautify_axes(ax)

    add_standard_colorbar(
        fig=fig,
        ax=ax,
        scatter=scatter,
        label="row ID",
        ticks=np.arange(0, grid_size, max(1, grid_size // 8)),
        rotate_ticklabels=False,
    )

    # --------------------------------------------------------
    # Panel 2: LLaVA col
    # --------------------------------------------------------
    ax = axes[1]

    scatter = ax.scatter(
        llava_coords[:, 0],
        llava_coords[:, 1],
        c=col_ids,
        cmap=cmap,
        s=point_size,
        alpha=tsne_alpha,
        linewidths=edge_width,
        edgecolors=edge_color if edge_width > 0 else "none",
    )

    ax.set_title("(b) PE in LLaVA (col)", fontsize=12, pad=8)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.set_xlim(*tsne_xlim)
    ax.set_ylim(*tsne_ylim)

    beautify_axes(ax)

    add_standard_colorbar(
        fig=fig,
        ax=ax,
        scatter=scatter,
        label="col ID",
        ticks=np.arange(0, grid_size, max(1, grid_size // 8)),
        rotate_ticklabels=False,
    )

    # --------------------------------------------------------
    # Panel 3: Qwen with 2D RoPE
    # --------------------------------------------------------
    ax = axes[2]

    label_ids = np.array([
        relation_to_id[label]
        for label in qwen_labels_with_rope
    ])

    scatter = ax.scatter(
        qwen_coords_with_rope[:, 0],
        qwen_coords_with_rope[:, 1],
        c=label_ids,
        cmap=cmap,
        s=point_size,
        alpha=0.9,
        linewidths=0,
        vmin=0,
        vmax=len(RELATION_ORDER) - 1,
    )

    ax.set_title("(c) With 2D RoPE", fontsize=12, pad=8)
    ax.set_xlabel("PCA Dimension 1", fontsize=10)
    ax.set_ylabel("PCA Dimension 2", fontsize=10)

    beautify_axes(ax)

    add_standard_colorbar(
        fig=fig,
        ax=ax,
        scatter=scatter,
        ticks=list(range(len(RELATION_ORDER))),
        ticklabels=[RELATION_DISPLAY[rel] for rel in RELATION_ORDER],
        rotate_ticklabels=True,
    )

    # --------------------------------------------------------
    # Panel 4: Qwen without 2D RoPE
    # --------------------------------------------------------
    ax = axes[3]

    label_ids = np.array([
        relation_to_id[label]
        for label in qwen_labels_without_rope
    ])

    scatter = ax.scatter(
        qwen_coords_without_rope[:, 0],
        qwen_coords_without_rope[:, 1],
        c=label_ids,
        cmap=cmap,
        s=point_size,
        alpha=0.9,
        linewidths=0,
        vmin=0,
        vmax=len(RELATION_ORDER) - 1,
    )

    ax.set_title("(d) Without 2D RoPE", fontsize=12, pad=8)
    ax.set_xlabel("PCA Dimension 1", fontsize=10)
    ax.set_ylabel("PCA Dimension 2", fontsize=10)

    beautify_axes(ax)

    add_standard_colorbar(
        fig=fig,
        ax=ax,
        scatter=scatter,
        ticks=list(range(len(RELATION_ORDER))),
        ticklabels=[RELATION_DISPLAY[rel] for rel in RELATION_ORDER],
        rotate_ticklabels=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="none" if transparent else "white",
        transparent=transparent,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    llava_coords, row_ids, col_ids, grid_size = compute_llava_tsne(args)

    (
        qwen_coords_with_rope,
        qwen_labels_with_rope,
        qwen_coords_without_rope,
        qwen_labels_without_rope,
    ) = compute_qwen_pca(args)

    output_path = args.output_dir / args.output_name

    plot_combined_four_panels(
        llava_coords=llava_coords,
        row_ids=row_ids,
        col_ids=col_ids,
        grid_size=grid_size,
        qwen_coords_with_rope=qwen_coords_with_rope,
        qwen_labels_with_rope=qwen_labels_with_rope,
        qwen_coords_without_rope=qwen_coords_without_rope,
        qwen_labels_without_rope=qwen_labels_without_rope,
        output_path=output_path,
        cmap=args.cmap,
        point_size=args.point_size,
        tsne_alpha=args.tsne_alpha,
        edge_width=args.edge_width,
        edge_color=args.edge_color,
        dpi=args.dpi,
        transparent=args.transparent,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        wspace=args.wspace,
    )

    print(f"Saved combined figure: {output_path}")


if __name__ == "__main__":
    main()
