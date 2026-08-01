#!/usr/bin/env python3
"""
Visualize Qwen2.5-VL direction vectors for spatial relations.

This script implements the PCA panels (a) and (b) of Figure 7 in:
"Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models".

Paper method, adapted to the local whatsUP data:
  - Use What’sUp subset B and Qwen2.5-VL-7B.
  - For each relation r in {left, right, behind, in front of}, extract visual
    embeddings for the satellite object S (target) and nucleus object N
    (reference).
  - Compute a direction vector:

        v_r = mean(V^r_S) - mean(V^r_N)

    where V^r_S and V^r_N are visual embeddings whose token cells overlap the
    target/reference boxes.
  - For each relation, repeat random object-token sampling 100 times.
  - Run PCA over the 400 direction vectors and plot two panels:
      (a) with 2D RoPE position embedding;
      (b) without 2D RoPE position embedding.

The "without position embedding" condition is implemented by monkey-patching
the Qwen visual tower's RoPE paths so rotary embeddings become no-ops during
visual embedding extraction.

No model generation is performed; this only forwards images through the visual
tower and projector/merger to obtain visual embeddings.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_DIR = PROJECT_ROOT / "exp1"
if str(EXP1_DIR) not in sys.path:
    sys.path.insert(0, str(EXP1_DIR))

from exp1_utils import DEFAULT_DATA_ROOT, load_exp1_samples  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp1" / "outputs"

RELATION_ORDER = ["left_of", "right_of", "behind", "in_front_of"]
RELATION_DISPLAY = {
    "left_of": "Left",
    "right_of": "Right",
    "behind": "Behind",
    "in_front_of": "In front of",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw PCA of Qwen2.5-VL spatial direction vectors with/without 2D RoPE."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="qwen_direction_vectors_pca7.png")
    parser.add_argument("--source-dataset", default="B", help="Default follows the paper: What’sUp subset B.")
    parser.add_argument(
        "--pair-target",
        default=None,
        help="Target/satellite object name for a fixed object pair, e.g. mug.",
    )
    parser.add_argument(
        "--pair-reference",
        default=None,
        help="Reference/nucleus object name for a fixed object pair, e.g. knife.",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help=(
            "If --pair-target/--pair-reference are not provided, choose this complete "
            "target/reference pair among pairs that have all four relations."
        ),
    )
    parser.add_argument(
        "--list-complete-pairs",
        action="store_true",
        help="List complete target/reference pairs and exit.",
    )
    parser.add_argument("--samples-per-relation", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--object-token-source",
        choices=["mask", "bbox"],
        default="mask",
        help="Use SAM masks or bbox rectangles to select object visual tokens. Mask is closer to the paper.",
    )
    parser.add_argument(
        "--pca-normalization",
        choices=["none", "center", "zscore", "l2", "center_l2"],
        default="zscore",
        help="Preprocess direction vectors before PCA. zscore reduces semantic/norm scale dominance.",
    )
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
    parser.add_argument(
        "--resized-height",
        type=int,
        default=None,
        help="Optional fixed image height for qwen_vl_utils, e.g. 448.",
    )
    parser.add_argument(
        "--resized-width",
        type=int,
        default=None,
        help="Optional fixed image width for qwen_vl_utils, e.g. 448.",
    )
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-npz", action="store_true")
    return parser.parse_args()


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

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": resolve_torch_dtype(args.dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir, **model_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def get_visual_module(model: Any) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen visual module at model.visual or model.model.visual")


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


def build_messages(image_path: str, prompt: str, resized_height: int | None, resized_width: int | None) -> list[dict[str, Any]]:
    image_item: dict[str, Any] = {"type": "image", "image": image_path}
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
    messages = build_messages(image_path, prompt, resized_height, resized_width)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
    """Convert a RoPE argument to a no-op value while preserving its shape."""
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


def _make_rope_sanitizing_forward(original: Callable[..., Any], signature: inspect.Signature) -> Callable[..., Any]:
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
        _patch_attr(restorers, visual, "rot_pos_emb", _make_zero_output_wrapper(visual.rot_pos_emb))
        patch_count += 1

    rotary_pos_emb = getattr(visual, "rotary_pos_emb", None)
    if rotary_pos_emb is not None:
        if hasattr(rotary_pos_emb, "forward") and callable(rotary_pos_emb.forward):
            _patch_attr(restorers, rotary_pos_emb, "forward", _make_zero_output_wrapper(rotary_pos_emb.forward))
            patch_count += 1
        elif callable(rotary_pos_emb):
            _patch_attr(restorers, visual, "rotary_pos_emb", _make_zero_output_wrapper(rotary_pos_emb))
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
        _patch_attr(restorers, module, "apply_rotary_pos_emb_vision", identity_apply_rotary_pos_emb_vision)
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
        _patch_attr(restorers, module, "forward", _make_rope_sanitizing_forward(forward, signature))
        patch_count += 1

    return patch_count


@contextlib.contextmanager
def optionally_disable_visual_rope(model: Any, disable: bool):
    """Temporarily force Qwen visual RoPE to be a no-op."""
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
        raise AttributeError(
            "Could not find any Qwen visual RoPE hooks to disable. "
            "Your transformers version may have renamed rot_pos_emb, rotary_pos_emb, "
            "apply_rotary_pos_emb_vision, and the attention RoPE arguments."
        )

    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def unwrap_visual_output(visual_output: Any) -> torch.Tensor:
    """Normalize visual forward outputs from different transformers versions."""
    if torch.is_tensor(visual_output):
        image_embeds = visual_output
    elif hasattr(visual_output, "pooler_output") and visual_output.pooler_output is not None:
        # Newer transformers versions may return BaseModelOutputWithPooling
        # from Qwen's visual module. The VLM uses the post-merger/projector
        # visual embeddings, which are exposed as pooler_output in those
        # versions. This is closer to V = f_C(f_I(X_V)) in the paper.
        image_embeds = visual_output.pooler_output
    elif hasattr(visual_output, "last_hidden_state") and visual_output.last_hidden_state is not None:
        # Fallback: this is often the pre-merger vision-tower sequence.
        image_embeds = visual_output.last_hidden_state
    elif hasattr(visual_output, "hidden_states") and visual_output.hidden_states is not None:
        image_embeds = visual_output.hidden_states[-1]
    elif isinstance(visual_output, (tuple, list)) and visual_output:
        image_embeds = visual_output[0]
    else:
        raise TypeError(
            "Unsupported visual output type. Expected a tensor or an object with "
            f"last_hidden_state/hidden_states, got {type(visual_output)!r}."
        )

    if not torch.is_tensor(image_embeds):
        raise TypeError(f"Could not unwrap visual output to a tensor, got {type(image_embeds)!r}.")

    if image_embeds.ndim == 3:
        if image_embeds.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for visual embeddings, got shape {tuple(image_embeds.shape)}")
        image_embeds = image_embeds[0]

    if image_embeds.ndim != 2:
        raise ValueError(f"Expected visual embeddings with shape [num_tokens, hidden_dim], got {tuple(image_embeds.shape)}")

    return image_embeds


def extract_visual_embeddings(
    model: Any,
    processor: Any,
    image_path: str,
    disable_rope: bool,
    resized_height: int | None,
    resized_width: int | None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return visual embeddings and their post-merge token grid size (H, W)."""
    visual = get_visual_module(model)
    device = get_model_device(model)
    inputs = build_qwen_inputs(processor, image_path, resized_height, resized_width)
    inputs = inputs.to(device)

    if not hasattr(inputs, "pixel_values") and "pixel_values" not in inputs:
        raise KeyError("Processor output does not contain pixel_values")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    visual_dtype = getattr(visual, "dtype", next(visual.parameters()).dtype)
    pixel_values = pixel_values.to(dtype=visual_dtype)

    with optionally_disable_visual_rope(model, disable=disable_rope):
        with torch.inference_mode():
            visual_output = visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = unwrap_visual_output(visual_output)

    grid = image_grid_thw[0].detach().cpu().tolist()
    if len(grid) != 3:
        raise ValueError(f"Expected image_grid_thw row to have 3 values, got {grid}")
    temporal, grid_h, grid_w = [int(v) for v in grid]
    if temporal != 1:
        raise ValueError(f"This visualization expects a single-frame image, got temporal={temporal}")

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
            f"expected either post-merge {expected_post_merge_tokens} "
            f"or pre-merge {expected_pre_merge_tokens} from grid {grid_h}x{grid_w} and merge={merge}."
        )
    if not hasattr(extract_visual_embeddings, "_printed_shape"):
        print(
            "Visual embedding shape:",
            tuple(image_embeds.shape),
            "token_grid_hw:",
            token_grid_hw,
            "raw_grid_hw:",
            (grid_h, grid_w),
            "merge:",
            merge,
        )
        extract_visual_embeddings._printed_shape = True
    return image_embeds.detach().float().cpu(), token_grid_hw


def bbox_to_token_indices(
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
    token_grid_hw: tuple[int, int],
) -> list[int]:
    """Map an xyxy bbox in original pixels to post-merge visual token indices."""
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

    return [r * grid_w + c for r in range(row_start, row_end + 1) for c in range(col_start, col_end + 1)]


def mask_to_token_indices(mask_path: Path, token_grid_hw: tuple[int, int]) -> list[int]:
    """Map a binary object mask to token indices via area downsampling."""
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
    return [int(r) * grid_w + int(c) for r, c in zip(rows, cols)]


def object_token_indices(
    sample: dict[str, Any],
    role: str,
    data_root: Path,
    token_grid_hw: tuple[int, int],
    source: str,
) -> list[int]:
    if role not in {"target", "reference"}:
        raise ValueError(f"Unsupported role: {role}")

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


def plot_pca(
    coords_with_rope: np.ndarray,
    labels_with_rope: list[str],
    coords_without_rope: np.ndarray,
    labels_without_rope: list[str],
    output_path: Path,
    point_size: float,
    dpi: int,
) -> None:
    relation_to_id = {rel: idx for idx, rel in enumerate(RELATION_ORDER)}

    # 使用 cividis 色系
    cmap = "magma"

    # 两个子图，保证方框大小一致
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.4),
        constrained_layout=True,
    )

    panels = [
        (axes[0], coords_with_rope, labels_with_rope, "(a) With 2D RoPE"),
        (axes[1], coords_without_rope, labels_without_rope, "(b) Without 2D RoPE"),
    ]

    for ax, coords, labels, title in panels:
        label_ids = np.array([relation_to_id[label] for label in labels])

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=label_ids,
            cmap=cmap,
            s=point_size,
            alpha=0.9,
            linewidths=0,
            vmin=0,
            vmax=len(RELATION_ORDER) - 1,
        )

        # 坐标轴和标题
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("PCA Dimension 1", fontsize=12)
        ax.set_ylabel("PCA Dimension 2", fontsize=12)

        ax.tick_params(
            axis="both",
            labelsize=10,
            direction="out",
            length=4,
            width=1.0,
        )

        ax.grid(False)

        # 边框略微加粗，保证清晰
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        # colorbar
        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            fraction=0.046,
            pad=0.055,
        )

        colorbar.set_ticks(range(len(RELATION_ORDER)))
        colorbar.set_ticklabels(
            [RELATION_DISPLAY[rel] for rel in RELATION_ORDER]
        )

        # colorbar 刻度样式
        colorbar.ax.tick_params(
            labelsize=10,
            length=4,
            width=1.0,
            pad=8,
        )

        # 关键修改：让 colorbar 的文字整体旋转 90 度
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_rotation(90)
            tick_label.set_rotation_mode("anchor")
            tick_label.set_va("center")
            tick_label.set_ha("center")

        colorbar.outline.set_linewidth(1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
            raise ValueError("Please provide both --pair-target and --pair-reference, or neither.")
        selected_pair = (pair_target, pair_reference)
        if selected_pair not in pair_to_relation_samples:
            raise ValueError(f"Pair not found after filtering: {selected_pair}")
        relation_samples = pair_to_relation_samples[selected_pair]
        missing = [relation for relation in RELATION_ORDER if not relation_samples.get(relation)]
        if missing:
            raise ValueError(f"Pair {selected_pair} does not have all four relations. Missing: {missing}")
    else:
        if not complete_pairs:
            raise ValueError("No complete target/reference pair has all four relations.")
        if pair_index < 0 or pair_index >= len(complete_pairs):
            raise ValueError(f"--pair-index {pair_index} out of range; found {len(complete_pairs)} complete pairs.")
        selected_pair = complete_pairs[pair_index]
        relation_samples = pair_to_relation_samples[selected_pair]

    grouped = {relation: list(relation_samples[relation]) for relation in RELATION_ORDER}
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
                    f"Too many failed attempts while sampling relation {relation}. "
                    "Check bbox-token mapping or reduce --samples-per-relation."
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    coords_with_rope = pca_2d(vectors_with_rope, normalization=args.pca_normalization)
    coords_without_rope = pca_2d(vectors_without_rope, normalization=args.pca_normalization)

    output_path = args.output_dir / args.output_name
    plot_pca(
        coords_with_rope=coords_with_rope,
        labels_with_rope=labels_with_rope,
        coords_without_rope=coords_without_rope,
        labels_without_rope=labels_without_rope,
        output_path=output_path,
        point_size=args.point_size,
        dpi=args.dpi,
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

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
