#!/usr/bin/env python3
"""Experiment 3 probes for LLaVA-1.5-7B.

Probe 1: LLM-decoder image-token position probe.
Probe 2: LLM-decoder final-token relation probe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp3_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_IMAGENET_ROOT,
    PROJECT_ROOT,
    RELATION_LABELS,
    RELATION_TO_ID,
    OPTION_LABELS,
    OPTION_TO_ID,
    PositionProbeBank,
    answer_logit_lens_batch_rows,
    args_to_jsonable,
    average_layer_rows,
    batched,
    build_option_token_id_map,
    format_answer_logit_lens_rows,
    image_token_positions_from_input_ids,
    infer_hidden_size_from_states,
    last_nonpad_positions,
    list_imagenet_val_images,
    load_relation_samples,
    make_position_labels,
    merge_position_train_eval,
    position_probe_loss_and_stats,
    save_checkpoint,
    save_probe_plot,
    select_decoder_hidden_states,
    set_seed,
    split_train_eval,
    write_csv,
    write_json,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/llava-1.5-7b-hf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp3" / "results_llava"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 3 LLaVA position and relation probes.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--imagenet-root", type=Path, default=DEFAULT_IMAGENET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-embedding-layer", action="store_true")

    parser.add_argument("--skip-position-probe", action="store_true")
    parser.add_argument("--position-epochs", type=int, default=10)
    parser.add_argument("--position-batch-size", type=int, default=1)
    parser.add_argument("--position-lr", type=float, default=1e-3)
    parser.add_argument("--position-prompt", default="Describe the image.")
    parser.add_argument("--imagenet-start-index", type=int, default=0)
    parser.add_argument("--max-position-images", type=int, default=None)
    parser.add_argument("--max-position-eval-samples", type=int, default=None)

    parser.add_argument("--skip-relation-probe", action="store_true")
    parser.add_argument("--relation-batch-size", type=int, default=1)
    parser.add_argument("--relation-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--relation-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--relation-train-fraction", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-relation-samples", type=int, default=None)
    parser.add_argument(
        "--logit-lens-norm-mode",
        choices=["all_but_last", "all", "none"],
        default="all_but_last",
        help="How to apply the model final norm before lm_head for intermediate hidden states.",
    )

    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--save-probes", action="store_true")
    parser.add_argument("--plot-error", choices=["sem", "std", "none"], default="sem")
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


def get_model_device(model: Any) -> Any:
    return next(model.parameters()).device


def load_llava_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=resolve_torch_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def expected_decoder_layers(model: Any) -> int | None:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    return None


def llava_token_grid(model: Any) -> tuple[int, int]:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is None:
        raise AttributeError("LLaVA model config has no vision_config")
    image_size = int(getattr(vision_config, "image_size"))
    patch_size = int(getattr(vision_config, "patch_size"))
    grid = image_size // patch_size
    return grid, grid


def llava_image_token_id(model: Any, processor: Any) -> int:
    token_id = getattr(model.config, "image_token_index", None)
    if token_id is not None:
        return int(token_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        converted = tokenizer.convert_tokens_to_ids("<image>")
        if converted is not None and converted != tokenizer.unk_token_id:
            return int(converted)
    raise AttributeError("Could not infer LLaVA image token id")


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


def move_inputs_to_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = get_model_device(model)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def build_llava_inputs(
    processor: Any,
    image_paths: list[str],
    prompts: list[str],
    model: Any,
) -> dict[str, Any]:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    texts = [format_llava_prompt(processor, prompt) for prompt in prompts]
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
    return move_inputs_to_device(dict(inputs), model)


def forward_decoder_states(
    model: Any,
    inputs: dict[str, Any],
    args: argparse.Namespace,
) -> list[Any]:
    import torch

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    return select_decoder_hidden_states(
        outputs,
        expected_layers=expected_decoder_layers(model),
        include_embedding_layer=args.include_embedding_layer,
    )


def evaluate_position_probe(
    model: Any,
    processor: Any,
    probes: PositionProbeBank,
    image_paths: list[str],
    grid_hw: tuple[int, int],
    image_token_id: int,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    import torch

    probes.eval()
    device = next(iter(probes.parameters())).device
    x_labels, y_labels = make_position_labels(grid_hw, device)
    rows = []
    batches = batched(image_paths, args.position_batch_size)
    for batch_idx, batch_paths in enumerate(batches, start=1):
        prompts = [args.position_prompt] * len(batch_paths)
        inputs = build_llava_inputs(processor, batch_paths, prompts, model)
        layer_states = forward_decoder_states(model, inputs, args)
        image_positions = image_token_positions_from_input_ids(
            inputs["input_ids"],
            image_token_id=image_token_id,
            expected_tokens=grid_hw[0] * grid_hw[1],
        )
        with torch.no_grad():
            _, batch_rows = position_probe_loss_and_stats(probes, layer_states, image_positions, x_labels, y_labels)
        rows.extend(batch_rows)
        if args.print_every > 0 and batch_idx % args.print_every == 0:
            print(f"[llava][position-eval] batch {batch_idx}/{len(batches)}")
    return average_layer_rows(rows)


def train_position_probe(
    model: Any,
    processor: Any,
    grid_hw: tuple[int, int],
    image_token_id: int,
    relation_samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[PositionProbeBank, list[dict[str, Any]]]:
    import torch

    train_images = list_imagenet_val_images(
        args.imagenet_root,
        start_index=args.imagenet_start_index,
        max_images=args.max_position_images,
    )
    eval_samples = relation_samples[: args.max_position_eval_samples] if args.max_position_eval_samples else relation_samples
    eval_images = [sample["image_path"] for sample in eval_samples]

    device = get_model_device(model)
    x_labels, y_labels = make_position_labels(grid_hw, device)
    probes = None
    optimizer = None
    latest_train_rows: list[dict[str, float]] = []

    print(f"[llava][position] train_images={len(train_images)} eval_images={len(eval_images)} grid={grid_hw}")
    for epoch in range(1, args.position_epochs + 1):
        random_order = list(train_images)
        import random

        random.shuffle(random_order)
        train_rows = []
        batches = batched(random_order, args.position_batch_size)
        for batch_idx, batch_paths in enumerate(batches, start=1):
            prompts = [args.position_prompt] * len(batch_paths)
            inputs = build_llava_inputs(processor, batch_paths, prompts, model)
            layer_states = forward_decoder_states(model, inputs, args)
            image_positions = image_token_positions_from_input_ids(
                inputs["input_ids"],
                image_token_id=image_token_id,
                expected_tokens=grid_hw[0] * grid_hw[1],
            )
            if probes is None:
                hidden_size = infer_hidden_size_from_states(layer_states)
                probes = PositionProbeBank(len(layer_states), hidden_size, grid_hw, device)
                optimizer = torch.optim.AdamW(probes.parameters(), lr=args.position_lr)
                print(f"[llava][position] layers={len(layer_states)} hidden_size={hidden_size}")
            assert optimizer is not None and probes is not None
            probes.train()
            optimizer.zero_grad(set_to_none=True)
            loss, batch_rows = position_probe_loss_and_stats(probes, layer_states, image_positions, x_labels, y_labels)
            loss.backward()
            optimizer.step()
            train_rows.extend(batch_rows)
            if args.print_every > 0 and batch_idx % args.print_every == 0:
                print(
                    f"[llava][position] epoch {epoch}/{args.position_epochs} "
                    f"batch {batch_idx}/{len(batches)} loss={loss.item():.4f}"
                )
        latest_train_rows = average_layer_rows(train_rows)
        print(f"[llava][position] epoch {epoch} complete")

    if probes is None:
        raise RuntimeError("Position probe was not initialized")
    eval_rows = evaluate_position_probe(model, processor, probes, eval_images, grid_hw, image_token_id, args)
    merged = merge_position_train_eval(latest_train_rows, eval_rows)
    return probes, merged


def evaluate_answer_logit_lens(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    option_token_ids: dict[str, list[int]],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    import torch

    rows = []
    batches = batched(samples, args.relation_batch_size)
    for batch_idx, batch_samples in enumerate(batches, start=1):
        image_paths = [sample["image_path"] for sample in batch_samples]
        prompts = [sample["relation_prompt"] for sample in batch_samples]
        labels = torch.tensor([OPTION_TO_ID[sample["answer"]] for sample in batch_samples], dtype=torch.long, device=get_model_device(model))
        inputs = build_llava_inputs(processor, image_paths, prompts, model)
        layer_states = forward_decoder_states(model, inputs, args)
        final_positions = last_nonpad_positions(inputs["attention_mask"])
        with torch.no_grad():
            batch_rows = answer_logit_lens_batch_rows(
                model=model,
                layer_states=layer_states,
                final_positions=final_positions,
                labels=labels,
                option_token_ids=option_token_ids,
                norm_mode=args.logit_lens_norm_mode,
            )
        rows.extend(batch_rows)
        if args.print_every > 0 and batch_idx % args.print_every == 0:
            print(f"[llava][answer-logit-lens] batch {batch_idx}/{len(batches)}")
    return average_layer_rows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relation_samples = load_relation_samples(
        data_root=args.data_root,
        max_samples=args.max_relation_samples,
    )
    if not relation_samples:
        raise RuntimeError("No usable relation samples found")

    model, processor = load_llava_model(args)
    grid_hw = llava_token_grid(model)
    image_token_id = llava_image_token_id(model, processor)
    print(f"[llava] grid={grid_hw} image_token_id={image_token_id} relation_samples={len(relation_samples)}")

    summary: dict[str, Any] = {
        "model": "llava-1.5-7b-hf",
        "args": args_to_jsonable(args),
        "grid_hw": list(grid_hw),
        "image_token_id": image_token_id,
        "relation_labels": list(RELATION_LABELS),
        "relation_label_to_id": RELATION_TO_ID,
        "option_labels": list(OPTION_LABELS),
        "option_label_to_id": OPTION_TO_ID,
        "num_relation_samples": len(relation_samples),
    }

    if not args.skip_position_probe:
        position_probes, position_rows = train_position_probe(
            model=model,
            processor=processor,
            grid_hw=grid_hw,
            image_token_id=image_token_id,
            relation_samples=relation_samples,
            args=args,
        )
        write_json(args.output_dir / "llava_position_probe_metrics.json", position_rows)
        write_csv(args.output_dir / "llava_position_probe_metrics.csv", position_rows)
        save_probe_plot(
            position_rows,
            args.output_dir / "llava_position_probe_accuracy.png",
            "LLaVA Position Probe Accuracy vs Layer",
            ["eval_x_acc", "eval_y_acc", "eval_xy_acc"],
            error_suffix=None if args.plot_error == "none" else args.plot_error,
        )
        if args.save_probes:
            save_checkpoint(
                args.output_dir / "llava_position_probes.pt",
                {"probe_state": position_probes.state_dict(), "grid_hw": grid_hw, "args": args_to_jsonable(args)},
            )
        summary["position_probe_metrics"] = position_rows

    if not args.skip_relation_probe:
        tokenizer = getattr(processor, "tokenizer", processor)
        option_token_ids = build_option_token_id_map(tokenizer)
        relation_rows = format_answer_logit_lens_rows(
            evaluate_answer_logit_lens(model, processor, relation_samples, option_token_ids, args)
        )
        write_json(args.output_dir / "llava_answer_logit_lens_metrics.json", relation_rows)
        write_csv(args.output_dir / "llava_answer_logit_lens_metrics.csv", relation_rows)
        save_probe_plot(
            relation_rows,
            args.output_dir / "llava_answer_logit_lens_accuracy.png",
            "LLaVA Final-token Answer Logit Lens Accuracy vs Layer",
            ["answer_acc"],
            error_suffix=None if args.plot_error == "none" else args.plot_error,
        )
        summary["option_token_ids"] = option_token_ids
        summary["answer_logit_lens_metrics"] = relation_rows

    write_json(args.output_dir / "llava_exp3_summary.json", summary)
    print(f"Done. Summary: {args.output_dir / 'llava_exp3_summary.json'}")


if __name__ == "__main__":
    main()
