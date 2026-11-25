"""模型适配层：把不同家族的 VLM 统一成同一套接口，供冒烟测试和后续实验使用。

统一接口（Adapter）：
  prepare(image, prompt) -> Prepared      构造输入，并给出视觉 token 位置与二维网格映射
  forward(prep, **kw)                     一次完整前向（eager 注意力），返回 logits / hidden_states / attentions
  generate(prep, max_new_tokens)          贪心解码，返回文本与第一步 logits
  lm_layers                               语言模型 decoder 层（nn.ModuleList），hook 都挂在这里
所有路径来自 configs/paths.yaml；模型事实来自 configs/models/<name>.yaml；用法来自 configs/adapters.yaml。
"""
from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch import nn

from src import paths

CFG_DIR = paths.REPO_ROOT / "configs"

# 贪心解码：显式清掉仓库 generation_config 里可能带的采样/惩罚参数（如 RoboBrain 的 repetition_penalty=1.05）
GREEDY = dict(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)


def adapter_spec(name: str) -> dict:
    raw = yaml.safe_load(open(CFG_DIR / "adapters.yaml"))
    if name not in raw["models"]:
        raise KeyError(f"configs/adapters.yaml 中没有模型 {name}")
    spec = dict(raw["defaults"])
    spec.update(raw["models"][name])
    spec.setdefault("root", "existing_models")
    spec["name"] = name
    return spec


def model_facts(name: str) -> dict:
    f = CFG_DIR / "models" / f"{name}.yaml"
    return yaml.safe_load(open(f)) if f.is_file() else {}


def config_hash(name: str) -> str:
    """模型事实 yaml + adapters.yaml + bbox 约定 的联合 sha256（写进每条冒烟记录）。"""
    h = hashlib.sha256()
    for f in [CFG_DIR / "models" / f"{name}.yaml", CFG_DIR / "adapters.yaml", CFG_DIR / "bbox_conventions.yaml"]:
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()


@dataclass
class Prepared:
    inputs: dict                      # 直接喂给 model(**inputs) 的张量
    prompt_text: str                  # 用户提示词原文
    visual_positions: list[int]       # 视觉 token 在输入序列中的下标（升序）
    grid_hw: tuple[int, int] | None   # 可映射的视觉 token 排成的 (行, 列)
    grid_positions: list[int] | None  # 与网格一一对应（行优先）的序列下标；None 表示无固定映射
    model_wh: tuple[int, int] | None  # 模型实际看到的图像尺寸（缩放后），abs_model_xyxy 约定需要
    orig_wh: tuple[int, int] = (0, 0)
    region_note: str = ""
    extra: dict = field(default_factory=dict)

    def token_box(self, k: int) -> list[float]:
        """第 k 个网格 token（行优先）对应的原图像素区域 [x1, y1, x2, y2]。"""
        gh, gw = self.grid_hw
        r, c = divmod(k, gw)
        x0, y0, x1, y1 = self.extra.get("view_box", (0, 0, *self.orig_wh))  # 网格覆盖的原图区域
        cw, ch = (x1 - x0) / gw, (y1 - y0) / gh
        return [x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch]


def find_lm_layers(model: nn.Module, num_layers: int) -> tuple[str, nn.ModuleList]:
    """找语言模型的 decoder 层列表：长度等于 num_layers、路径不含视觉/音频塔的 ModuleList。"""
    cands = []
    for n, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) == num_layers:
            low = n.lower()
            if any(s in low for s in ("vision", "visual", "vpm", "audio", "resampler", "mlp1", "projector")):
                continue
            cands.append((n, m))
    if len(cands) != 1:
        # 取每层参数量最大的那个（语言模型层远大于其它同长度列表）
        cands.sort(key=lambda nm: -sum(p.numel() for p in nm[1][0].parameters()))
    if not cands:
        raise RuntimeError(f"找不到长度为 {num_layers} 的 decoder 层列表")
    return cands[0]


class BaseAdapter:
    kind = "base"

    def __init__(self, name: str):
        self.name = name
        self.spec = adapter_spec(name)
        self.facts = model_facts(name)
        self.path = paths.get(self.spec["root"]) / name
        self.model = None
        self.tokenizer = None
        self.load_seconds = None

    # ---- 公共 ----
    @property
    def device(self):
        return next(self.model.parameters()).device

    def _finish_load(self, t0):
        self.model.eval()
        self.load_seconds = time.time() - t0
        n = self.facts.get("language_model", {}).get("num_layers")
        self.lm_layers_path, self.lm_layers = find_lm_layers(self.model, n)
        self.layer_mixers = self.facts["language_model"]["layer_mixers"]
        self.n_heads = self.facts["language_model"]["num_attention_heads"]

    def attn_layers(self) -> list[int]:
        """有 softmax 注意力权重的层（full / sliding）。"""
        return [i for i, t in enumerate(self.layer_mixers) if t in ("full_attention", "sliding_attention")]

    def option_token_ids(self, letters=("A", "B", "C", "D")) -> dict[str, dict[str, int | None]]:
        out = {}
        for L in letters:
            ids = {}
            for tag, s in (("bare", L), ("space", " " + L)):
                enc = self.tokenizer.encode(s, add_special_tokens=False)
                ids[tag] = enc[0] if len(enc) == 1 else None
            out[L] = ids
        return out

    @torch.no_grad()
    def forward(self, prep: Prepared, output_hidden_states=False, output_attentions=False):
        return self.model(**prep.inputs, output_hidden_states=output_hidden_states,
                          output_attentions=output_attentions, return_dict=True)

    @torch.no_grad()
    def generate(self, prep: Prepared, max_new_tokens: int = 64) -> dict:
        out = self.model.generate(**prep.inputs, max_new_tokens=max_new_tokens, output_logits=True,
                                  return_dict_in_generate=True, **GREEDY)
        seq = out.sequences[0]
        n_in = prep.inputs["input_ids"].shape[1] if self._sequences_include_prompt else 0
        new = seq[n_in:]
        return {"text": self.tokenizer.decode(new, skip_special_tokens=True),
                "text_raw": self.tokenizer.decode(new, skip_special_tokens=False),
                "token_ids": new.tolist(), "first_logits": out.logits[0][0].float().cpu()}

    _sequences_include_prompt = True


class HFChatAdapter(BaseAdapter):
    """transformers 原生模型：AutoProcessor + AutoModelForImageTextToText + chat template。"""
    kind = "hf_chat"

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.path)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.path, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        self._finish_load(t0)
        return self

    def prepare(self, image: Image.Image, prompt: str) -> Prepared:
        image = image.convert("RGB")
        W, H = image.size
        side = self.spec.get("max_image_side")
        if side and max(W, H) > side:     # 仅用于限制 anyres 切片数（见 configs/adapters.yaml）；等比缩放，网格映射仍按原图比例
            r = side / max(W, H)
            image = image.resize((round(W * r), round(H * r)), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        kw = dict(self.spec.get("chat_kwargs") or {})
        base = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kw)
        variants = [{}]
        if self.spec["grid"] == "qwen_thw" and self.spec.get("max_pixels"):
            ik = {"images_kwargs": {"max_pixels": int(self.spec["max_pixels"])}}
            variants = [{"processor_kwargs": ik}, ik, {}]      # 新式 API → 旧式 API → 默认分辨率（记录原因）
        inputs, errs = None, []
        for i, v in enumerate(variants):
            try:
                inputs = self.processor.apply_chat_template(messages, **base, **v)
                self.processor_kwargs_variant = i
                break
            except Exception as e:
                errs.append(repr(e))
        if inputs is None:
            raise RuntimeError("apply_chat_template 失败: " + " | ".join(errs))
        self.max_pixels_errors = errs
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        ids = inputs["input_ids"][0]
        vis = (ids == self.image_token_id).nonzero().flatten().tolist()
        grid_hw, grid_pos, model_wh, note, extra = None, None, None, "", {}
        g = self.spec["grid"]
        vc = self.model.config.vision_config
        if g == "qwen_thw":
            t, h, w = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
            merge, patch = int(vc.spatial_merge_size), int(vc.patch_size)
            grid_hw, model_wh = (h // merge, w // merge), (w * patch, h * patch)
            grid_pos = vis
            note = f"行优先；每个 token 对应缩放后图像上 {patch * merge}px 方格（缩放后 {model_wh[0]}x{model_wh[1]}）"
        elif g == "gemma_pool":
            pv = inputs.get("pixel_values")
            k = int(getattr(vc, "pooling_kernel_size", 3)) * int(vc.patch_size)
            if pv is not None and pv.dim() == 4:
                hh, ww = pv.shape[-2:]
                grid_hw, model_wh = (hh // k, ww // k), (ww, hh)
                grid_pos = vis
                note = f"行优先；每个 soft token 对应缩放后图像上 {k}px 方格"
            else:
                extra["pixel_values_shape"] = None if pv is None else tuple(pv.shape)
                ipi = inputs.get("image_position_ids")
                if ipi is not None:   # patch 坐标 (x, y)，据此反推网格
                    xy = ipi.reshape(-1, 2)
                    xy = xy[(xy >= 0).all(-1)]
                    pw, ph = int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1
                    pk = int(getattr(vc, "pooling_kernel_size", 3))
                    grid_hw, model_wh = (ph // pk, pw // pk), (pw * int(vc.patch_size), ph * int(vc.patch_size))
                    grid_pos = vis
                    note = f"由 image_position_ids 反推：patch 网格 {ph}x{pw}，{pk}x{pk} 池化"
        elif g == "llava15":
            n = int(round(len(vis) ** 0.5))
            grid_hw, grid_pos = (n, n), vis
            size = int(vc.image_size)
            model_wh = (size, size)
            s = min(W, H)   # CLIP 预处理：短边缩放到 size 后中心裁剪 → 网格只覆盖原图中心的 s x s 区域
            extra["view_box"] = ((W - s) / 2, (H - s) / 2, (W + s) / 2, (H + s) / 2)
            note = "CLIP 短边缩放 + 中心裁剪；网格覆盖原图中心正方形区域"
        elif g == "llava_ov":
            n = int(vc.image_size) // int(vc.patch_size)
            grid_hw, grid_pos = (n, n), vis[: n * n]
            model_wh = (int(vc.image_size),) * 2
            note = f"仅前 {n * n} 个（base 视图，整图压缩到正方形）有简单映射；其余 {len(vis) - n * n} 个是 anyres 切片 token（含 newline）"
        if grid_hw is not None and grid_hw[0] * grid_hw[1] != len(grid_pos):
            extra["grid_mismatch"] = f"grid {grid_hw} != {len(grid_pos)} tokens"
            grid_hw = grid_pos = None
        return Prepared(inputs=inputs, prompt_text=prompt, visual_positions=vis, grid_hw=grid_hw,
                        grid_positions=grid_pos, model_wh=model_wh, orig_wh=(W, H), region_note=note, extra=extra)


class InternVLRemoteAdapter(BaseAdapter):
    """InternVL “GitHub 格式”（internvl_chat，远程代码）。固定单视图：整图缩放到 448x448，256 个视觉 token。"""
    kind = "internvl_remote"
    _sequences_include_prompt = False   # generate 走 inputs_embeds，返回序列只含新 token
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        t0 = time.time()
        rev = self.facts.get("revision")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)
        # use_flash_attn=False：远程代码据此把 LLM 注意力设为 eager（否则拿不到注意力权重）
        # code_revision：SenseNova 的 auto_map 带仓库前缀，远程代码会从 Hub 取，固定到权重同一 commit
        self.model = AutoModel.from_pretrained(self.path, dtype=torch.bfloat16, trust_remote_code=True,
                                               use_flash_attn=False, code_revision=rev, device_map="auto")
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.image_token_id = self.model.img_context_token_id
        self._conv_mod = sys.modules[type(self.model).__module__]
        self._finish_load(t0)
        return self

    def prepare(self, image: Image.Image, prompt: str) -> Prepared:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        image = image.convert("RGB")
        W, H = image.size
        size = int(self.model.config.force_image_size or self.model.config.vision_config.image_size)
        tf = T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                        T.Normalize(self.MEAN, self.STD)])
        pixel_values = tf(image).unsqueeze(0).to(self.device, torch.bfloat16)
        n_tok = int(self.model.num_image_token)
        conv = self._conv_mod.get_conv_template(self.model.template)
        conv.system_message = self.model.system_message
        conv.append_message(conv.roles[0], "<image>\n" + prompt)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt().replace("<image>", "<img>" + "<IMG_CONTEXT>" * n_tok + "</img>", 1)
        enc = self.tokenizer(query, return_tensors="pt")
        inputs = {"pixel_values": pixel_values, "input_ids": enc["input_ids"].to(self.device),
                  "attention_mask": enc["attention_mask"].to(self.device)}
        vis = (inputs["input_ids"][0] == self.image_token_id).nonzero().flatten().tolist()
        g = int(round(n_tok ** 0.5))
        return Prepared(inputs=inputs, prompt_text=prompt, visual_positions=vis, grid_hw=(g, g), grid_positions=vis,
                        model_wh=(size, size), orig_wh=(W, H),
                        region_note=f"单视图：整图缩放到 {size}x{size}；{g}x{g} 行优先，每 token {size // g}px 方格",
                        extra={"eos_token_id": self.tokenizer.convert_tokens_to_ids(conv.sep.strip())})

    @torch.no_grad()
    def forward(self, prep, output_hidden_states=False, output_attentions=False):
        flags = torch.ones(prep.inputs["pixel_values"].shape[0], 1, dtype=torch.long, device=self.device)
        return self.model(**prep.inputs, image_flags=flags, output_hidden_states=output_hidden_states,
                          output_attentions=output_attentions, return_dict=True)

    @torch.no_grad()
    def generate(self, prep, max_new_tokens=64):
        out = self.model.generate(**prep.inputs, max_new_tokens=max_new_tokens, output_logits=True,
                                  return_dict_in_generate=True, eos_token_id=prep.extra["eos_token_id"], **GREEDY)
        new = out.sequences[0]
        return {"text": self.tokenizer.decode(new, skip_special_tokens=True),
                "text_raw": self.tokenizer.decode(new, skip_special_tokens=False),
                "token_ids": new.tolist(), "first_logits": out.logits[0][0].float().cpu()}


class MiniCPMRemoteAdapter(BaseAdapter):
    """MiniCPM-V（远程代码）。视觉 token 是 resampler 的 64 个 learned query，没有固定的图像区域对应。
    做法：用官方 processor 构造输入（max_slice_nums=1 固定单视图），调用官方 get_vllm_embedding 得到
    inputs_embeds，再直接驱动内部的语言模型 self.model.llm。"""
    kind = "minicpm_remote"
    _sequences_include_prompt = False

    def load(self):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.path, dtype=torch.bfloat16, trust_remote_code=True,
                                               attn_implementation="eager", device_map={"": 0})
        self._finish_load(t0)
        return self

    def prepare(self, image, prompt):
        image = image.convert("RGB")
        msgs = [{"role": "user", "content": "(<image>./</image>)\n" + prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        data = self.processor(text, [image], max_slice_nums=1, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds, _ = self.model.get_vllm_embedding(
                {"input_ids": data["input_ids"], "pixel_values": data["pixel_values"],
                 "tgt_sizes": data["tgt_sizes"], "image_bound": data["image_bound"]})
        vis = []
        for s, e in data["image_bound"][0].tolist():
            vis += list(range(s, e))
        inputs = {"inputs_embeds": embeds, "attention_mask": data["attention_mask"]}
        return Prepared(inputs=inputs, prompt_text=prompt, visual_positions=vis, grid_hw=None, grid_positions=None,
                        model_wh=None, orig_wh=image.size,
                        region_note="resampler 的 learned query，视觉 token 与图像区域没有固定对应",
                        extra={"input_ids": data["input_ids"]})

    @torch.no_grad()
    def forward(self, prep, output_hidden_states=False, output_attentions=False):
        return self.model.llm(**prep.inputs, output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions, return_dict=True)

    @torch.no_grad()
    def generate(self, prep, max_new_tokens=64):
        out = self.model.llm.generate(**prep.inputs, max_new_tokens=max_new_tokens, output_logits=True,
                                      return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id or 0, **GREEDY)
        new = out.sequences[0]
        return {"text": self.tokenizer.decode(new, skip_special_tokens=True),
                "text_raw": self.tokenizer.decode(new, skip_special_tokens=False),
                "token_ids": new.tolist(), "first_logits": out.logits[0][0].float().cpu()}


_KINDS = {"hf_chat": HFChatAdapter, "internvl_remote": InternVLRemoteAdapter, "minicpm_remote": MiniCPMRemoteAdapter}


def load_adapter(name: str) -> BaseAdapter:
    spec = adapter_spec(name)
    if spec["kind"] not in _KINDS:
        raise NotImplementedError(f"{name}: {spec.get('reason', spec['kind'])}")
    return _KINDS[spec["kind"]](name).load()
