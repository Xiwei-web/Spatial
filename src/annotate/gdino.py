"""只读调用本地的原版 GroundingDINO（代码与权重路径见 configs/paths.yaml）。

说明：仓库没有编译 C++/CUDA 算子（_C）。原版代码在 GPU 上会强制调用 _C；这里把多尺度可变形注意力
固定到仓库自带的纯 PyTorch 实现 multi_scale_deformable_attn_pytorch（CPU 路径用的同一函数），
这样可以在 GPU 上运行且不需要在用户的仓库里编译任何东西。
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

from src import paths

BOX_THRESHOLD_FLOOR = 0.10   # 只是候选的下限；质检阈值在 scripts/annotate_whatsup.py 里定


def load(variant: str = "swinb", device: str = "cuda"):
    repo = paths.get("groundingdino_repo")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    sys.dont_write_bytecode = True      # 不在用户仓库里写 __pycache__
    from groundingdino.models import build_model
    from groundingdino.models.GroundingDINO import ms_deform_attn as M
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    def _forward(self, query, key=None, value=None, query_pos=None, key_padding_mask=None, reference_points=None,
                 spatial_shapes=None, level_start_index=None, **kwargs):
        # 与原 forward 相同，只是最后一步固定走纯 PyTorch 分支
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query, value = query.permute(1, 0, 2), value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        so = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        aw = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        aw = aw.softmax(-1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            loc = reference_points[:, :, None, :, None, :] + so / norm[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            loc = (reference_points[:, :, None, :, None, :2]
                   + so / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(reference_points.shape)
        out = M.multi_scale_deformable_attn_pytorch(value, spatial_shapes, loc, aw)
        out = self.output_proj(out)
        if not self.batch_first:
            out = out.permute(1, 0, 2)
        return out

    M.MultiScaleDeformableAttention.forward = _forward
    cfg_name, ckpt_name = {"swinb": ("GroundingDINO_SwinB_cfg.py", "groundingdino_swinb_cogcoor.pth"),
                           "swint": ("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth")}[variant]
    args = SLConfig.fromfile(str(repo / "groundingdino" / "config" / cfg_name))
    args.device = device
    args.text_encoder_type = str(paths.get("groundingdino_ckpt") / "bert-base-uncased")   # 本地 BERT，不联网
    model = build_model(args)
    sd = torch.load(paths.get("groundingdino_ckpt") / ckpt_name, map_location="cpu", weights_only=False)
    missing = model.load_state_dict(clean_state_dict(sd["model"]), strict=False)
    model.eval().to(device)
    model._load_report = {"variant": variant, "ckpt": ckpt_name, "missing_keys": list(missing.missing_keys),
                          "unexpected_keys": list(missing.unexpected_keys)}
    return model


def _transform(image: Image.Image):
    import groundingdino.datasets.transforms as T
    tf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return tf(image.convert("RGB"), None)[0]


@torch.no_grad()
def detect(model, image: Image.Image, phrase: str, floor: float = BOX_THRESHOLD_FLOOR) -> list[dict]:
    """返回按分数降序的候选框：[{box_xyxy(原图像素), score}]。分数 = 该 query 在提示词 token 上的最大 sigmoid logit。"""
    dev = next(model.parameters()).device
    caption = phrase.lower().strip()
    if not caption.endswith("."):
        caption += " ."
    out = model(_transform(image)[None].to(dev), captions=[caption])
    logits = out["pred_logits"].sigmoid()[0]          # [900, 256]
    boxes = out["pred_boxes"][0]                       # cxcywh 归一化
    score = logits.max(dim=1)[0]
    keep = score > floor
    W, H = image.size
    res = []
    for b, s in zip(boxes[keep].cpu().numpy(), score[keep].cpu().numpy()):
        cx, cy, w, h = b
        res.append({"box_xyxy": [float(np.clip((cx - w / 2) * W, 0, W)), float(np.clip((cy - h / 2) * H, 0, H)),
                                 float(np.clip((cx + w / 2) * W, 0, W)), float(np.clip((cy + h / 2) * H, 0, H))],
                    "score": float(s)})
    res.sort(key=lambda r: -r["score"])
    return res
