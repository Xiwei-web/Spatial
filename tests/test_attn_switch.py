"""AttnSwitch：只在带 self_attn 的层上挂 hook；对每层都有 self_attn 的模型与修改前一致；Qwen3.5 只挂在全注意力层（3、7、11……）。
用缩小尺寸的随机模型在 CPU 上检查（不加载权重）。"""
import types

import pytest
import torch
import yaml
from torch import nn

from src import paths
from src.models.adapter import find_lm_layers, patch_eager_fp32
from src.models.attn_switch import AttnSwitch, attn_layer_indices

ALL_ATTN = ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "InternVL3-8B", "InternVL3_5-8B", "SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B",
            "GLM-4.6V-Flash", "VST-7B-RL", "RoboBrain2.0-7B", "InternVL3-2B", "SenseNova-SI-1.1-InternVL3-2B", "LLaVA-1.5-7B", "LLaVA-1.5-13B"]
QWEN35 = ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"]


def facts(m):
    return yaml.safe_load(open(paths.REPO_ROOT / "configs" / "models" / f"{m}.yaml"))["language_model"]


def test_layer_facts():
    """E0 的 16 个模型：13 个每层都是注意力层；Qwen3.5 三个只有 3、7、11…… 是全注意力层。"""
    for m in ALL_ATTN:
        assert set(facts(m)["layer_mixers"]) == {"full_attention"}, m
    for m in QWEN35:
        mx = facts(m)["layer_mixers"]
        assert [i for i, t in enumerate(mx) if t == "full_attention"] == list(range(3, len(mx), 4)), m


class _Attn(nn.Module):
    def forward(self, x):
        return x


class _Layer(nn.Module):
    def __init__(self, attn):
        super().__init__()
        if attn:
            self.self_attn = _Attn()
        else:
            self.linear_attn = nn.Identity()


def test_attn_layer_indices_and_error():
    assert attn_layer_indices(nn.ModuleList([_Layer(True) for _ in range(4)])) == [0, 1, 2, 3]
    assert attn_layer_indices(nn.ModuleList([_Layer(i % 4 == 3) for i in range(8)])) == [3, 7]
    ad = types.SimpleNamespace(model=nn.Module(), lm_layers=nn.ModuleList([_Layer(False) for _ in range(4)]))
    with pytest.raises(RuntimeError):
        AttnSwitch(ad)


def _tiny_text(cfg, n_layers, **extra):
    for k, v in dict(hidden_size=64, intermediate_size=64, num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=512, **extra).items():
        if hasattr(cfg, k) or k in ("head_dim",):
            setattr(cfg, k, v)
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:n_layers]
    if getattr(cfg, "pad_token_id", None) is not None and cfg.pad_token_id >= cfg.vocab_size:
        cfg.pad_token_id = None
    return cfg


def _hooked_layers(model, n_layers):
    _, layers = find_lm_layers(model, n_layers)
    patch = patch_eager_fp32(layers)
    sw = AttnSwitch(types.SimpleNamespace(model=model, lm_layers=layers))
    hooked = [i for i, l in enumerate(layers) if getattr(l, "self_attn", None) is not None and l.self_attn._forward_pre_hooks]
    return layers, patch, sw, hooked


@pytest.mark.parametrize("mt", ["qwen2", "qwen3", "llama"])
def test_all_attention_lm_classes_hook_every_layer(mt):
    """InternVL3 / SenseNova（Qwen2 语言模型）、InternVL3.5（Qwen3）、LLaVA-1.5（Llama）：修改前 hook 挂在全部层上，修改后集合相同。"""
    tr = pytest.importorskip("transformers")
    cfg = _tiny_text(tr.AutoConfig.for_model(mt), 4)
    m = tr.AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    layers, patch, sw, hooked = _hooked_layers(m, 4)
    assert hooked == list(range(4)) == sw.attn_layer_idx and patch["applied"] and patch["n_lm_attn_modules"] == 4


@pytest.mark.parametrize("name", ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct", "GLM-4.6V-Flash"])
def test_all_attention_vlm_classes_hook_every_layer(name):
    """Qwen2.5-VL（也是 VST、RoboBrain 的类）、Qwen3-VL、GLM-4.6V：每层都有 self_attn，hook 集合 = 全部层。"""
    tr = pytest.importorskip("transformers")
    sp = yaml.safe_load(open(paths.REPO_ROOT / "configs" / "adapters.yaml"))["models"][name]
    path = paths.get(sp.get("root", "existing_models")) / name
    cfg = tr.AutoConfig.from_pretrained(path)
    _tiny_text(cfg.text_config, 4)
    # 只建语言模型部分（视觉塔与本测试无关）
    from transformers import AutoModel
    lm = AutoModel.from_config(cfg.text_config, attn_implementation="eager")
    layers, patch, sw, hooked = _hooked_layers(lm, 4)
    assert hooked == list(range(4)) == sw.attn_layer_idx and patch["applied"]


def test_qwen35_hooks_only_full_attention_layers_and_patch_applies():
    tr = pytest.importorskip("transformers")
    path = paths.get("existing_models") / "Qwen3.5-2B"
    if not path.exists():
        path = paths.get("new_models") / "Qwen3.5-2B"
    cfg = tr.AutoConfig.from_pretrained(path).text_config
    _tiny_text(cfg, 8, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16)
    cfg.layer_types = ["linear_attention" if i % 4 != 3 else "full_attention" for i in range(8)]
    from transformers import AutoModel
    torch.manual_seed(0)
    lm = AutoModel.from_config(cfg, attn_implementation="eager").eval()
    layers, patch, sw, hooked = _hooked_layers(lm, 8)
    assert hooked == [3, 7] == sw.attn_layer_idx
    assert patch["applied"] and patch["n_lm_attn_modules"] == 2
    sw.use("eager")
    with torch.no_grad():
        lm(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))
    assert sw.calls["eager"] == 2 and sw.calls["sdpa"] == 0          # 只有两层全注意力走了（打过补丁的）eager
    sw.use("sdpa")
    with torch.no_grad():
        lm(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))
    assert sw.calls["sdpa"] == 2 and sw.calls["eager"] == 0
