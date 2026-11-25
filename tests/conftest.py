"""测试夹具。默认用一个随机初始化的小型 Qwen2 语言模型（CPU 即可，不需要任何权重）；
设置环境变量 INTERP_TEST_MODEL=<模型名> 时改为加载真实模型（需要 GPU，通过调度系统运行）。"""
import os

import pytest
import torch

from src.models.adapter import Prepared


class TinyAdapter:
    """与真实 adapter 相同的最小接口，包一层随机初始化的 4 层 Qwen2。"""
    name = "tiny-random-qwen2"

    def __init__(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                          attn_implementation="eager")
        self.model = Qwen2ForCausalLM(cfg).eval()
        self.lm_layers = self.model.model.layers
        self.layer_mixers = ["full_attention"] * 4
        self.n_heads = 4

    def attn_layers(self):
        return [0, 1, 2, 3]

    @torch.no_grad()
    def forward(self, prep, output_hidden_states=False, output_attentions=False):
        return self.model(**prep.inputs, output_hidden_states=output_hidden_states,
                          output_attentions=output_attentions, return_dict=True)

    @torch.no_grad()
    def generate(self, prep, max_new_tokens=8):
        out = self.model.generate(**prep.inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  output_logits=True, return_dict_in_generate=True, pad_token_id=0)
        return {"token_ids": out.sequences[0].tolist(), "first_logits": out.logits[0][0]}


@pytest.fixture(scope="session")
def adapter_and_prep():
    name = os.environ.get("INTERP_TEST_MODEL")
    if not name:
        ad = TinyAdapter()
        ids = torch.arange(3, 23).unsqueeze(0)
        prep = Prepared(inputs={"input_ids": ids, "attention_mask": torch.ones_like(ids)}, prompt_text="",
                        visual_positions=list(range(4, 12)), grid_hw=(2, 4), grid_positions=list(range(4, 12)),
                        model_wh=(4, 2), orig_wh=(400, 200))
        return ad, prep
    from PIL import Image

    from src.data import load_split
    from src.models import load_adapter
    from src.prompts import relation_mcq
    ad = load_adapter(name)
    s = load_split("whatsup_a", "dev")[0]
    prompt, _ = relation_mcq("whatsup_a", s.target, s.reference)
    return ad, ad.prepare(Image.open(s.image_path), prompt)
