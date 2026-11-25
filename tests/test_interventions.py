"""干预的正确性单元测试（任务书 阶段 E 第 7 项）。判定标准定义在 src/checks.py，这里不放宽。"""
from src import checks


def test_noop_hooks_bitwise_identical(adapter_and_prep):
    """不做任何干预时，带 hook 的前向输出与原模型逐位一致。"""
    ad, prep = adapter_and_prep
    r = checks.check_noop_identity(ad, prep)
    assert r["pass"], r


def test_self_patch_leaves_output_unchanged(adapter_and_prep):
    """把一次运行的激活 patch 回它自己，输出不变；patch 进零则输出必须改变。"""
    ad, prep = adapter_and_prep
    r = checks.check_self_patch(ad, prep)
    assert r["pass"], r


def test_knockout_zeroes_attention(adapter_and_prep):
    """屏蔽注意力后，被屏蔽位置的注意力权重确实为 0，且其余行仍归一、更早的层不受影响。"""
    ad, prep = adapter_and_prep
    r = checks.check_knockout(ad, prep)
    assert r["pass"], r
    assert r["max_weight_after"] == 0.0


def test_head_hook_read_and_rewrite(adapter_and_prep):
    """单个 head 的输出可取到；只读/恒等改写不改变输出，置零则改变输出。"""
    ad, prep = adapter_and_prep
    r = checks.check_head_hook(ad, prep)
    assert r["pass"], r


def test_hidden_states_and_attention_shapes(adapter_and_prep):
    ad, prep = adapter_and_prep
    assert checks.check_hidden_states(ad, prep)["pass"]
    assert checks.check_attention_weights(ad, prep)["pass"]


def test_greedy_generation_is_deterministic(adapter_and_prep):
    ad, prep = adapter_and_prep
    assert checks.check_determinism(ad, prep, 8)["pass"]


def test_eager_fp32_patch_matches_reference_in_float32():
    """float32 模型上，“打分放到 float32 计算”的 eager 注意力与原实现应数值一致（同一算式）。"""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from src.models.adapter import patch_eager_fp32
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, attn_implementation="eager")
    m = Qwen2ForCausalLM(cfg).eval()
    ids = torch.arange(3, 23).unsqueeze(0)
    with torch.no_grad():
        ref = m(input_ids=ids, output_attentions=True)
        rep = patch_eager_fp32(m.model.layers)
        assert rep["applied"], rep
        new = m(input_ids=ids, output_attentions=True)
    assert torch.allclose(ref.logits, new.logits, atol=1e-5)
    assert torch.allclose(ref.attentions[0], new.attentions[0], atol=1e-6)
