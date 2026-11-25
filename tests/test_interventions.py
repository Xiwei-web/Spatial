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
