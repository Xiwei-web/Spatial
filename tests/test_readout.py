"""框 teacher-forced 读出的 CPU 测试：枚举树、token 区间定位、以及随机小模型（float32）上两条计算路径一致。"""
import json

import pytest
import torch

from src import paths
from src.readout import coord_token_spans, digit_trie, qwen25vl_encode, qwen25vl_response, trie_distribution

NAME = "Qwen2.5-VL-7B-Instruct"


def test_trie_sizes():
    assert {k: len(v) for k, v in digit_trie(1000).items()} == {1: 10, 2: 90, 3: 1}
    assert {k: len(v) for k, v in digit_trie(1148).items()} == {1: 10, 2: 90, 3: 15}
    assert digit_trie(1148)[3] == [str(v) for v in range(100, 115)]


def _random_pnext(N, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for p in [""] + [s for v in digit_trie(N).values() for s in v]:
        x = torch.rand(11, generator=g, dtype=torch.double)
        out[p] = (x / x.sum())[:10]                 # 第 11 项是“结束”
    return out


@pytest.mark.parametrize("N", [840, 1000, 1148, 2047])
def test_trie_against_bruteforce(N):
    pn = _random_pnext(N)
    P = trie_distribution(N, pn)
    for v in list(range(0, N + 1, 7)) + [0, 9, 10, 99, 100, 999 if N >= 999 else N, N]:
        s, w = str(v), 1.0
        for i, ch in enumerate(s):
            w *= float(pn[s[:i]][int(ch)])
        if s in pn:
            w *= 1 - float(pn[s].sum())
        assert abs(float(P[v]) - w) < 1e-15, v


def test_trie_matches_internvl_formula():
    """max_value=1000 时与 InternVLBoxReadout._tree 的公式逐值相同。"""
    pn = _random_pnext(1000, 3)
    P = torch.zeros(1001, dtype=torch.double)
    for d in range(10):
        P[d] = pn[""][d] * (1 - pn[str(d)].sum())
    for x in range(1, 10):
        for y in range(10):
            q = pn[""][x] * pn[str(x)][y]
            P[10 * x + y] = q * (1 - pn[f"{x}{y}"].sum())
            for z in range(10):
                P[100 * x + 10 * y + z] = q * pn[f"{x}{y}"][z]
    P[1000] = P[100] * pn["100"][0]
    P[100] = P[100] * (1 - pn["100"].sum())
    assert torch.allclose(P, trie_distribution(1000, pn), rtol=0, atol=1e-15)


def test_trie_mass():
    """所有会越界或非法的分支概率为 0 时，总质量等于首位是数字的概率；首位归一到数字后为 1。"""
    N = 1148
    pn = _random_pnext(N, 1)
    z = torch.zeros(10, dtype=torch.double)
    pn["0"] = z
    for s, q in pn.items():
        lim = torch.tensor([bool(s) and s != "0" and int(s + str(d)) <= N or s == "" for d in range(10)])
        pn[s] = torch.where(lim, q, z)
    assert abs(float(trie_distribution(N, pn).sum()) - float(pn[""].sum())) < 1e-12
    pn[""] = pn[""] / pn[""].sum()
    assert abs(float(trie_distribution(N, pn).sum()) - 1) < 1e-12


@pytest.fixture(scope="module")
def tok():
    tr = pytest.importorskip("transformers")
    return tr.AutoTokenizer.from_pretrained(paths.get("new_models") / NAME)


@pytest.mark.parametrize("box", [[5, 12, 1003, 840], [1000, 0, 1148, 7], [1, 999, 1100, 10], [223, 772, 385, 873]])
def test_spans_with_four_digit(tok, box):
    text, rid, spans, digit = qwen25vl_encode(tok, "bottle of orange juice 2", box)
    assert [b - a for a, b in spans] == [len(str(v)) for v in box]
    for (a, b), v in zip(spans, box):
        assert tok.decode(rid[a:b]) == str(v)
        assert tok.decode(rid[a - 1:a]).endswith((" ", "["))       # 数字前面是完整的空格 / 括号 token，树的前缀不含数字
        assert rid[b] not in digit
    assert tok.decode(rid) == text


def test_span_helper():
    assert coord_token_spans([9, 1, 2, 9, 3, 9, 9, 4, 5, 6, 9, 1, 9], [1, 2, 3, 4, 5, 6], 0) == [(1, 3), (4, 5), (7, 10), (11, 12)]


def test_response_format_is_the_models_own(tok):
    f = paths.get("outputs") / "smoke" / NAME / "default_vs_eager.json"
    if not f.is_file():
        pytest.skip("没有冒烟输出")
    import re
    runs = json.load(open(f))["runs"]["observed"]
    n = 0
    for r in runs:
        if r["task"] != "bbox":
            continue
        body = r["text"].replace("<|im_end|>", "")
        box = [int(x) for x in re.search(r"\[(\d+), (\d+), (\d+), (\d+)\]", body).groups()]
        label = re.search(r'"label": "([^"]*)"', body).group(1)
        assert qwen25vl_response(label, box) == body
        assert qwen25vl_encode(tok, label, box)[1] + [tok.convert_tokens_to_ids("<|im_end|>")] == r["tokens"]
        n += 1
    assert n == 40


def test_tiny_random_model_two_paths_agree(tok):
    """真实处理器 + 随机小模型（float32，CPU）：lp_tree 与 lp_digits(_end) 一致；embeds / 位置拆分与官方 forward 一致；chunk 大小不改变结果。"""
    tr = pytest.importorskip("transformers")
    from types import SimpleNamespace

    from PIL import Image

    from src.models.adapter import HFChatAdapter
    from src.readout import Qwen25VLBoxReadout
    p = paths.get("new_models") / NAME
    cfg = tr.AutoConfig.from_pretrained(p)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers, t.num_attention_heads, t.num_key_value_heads = 96, 64, 2, 4, 2
    if hasattr(t, "layer_types") and t.layer_types is not None:
        t.layer_types = t.layer_types[:2]
    t.rope_parameters = {**t.rope_parameters, "mrope_section": [4, 4, 4]}
    v.hidden_size, v.intermediate_size, v.depth, v.num_heads, v.out_hidden_size = 64, 64, 2, 2, 96
    v.fullatt_block_indexes = [1]
    torch.manual_seed(0)
    m = tr.Qwen2_5_VLForConditionalGeneration._from_config(cfg, dtype=torch.float32, attn_implementation="eager").eval()
    ad = object.__new__(HFChatAdapter)
    ad.name, ad.model, ad.spec = NAME, m, {"grid": "qwen_thw", "max_pixels": 100352, "chat_kwargs": {}}
    ad.processor = tr.AutoProcessor.from_pretrained(p)
    ad.tokenizer, ad.image_token_id = ad.processor.tokenizer, cfg.image_token_id
    import numpy as np
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (960, 1280, 3), dtype=np.uint8))
    prep = ad.prepare(img, "Locate cup, output its bbox coordinates using JSON format.")
    prep = SimpleNamespace(inputs=prep.inputs, model_wh=(1148, 840), grid_hw=prep.grid_hw)     # 只影响枚举上限
    outs = [Qwen25VLBoxReadout(ad, c).run(prep, "cup", box, official_check=True) for c in (10, 5) for box in ([1003, 12, 1148, 840], [104, 5, 250, 1000])]
    for o in outs:
        assert o["official_forward_max_abs_logit_diff"] < 1e-5
        for k in ("native", "headfp32"):
            r = o["readout"][k]
            for lt, ld, le in zip(r["lp_tree"], r["lp_digits"], r["lp_digits_end"]):
                assert abs(lt - (ld if le is None else le)) < 1e-3
    for a, b in zip(outs[:2], outs[2:]):
        assert all(abs(x - y) < 1e-2 for x, y in zip(a["readout"]["native"]["e_tree"], b["readout"]["native"]["e_tree"]))
