"""框的 teacher-forced 读出（目前只支持 InternVL 远程代码系；坐标约定 norm1000_xyxy）。

坐标在该分词器下是逐位数字 token（"775" → 7,7,5），没有单个的“坐标 token”，因此定义三个读出量，都以真值的前缀作为输入：
  lp_digits[c]   坐标 c 的真值各位数字的对数概率之和（一次完整前向，无 KV cache）
  e_chain[c]     逐位条件期望：sum_k 10^(n-1-k) * E[d_k | 真值前缀与真值的前几位]，E 在 10 个数字内归一（同一次前向；位数取真值的位数）
  e_tree[c]      精确期望坐标：枚举 0..1000 的全部数字串（KV cache + 批量分支），前面的坐标用真值；
                 P(n) = 各位数字概率之积 × 结束概率（1 - 下一位仍是数字的概率），在合法串内归一，归一前的总质量记为 mass_tree
  lp_tree[c]     枚举得到的 log P(真值数字串)（未归一）；真值为三位数且不是 100 时应与 lp_digits 相同，用作两条计算路径的互检
logits 两种算法：native = 模型自带 lm_head（bf16 模型里是 bf16 的 logits，有量化）；headfp32 = 同一个末层隐状态，lm_head 在 float32 里算。"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


@torch.no_grad()
def internvl_embeds(ad, prep):
    """与远程代码 generate 相同的 inputs_embeds 构造方式（多卡切分时也可用；远程 forward 在多卡下会因跨设备索引报错）。"""
    m = ad.model
    e = m.language_model.get_input_embeddings()(prep.inputs["input_ids"]).clone()
    v = m.extract_feature(prep.inputs["pixel_values"])
    sel = prep.inputs["input_ids"] == m.img_context_token_id
    e[sel] = v.reshape(-1, e.shape[-1]).to(e.device, e.dtype)
    return e


class InternVLBoxReadout:
    HEADS = ("native", "headfp32")

    def __init__(self, ad):
        self.ad, self.lm = ad, ad.model.language_model
        self.W = self.lm.lm_head.weight.detach().float()
        assert self.lm.lm_head.bias is None
        tok = ad.tokenizer
        self.digit = []
        for d in range(10):
            ids = tok.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1
            self.digit.append(ids[0])
        self.digit_t = torch.tensor(self.digit)

    def _hidden(self, embeds, cache=None, n_past=0, use_cache=False):
        mask = torch.ones(embeds.shape[0], n_past + embeds.shape[1], dtype=torch.long, device=embeds.device)
        out = self.lm.model(inputs_embeds=embeds, attention_mask=mask, past_key_values=cache, use_cache=use_cache, return_dict=True)
        return out.last_hidden_state, out.past_key_values

    def _logprobs(self, h):
        """h: (..., hidden) → {head: log_softmax（float32，CPU）}"""
        nat = self.lm.lm_head(h).float().log_softmax(-1).cpu()
        f32 = F.linear(h.to(self.W.device).float(), self.W).log_softmax(-1).cpu()
        return {"native": nat, "headfp32": f32}

    @torch.no_grad()
    def run(self, prep, obj: str, box1000: list[int]) -> dict:
        tok, emb = self.ad.tokenizer, self.lm.get_input_embeddings()
        resp = f"<ref>{obj}</ref><box>[[" + ", ".join(str(int(v)) for v in box1000) + "]]</box>"
        rid = tok.encode(resp, add_special_tokens=False)
        # 定位 4 个坐标各自的数字 token 区间
        start = rid.index(tok.encode("[[", add_special_tokens=False)[0]) + 1
        spans, i = [], start
        while len(spans) < 4:
            j = i
            while rid[j] in self.digit:
                j += 1
            assert j > i, (resp, i)
            spans.append((i, j))
            i = j
            while len(spans) < 4 and rid[i] not in self.digit:
                i += 1
        assert [int("".join(str(self.digit.index(t)) for t in rid[a:b])) for a, b in spans] == [int(v) for v in box1000]
        base = internvl_embeds(self.ad, prep)
        n0 = base.shape[1]
        ids = torch.tensor([rid], device=base.device)
        full = torch.cat([base, emb(ids).to(base.device, base.dtype)], 1)
        h, _ = self._hidden(full)
        out = {k: {"lp_digits": [], "e_chain": [], "e_tree": [], "mass_tree": [], "lp_tree": [], "lp_digits_each": []} for k in self.HEADS}
        place = lambda n: [10 ** (n - 1 - k) for k in range(n)]
        for a, b in spans:
            lps = self._logprobs(h[0, n0 + a - 1: n0 + b - 1])          # 预测第 a..b-1 个 token 的位置
            for k in self.HEADS:
                each = [float(lps[k][q, rid[a + q]]) for q in range(b - a)]
                p = lps[k][:, self.digit_t].softmax(-1)
                ed = (p * torch.arange(10.)).sum(-1)
                out[k]["lp_digits_each"].append(each)
                out[k]["lp_digits"].append(sum(each))
                out[k]["e_chain"].append(float((ed * torch.tensor(place(b - a), dtype=torch.float)).sum()))
        for (a, _), v in zip(spans, box1000):
            t = self._tree(full[:, : n0 + a], int(v))
            for k in self.HEADS:
                out[k]["e_tree"].append(t[k][0])
                out[k]["mass_tree"].append(t[k][1])
                out[k]["lp_tree"].append(t[k][2])
        return {"response": resp, "n_input": n0, "readout": out}

    def _branch(self, cache, n_past, seqs):
        emb = self.lm.get_input_embeddings()
        c = copy.deepcopy(cache)
        c.batch_repeat_interleave(len(seqs))
        ids = torch.tensor([[self.digit[d] for d in s] for s in seqs], device=emb.weight.device)
        h, _ = self._hidden(emb(ids), c, n_past, use_cache=True)
        return {k: v[:, self.digit_t].double().exp() for k, v in self._logprobs(h[:, -1]).items()}   # (B, 10)

    def _tree(self, prefix, truth):
        L = prefix.shape[1]
        h, cache = self._hidden(prefix, use_cache=True)
        p1 = {k: v[0, self.digit_t].double().exp() for k, v in self._logprobs(h[:, -1]).items()}
        b1 = self._branch(cache, L, [(d,) for d in range(10)])
        two = [(x, y) for x in range(1, 10) for y in range(10)]
        b2 = self._branch(cache, L, two)
        b3 = self._branch(cache, L, [(1, 0, 0)])
        res = {}
        for k in self.HEADS:
            P = torch.zeros(1001, dtype=torch.double)
            for d in range(10):
                P[d] = p1[k][d] * (1 - b1[k][d].sum())
            for r, (x, y) in enumerate(two):
                q = p1[k][x] * b1[k][x, y]
                P[10 * x + y] = q * (1 - b2[k][r].sum())
                for z in range(10):
                    P[100 * x + 10 * y + z] = q * b2[k][r, z]
            P[1000] = P[100] * b3[k][0, 0]
            P[100] = P[100] * (1 - b3[k][0].sum())
            P = P.clamp_min(0)
            m = P.sum()
            res[k] = (float((P * torch.arange(1001, dtype=torch.double)).sum() / m), float(m), float(P[truth].clamp_min(1e-300).log()))
        return res


# ---------------------------------------------------------------------------------------------------------------
# Qwen2.5-VL（坐标为模型输入像素的绝对值，可能有四位数）。上面 InternVL 的类与数值路径不改动。

def digit_trie(max_value: int) -> dict[int, list[str]]:
    """枚举 0..max_value 的十进制串（无前导零）所需的前缀：{长度: [前缀…]}，需要对每个前缀求“下一位数字”的分布。
    包含：全部一位数（0 也求，用于它的结束概率）、以及能再接一位仍 ≤ max_value 的更长前缀。max_value=1000 时与 InternVL 的
    枚举相同（10 + 90 + 1 个）。"""
    out: dict[int, list[str]] = {1: [str(d) for d in range(10)]}
    n = 1
    while True:
        nxt = [p + str(d) for p in out[n] if p != "0" for d in range(10) if int(p + str(d)) <= max_value]
        nxt = [p for p in nxt if int(p) * 10 <= max_value]
        if not nxt:
            return out
        n += 1
        out[n] = nxt


def trie_distribution(max_value: int, p_next: dict[str, torch.Tensor]) -> torch.Tensor:
    """p_next[前缀] = 10 维下一位数字概率（前缀 "" 为首位）。返回 P(0..max_value)，未归一。
    可以再接一位的串乘以结束概率（1 − 下一位仍是数字的总概率）；其余（不在 p_next 里的）不乘。"""
    P = torch.zeros(max_value + 1, dtype=torch.double)
    stack = [("", 1.0)]
    while stack:
        s, w = stack.pop()
        q = p_next.get(s)
        if s:
            P[int(s)] = w * (1 - float(q.sum())) if q is not None else w
        if q is None or s == "0":
            continue
        for d in range(10):
            if int(s + str(d)) <= max_value:
                stack.append((s + str(d), w * float(q[d])))
    return P.clamp_min(0)


def coord_token_spans(rid: list[int], digit_ids: list[int], start: int, n: int = 4) -> list[tuple[int, int]]:
    """从 start 起依次取 n 个连续数字 token 区间 [a, b)。"""
    isd, spans, i = set(digit_ids), [], start
    while len(spans) < n:
        while rid[i] not in isd:
            i += 1
        j = i
        while rid[j] in isd:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def qwen25vl_response(obj: str, box: list[int]) -> str:
    """与该模型自由生成的 40 条输出同一格式（带 markdown 围栏的 JSON 列表）。"""
    return '```json\n[\n\t{"bbox_2d": [' + ", ".join(str(int(v)) for v in box) + f'], "label": "{obj}"}}\n]\n```'


def qwen25vl_encode(tok, obj: str, box: list[int]):
    """→ (文本, token 序列, 4 个坐标的 token 区间)。“"bbox_2d"”里的 2 也是数字 token，所以从键名之后的 “[” 开始找。"""
    text = qwen25vl_response(obj, box)
    rid = tok.encode(text, add_special_tokens=False)
    digit = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(x) == 1 for x in digit)
    digit = [x[0] for x in digit]
    key = text.index('"bbox_2d": [') + len('"bbox_2d": [')
    start = next(i for i in range(len(rid)) if len(tok.decode(rid[: i + 1])) >= key)      # 含 “[” 的 token 的下标
    spans = coord_token_spans(rid, digit, start)
    got = [int("".join(str(digit.index(t)) for t in rid[a:b])) for a, b in spans]
    assert got == [int(v) for v in box], (text, got)
    return text, rid, spans, digit


class _Stop(Exception):
    pass


class Qwen25VLBoxReadout:
    """读出量与 InternVLBoxReadout 同名同义（见文件开头），差别：枚举 0..max(模型输入宽, 高)；分支按 chunk 个一批跑；
    inputs_embeds 与三维位置由官方 forward 给出（在进入语言模型前截获），回答部分的位置 = 提示最后位置 + 1 + i。
    另有 lp_digits_end：lp_digits 再加上“到此结束”的对数概率（只对枚举里带结束概率的坐标，即可以再接一位的数；其余为 None）。"""
    HEADS = ("native", "headfp32")

    def __init__(self, ad, chunk: int = 10):
        import inspect
        self.ad, self.chunk = ad, chunk
        self.top, self.lm = ad.model.model, ad.model.model.language_model
        self.head = ad.model.lm_head
        assert self.head.bias is None
        self.W = self.head.weight.detach().float()
        self._cache_kw = "cache" if "cache" in inspect.signature(self.lm.forward).parameters else "past_key_values"

    @torch.no_grad()
    def prompt_embeds(self, prep):
        got = {}

        def grab(mod, args, kwargs):
            got.update(e=kwargs["inputs_embeds"], p=kwargs["position_ids"])
            raise _Stop

        h = self.lm.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            self.top(**prep.inputs, use_cache=False)
        except _Stop:
            pass
        finally:
            h.remove()
        assert got["p"] is not None and got["p"].shape[0] == 3 and bool((got["p"][:, 0, -1] == got["p"][0, 0, -1]).all())
        return got["e"], got["p"]

    @staticmethod
    def _extend(pos, n):
        step = torch.arange(1, n + 1, device=pos.device).view(1, 1, -1).expand(3, pos.shape[1], n)
        return torch.cat([pos, pos[:, :, -1:] + step], -1)

    def _hidden(self, embeds, pos, cache=None, n_past=0, use_cache=False):
        mask = torch.ones(embeds.shape[0], n_past + embeds.shape[1], dtype=torch.long, device=embeds.device)
        out = self.lm(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=use_cache, **{self._cache_kw: cache})
        return out.last_hidden_state, out[self._cache_kw] if use_cache else None

    def _logprobs(self, h):
        nat = self.head(h).float().log_softmax(-1).cpu()
        f32 = F.linear(h.to(self.W.device).float(), self.W).log_softmax(-1).cpu()
        return {"native": nat, "headfp32": f32}

    @torch.no_grad()
    def run(self, prep, obj: str, box: list[int], official_check: bool = False) -> dict:
        tok = self.ad.tokenizer
        emb = self.lm.get_input_embeddings()
        N = max(prep.model_wh)
        resp, rid, spans, self.digit = qwen25vl_encode(tok, obj, box)
        self.digit_t = torch.tensor(self.digit)
        base, pos0 = self.prompt_embeds(prep)
        n0 = base.shape[1]
        ids = torch.tensor([rid], device=base.device)
        full = torch.cat([base, emb(ids).to(base.device, base.dtype)], 1)
        pos = self._extend(pos0, len(rid))
        h, _ = self._hidden(full, pos)
        out = {k: {"lp_digits": [], "lp_digits_end": [], "e_chain": [], "e_tree": [], "mass_tree": [], "lp_tree": [], "lp_digits_each": []} for k in self.HEADS}
        res = {"response": resp, "n_input": n0, "n_response": len(rid), "max_value": N, "model_wh": list(prep.model_wh), "grid_hw": list(prep.grid_hw), "readout": out}
        if official_check:                         # 与官方完整 forward（不拆 embeds / 位置）比回答部分的 logits
            inp = dict(prep.inputs)
            inp["input_ids"] = torch.cat([inp["input_ids"], ids.to(inp["input_ids"].device)], 1)
            for k, v in (("attention_mask", 1), ("mm_token_type_ids", 0)):
                inp[k] = torch.cat([inp[k], torch.full((1, len(rid)), v, dtype=inp[k].dtype, device=inp[k].device)], 1)
            ho = self.top(**inp, use_cache=False).last_hidden_state
            res["official_forward_max_abs_logit_diff"] = float((self.head(ho[0, n0 - 1:]).float() - self.head(h[0, n0 - 1:]).float()).abs().max())
        place = lambda n: [10 ** (n - 1 - k) for k in range(n)]      # noqa: E731
        ends = {p for v in digit_trie(N).values() for p in v}
        for (a, b), v in zip(spans, box):
            lps = self._logprobs(h[0, n0 + a - 1: n0 + b])           # 最后一行是数字串之后那个位置
            for k in self.HEADS:
                each = [float(lps[k][q, rid[a + q]]) for q in range(b - a)]
                p = lps[k][: b - a, self.digit_t].softmax(-1)
                ed = (p * torch.arange(10.)).sum(-1)
                out[k]["lp_digits_each"].append(each)
                out[k]["lp_digits"].append(sum(each))
                pe = 1 - float(lps[k][b - a, self.digit_t].double().exp().sum())
                out[k]["lp_digits_end"].append(sum(each) + float(torch.tensor(pe).clamp_min(1e-300).log()) if str(int(v)) in ends else None)
                out[k]["e_chain"].append(float((ed * torch.tensor(place(b - a), dtype=torch.float)).sum()))
        for (a, _), v in zip(spans, box):
            t = self._tree(full[:, : n0 + a], pos[:, :, : n0 + a], int(v), N)
            for k in self.HEADS:
                for m in ("e_tree", "mass_tree", "lp_tree"):
                    out[k][m].append(t[k][m])
        return res

    def _tree(self, prefix, pos, truth, N):
        emb = self.lm.get_input_embeddings()
        L = prefix.shape[1]
        h, cache = self._hidden(prefix, pos, use_cache=True)
        first = self._logprobs(h[:, -1])
        pn = {k: {"": v[0, self.digit_t].double().exp()} for k, v in first.items()}
        for n, prefs in digit_trie(N).items():
            for i in range(0, len(prefs), self.chunk):
                part = prefs[i: i + self.chunk]
                c = copy.deepcopy(cache)
                c.batch_repeat_interleave(len(part))
                ids = torch.tensor([[self.digit[int(ch)] for ch in s] for s in part], device=prefix.device)
                p = self._extend(pos[:, :, -1:], n)[:, :, 1:].expand(3, len(part), n)
                hb, _ = self._hidden(emb(ids).to(prefix.dtype), p, c, L, use_cache=True)
                del c
                for k, v in self._logprobs(hb[:, -1]).items():
                    for r, s in enumerate(part):
                        pn[k][s] = v[r, self.digit_t].double().exp()
        res = {}
        for k in self.HEADS:
            P = trie_distribution(N, pn[k])
            m = P.sum()
            res[k] = {"e_tree": float((P * torch.arange(N + 1, dtype=torch.double)).sum() / m), "mass_tree": float(m), "lp_tree": float(P[truth].clamp_min(1e-300).log())}
        return res
