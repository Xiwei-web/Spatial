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
