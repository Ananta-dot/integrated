# src/sampler.py
from __future__ import annotations
from typing import List, Tuple
import torch, torch.nn.functional as F
from .data import SPECIAL, BASE_VOCAB, tokens_to_seq

Seq = List[int]
Instance = Tuple[Seq, Seq]

@torch.no_grad()
def sample_model(model, device: torch.device, n: int,
                 temperature: float = 1.0, top_p: float = 0.9,
                 max_len: int = 4096) -> Instance:
    model.eval()
    toks = [SPECIAL["BOS"]]

    def step(mask_valid: List[bool]) -> int:
        inp = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(inp)[0, -1]
        mask = torch.tensor(mask_valid, device=device)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits/temperature, dim=-1)
        sorted_probs, idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        keep = csum <= top_p
        if not torch.any(keep): keep[0] = True
        p = torch.zeros_like(probs).scatter(0, idx[keep], sorted_probs[keep])
        p = p / p.sum()
        return int(torch.multinomial(p, 1).item())

    counts = [0]*(n+1)
    while sum(counts) < 2*n and len(toks) < max_len:
        mask = [False]*(BASE_VOCAB + 128)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1

    toks.append(SPECIAL["SEP"])

    counts = [0]*(n+1)
    while sum(counts) < 2*n and len(toks) < max_len:
        mask = [False]*(BASE_VOCAB + 128)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1

    sep_idx = toks.index(SPECIAL["SEP"])
    H_tok = toks[1:sep_idx]
    V_tok = toks[sep_idx+1:]
    H = tokens_to_seq(H_tok); V = tokens_to_seq(V_tok)
    return H, V
