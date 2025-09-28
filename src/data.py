# src/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch, random

Seq = List[int]
Instance = Tuple[Seq, Seq]

SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}
BASE_VOCAB = 3
MAX_N = 128

def seq_to_tokens(seq: Seq) -> List[int]:
    return [BASE_VOCAB + (i-1) for i in seq]

def tokens_to_seq(tokens: List[int]) -> Seq:
    return [t - BASE_VOCAB + 1 for t in tokens]

@dataclass
class Batch:
    tokens: torch.Tensor
    targets: torch.Tensor

def make_batch(elites, B: int, device: torch.device, rng: random.Random) -> Batch:
    tlist=[]; tglist=[]
    for _ in range(B):
        _, H, V = rng.choice(elites)
        tok = [SPECIAL["BOS"]] + seq_to_tokens(H) + [SPECIAL["SEP"]] + seq_to_tokens(V) + [SPECIAL["EOS"]]
        tgt = tok[1:] + [SPECIAL["EOS"]]
        tlist.append(torch.tensor(tok, dtype=torch.long))
        tglist.append(torch.tensor(tgt, dtype=torch.long))
    L = max(len(t) for t in tlist); pad = SPECIAL["EOS"]
    tokens = torch.full((B, L), pad, dtype=torch.long)
    targets = torch.full((B, L), pad, dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)] = t; targets[i,:len(tt)] = tt
    return Batch(tokens=tokens.to(device), targets=targets.to(device))
