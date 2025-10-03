# src/backends/nanogpt_adapter.py
from __future__ import annotations
import os, sys, importlib.util
from typing import List, Tuple
import torch
import torch.nn.functional as F

from src.constants import SPECIAL, BASE_VOCAB, MAX_N, N_BASE, VOCAB_SIZE, n_token

Seq = List[int]
Instance = Tuple[Seq, Seq]

# ---- Robust import of nanoGPT's model.py by file path (avoid name collisions) ----
NANOGPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/nanogpt"))
NANOGPT_MODEL_PY = os.path.join(NANOGPT_ROOT, "model.py")
if not os.path.isfile(NANOGPT_MODEL_PY):
    raise FileNotFoundError(f"nanoGPT model.py not found at {NANOGPT_MODEL_PY}")

_spec = importlib.util.spec_from_file_location("nanogpt_model", NANOGPT_MODEL_PY)
_nanogpt_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_nanogpt_mod)
GPT = _nanogpt_mod.GPT
GPTConfig = _nanogpt_mod.GPTConfig
# -----------------------------------------------------------------------------

class NanoGPTWrapper(torch.nn.Module):
    """
    Thin wrapper around nanoGPT's GPT to present a stable interface that always
    returns logits of shape [B, T, VOCAB_SIZE].
    """
    def __init__(self, n_layer=3, n_head=6, n_embd=192, block_size=4096, dropout=0.1, device="cpu"):
        super().__init__()
        cfg = GPTConfig(
            vocab_size=VOCAB_SIZE,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=False,
        )
        self.gpt = GPT(cfg).to(device)
        self.device = device

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T] long
        returns logits: [B, T, VOCAB_SIZE]
        """
        out = self.gpt(tokens)
        # Some variants may return a tuple. Normalize:
        if isinstance(out, tuple):
            out = out[0]
        # Safety: ensure 3D
        if out.dim() == 2:
            # If a 2D [B, VOCAB] ever appears (due to some unexpected variant),
            # expand a time dimension so downstream logic doesn't break.
            out = out.unsqueeze(1)  # [B, 1, V]
        return out

def seq_to_tokens(seq: Seq) -> List[int]:
    return [BASE_VOCAB + (i - 1) for i in seq]

def tokens_to_seq(tokens: List[int]) -> Seq:
    out = []
    for t in tokens:
        if BASE_VOCAB <= t < BASE_VOCAB + MAX_N:
            out.append(t - BASE_VOCAB + 1)
    return out

def make_batch(elites, B: int, device, rng) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tokens  = [<N_n>, BOS] + H(2n) + [SEP] + V(2n) + [EOS]
    targets = next-token shift of tokens (same length), EOS used as ignore_index for padding tail.
    """
    tlist=[]; tglist=[]
    for _ in range(B):
        _, H, V = rng.choice(elites)
        n = max(H) if H else 1
        seq = [n_token(n), SPECIAL["BOS"]] + seq_to_tokens(H) + [SPECIAL["SEP"]] + seq_to_tokens(V) + [SPECIAL["EOS"]]
        tgt = seq[1:] + [SPECIAL["EOS"]]
        tlist.append(torch.tensor(seq, dtype=torch.long))
        tglist.append(torch.tensor(tgt, dtype=torch.long))
    L = max(len(t) for t in tlist)
    pad = SPECIAL["EOS"]
    tokens  = torch.full((B, L), pad, dtype=torch.long)
    targets = torch.full((B, L), pad, dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)]  = t
        targets[i,:len(tt)] = tt
    return tokens.to(device), targets.to(device)

@torch.no_grad()
def sample_model(model: NanoGPTWrapper, n: int, device,
                 temperature: float = 1.0, top_p: float = 0.9, max_len: int = 4096) -> Instance:
    """
    Sample:
      [<N_n>, BOS] + H(2n) + SEP + V(2n) + EOS
    with legality masks (each label <=2 per half).
    """
    import torch.nn.functional as F
    prefix = [n_token(n), SPECIAL["BOS"]]
    toks: List[int] = prefix[:]

    def step(valid_ids: List[int]) -> int:
        x = torch.tensor(toks, dtype=torch.long, device=device)[None, :]
        logits = model(x)[:, -1, :].squeeze(0)   # [VOCAB]
        mask = torch.full_like(logits, -1e9)
        mask.scatter_(0, torch.tensor(valid_ids, device=device), 0.0)
        probs = F.softmax((logits + mask)/temperature, dim=-1)
        # top-p
        sorted_p, idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_p, dim=-1)
        keep = csum <= top_p
        if not torch.any(keep): keep[0] = True
        p = torch.zeros_like(probs).scatter(0, idx[keep], sorted_p[keep])
        p = p / p.sum()
        return int(torch.multinomial(p, 1).item())

    # H half
    cnt = [0]*(n+1)
    while len(toks) < max_len and sum(cnt) < 2*n:
        valid = [BASE_VOCAB + (i-1) for i in range(1, n+1) if cnt[i] < 2]
        nxt = step(valid)
        toks.append(nxt)
        lab = nxt - BASE_VOCAB + 1
        cnt[lab] += 1

    # SEP
    toks.append(SPECIAL["SEP"])

    # V half
    cnt = [0]*(n+1)
    while len(toks) < max_len and sum(cnt) < 2*n:
        valid = [BASE_VOCAB + (i-1) for i in range(1, n+1) if cnt[i] < 2]
        nxt = step(valid)
        toks.append(nxt)
        lab = nxt - BASE_VOCAB + 1
        cnt[lab] += 1

    toks.append(SPECIAL["EOS"])

    # split back
    try:
        sep_idx = toks.index(SPECIAL["SEP"])
    except ValueError:
        return [], []
    H_tok = toks[2:sep_idx]
    V_tok = toks[sep_idx+1:-1]
    return tokens_to_seq(H_tok), tokens_to_seq(V_tok)

# add at top-level in this file:
_first_shape_warn_emitted = False

def _normalize_logits_to_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Ensure logits are [B, T, V] and T matches targets.shape[1].
    If logits come as [B, V] or with mismatched T, fix by repeat/crop/pad.
    """
    global _first_shape_warn_emitted
    if isinstance(logits, tuple):
        logits = logits[0]

    # Make 3D
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # [B, 1, V]

    if logits.dim() != 3:
        raise RuntimeError(f"Expected logits to be 3D [B,T,V], got {tuple(logits.shape)}")

    B_l, T_l, V = logits.shape
    B_t, T_t = targets.shape[:2]

    # Batch must match; if not, that's a real error
    if B_l != B_t:
        raise RuntimeError(f"Batch mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}")

    if T_l == T_t:
        return logits  # already aligned

    # One-time debug print to help diagnose upstream issues
    if not _first_shape_warn_emitted:
        print(f"[warn] Logits time dim != targets: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}. "
              f"Reconciling by repeat/crop/pad.")
        _first_shape_warn_emitted = True

    if T_l == 1 and T_t > 1:
        # last-step logits only; repeat across time
        logits = logits.repeat(1, T_t, 1)  # [B, T_t, V]
        return logits

    if T_l > T_t:
        # too long; crop
        logits = logits[:, :T_t, :]
        return logits

    # T_l < T_t and T_l != 1: pad by repeating last step
    pad_steps = T_t - T_l
    last_step = logits[:, -1:, :].repeat(1, pad_steps, 1)
    logits = torch.cat([logits, last_step], dim=1)
    return logits


def train_one_step(model: NanoGPTWrapper, opt: torch.optim.Optimizer,
                   tokens: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Safe training step that:
    - forces logits to [B,T,V] and aligns T with targets,
    - uses EOS as ignore_index (for padding tail).
    """
    model.train()
    out = model(tokens)
    if isinstance(out, tuple):
        out = out[0]
    logits = _normalize_logits_to_targets(out, targets)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=SPECIAL["EOS"]
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())

