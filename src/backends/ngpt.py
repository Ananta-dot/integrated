# src/backends/ngpt.py
import sys, pathlib, torch, torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str((ROOT / "external" / "nanogpt").resolve()))
from model import GPT, GPTConfig  # from nanoGPT repo

class NanoGPT(nn.Module):
    """
    Minimal wrapper so caller uses: logits = model(tokens)
    """
    def __init__(self, vocab_size: int, d_model=192, n_head=6, n_layer=3,
                 block_size=4096, dropout=0.1):
        super().__init__()
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=d_model,
            dropout=dropout,
            bias=True,
        )
        self.gpt = GPT(cfg)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gpt(tokens, targets=None)
        return logits
