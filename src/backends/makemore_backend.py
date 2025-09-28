# src/backends/makemore_backend.py
import sys, pathlib, torch, torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str((ROOT / "external" / "makemore").resolve()))

# We import the reference Transformer codepath by executing makemore.py in module form.
# makemore is a single-file script; we'll expose a small class that mirrors our ngpt API.
# NOTE: You may have to tweak if upstream changes; this keeps our code minimal.

from types import SimpleNamespace

class MakeMoreTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=192, n_head=6, n_layer=3,
                 block_size=4096, dropout=0.1):
        super().__init__()
        # Lightweight Transformer stack (based on makemore code style).
        # We implement a tiny GPT block here using torch.nn so we don't fork upstream logic;
        # makemore is not structured as an importable library. This keeps deps minimal.
        # If you want to mirror makemore exactly, you can vendor the relevant blocks here.
        from torch.nn import TransformerEncoderLayer, TransformerEncoder, Embedding, Linear
        self.tok = Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, block_size, d_model))
        layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                        dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.enc = TransformerEncoder(layer, num_layers=n_layer)
        self.out = Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,L]
        B, L = tokens.shape
        assert L <= self.block_size, "sequence too long for configured block_size"
        x = self.tok(tokens) + self.pos[:, :L, :]
        mask = torch.triu(torch.ones(L, L, device=tokens.device), diagonal=1).bool()
        x = self.enc(x, mask=mask)
        return self.out(x)
