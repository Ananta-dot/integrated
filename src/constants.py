# src/constants.py

# Special tokens
SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}
BASE_VOCAB = 3             # labels start here as BASE_VOCAB + (i-1)
MAX_N = 128                # safety ceiling

# n-conditioning tokens: one token per n in [1..MAX_N]
# These are used with nanoGPT since it doesn't take auxiliary embeddings.
N_BASE = BASE_VOCAB + MAX_N            # start index for <N_n> tokens
VOCAB_SIZE = BASE_VOCAB + MAX_N + MAX_N  # [BOS,SEP,EOS] + labels + N_tokens

def n_token(n: int) -> int:
    assert 1 <= n <= MAX_N
    return N_BASE + (n - 1)
