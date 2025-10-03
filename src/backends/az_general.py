# src/backends/az_general.py
from __future__ import annotations
import os, sys
from typing import List, Tuple
import numpy as np
import random

Seq = List[int]
Instance = Tuple[Seq, Seq]

# Optional import of AlphaZero-General
AZ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/alpha-zero-general"))
if AZ_ROOT not in sys.path:
    sys.path.insert(0, AZ_ROOT)

try:
    from MCTS import MCTS  # alpha-zero-general
    AZ_AVAILABLE = True
except Exception:
    AZ_AVAILABLE = False


class MISRBuildGame:
    """
    Stateless AlphaZero 'Game' for building (H,V) label sequences for MISR.
    Board representation (numpy int16, variable length):
        board = np.array( H + [-1] + V, dtype=np.int16 )
    where H and V are sequences of labels in 1..n, each repeated at most twice,
    and -1 is a separator. The process ends when |H| == 2n and |V| == 2n.
    """

    def __init__(self, n: int):
        self.n = n
        self.max_per = 2

    # ---- Required API ----
    def getInitBoard(self):
        # Start with just the separator
        return np.array([-1], dtype=np.int16)

    def getBoardSize(self):
        # Not strictly used by core MCTS for variable-length boards; return a dummy
        return (1,)

    def getActionSize(self):
        # Actions are labels 1..n (we map action index 0..n-1 -> label=idx+1)
        return self.n

    def getNextState(self, board, player, action_idx):
        """
        action_idx in [0..n-1] -> label = action_idx+1
        If H not full (len(H) < 2n), append to H; else append to V.
        Return (new_board, same_player)
        """
        label = int(action_idx + 1)
        H, V = self._split(board)
        if len(H) < 2 * self.n:
            H = H + [label]
        else:
            V = V + [label]
        new_board = np.array(H + [-1] + V, dtype=np.int16)
        return new_board, player

    def getValidMoves(self, board, player):
        """
        Return a binary vector of length n: which labels (1..n) can be played now
        (i.e., count in current half < 2).
        """
        H, V = self._split(board)
        half = H if len(H) < 2 * self.n else V
        cnt = [0] * (self.n + 1)
        for x in half:
            cnt[x] += 1
        v = np.zeros(self.n, dtype=np.int8)
        for i in range(1, self.n + 1):
            if cnt[i] < self.max_per:
                v[i - 1] = 1
        return v

    def getGameEnded(self, board, player):
        """
        Return 1 when terminal, else 0. (AlphaZero-General uses -1/1 for winners;
        here it's a construction task, so we just return 1 for 'ended'.)
        """
        H, V = self._split(board)
        done = (len(H) == 2 * self.n and len(V) == 2 * self.n)
        return 1 if done else 0

    def getCanonicalForm(self, board, player):
        # No symmetries; return as-is
        return board

    def stringRepresentation(self, board) -> str:
        # Used as a dict key inside MCTS; variable-length safe
        return board.tobytes()

    # ---- Helpers ----
    def _split(self, board) -> Tuple[Seq, Seq]:
        arr = np.asarray(board, dtype=np.int16)
        # Find separator (-1). If missing (shouldn't happen), treat all as H.
        sep_idx = np.where(arr == -1)[0]
        if len(sep_idx) == 0:
            H = arr.tolist()
            V = []
        else:
            s = int(sep_idx[0])
            H = arr[:s].tolist()
            V = arr[s + 1 :].tolist()
        return H, V

    def decode(self, board) -> Instance:
        H, V = self._split(board)
        return H[:], V[:]


class DummyNet:
    """
    Minimal policy/value net that returns uniform policy and zero value.
    AlphaZero-General expects a .predict(canonicalBoard) -> (pi, v).
    """
    def __init__(self, n: int):
        self.n = n
    def predict(self, canonicalBoard):
        p = np.ones(self.n, dtype=np.float32) / self.n
        v = 0.0
        return p, v


def propose_with_alphazero(n: int, k: int, rng: random.Random) -> List[Instance]:
    """
    Generate k (H,V) proposals using AlphaZero MCTS rollouts.
    If alpha-zero-general is not present, return [].
    """
    if not AZ_AVAILABLE or k <= 0:
        return []

    game = MISRBuildGame(n)
    net = DummyNet(n)
    # Typical small MCTS budget; adjust if needed
    args = type("Args", (), {"numMCTSSims": 64, "cpuct": 1.5})
    mcts = MCTS(game, net, args=args)

    out: List[Instance] = []
    for _ in range(k):
        board = game.getInitBoard()
        # roll until terminal
        while game.getGameEnded(board, 1) == 0:
            valid = game.getValidMoves(board, 1)               # [n] 0/1
            pi = mcts.getActionProb(game.getCanonicalForm(board, 1), temp=1)  # [n]
            pi = np.asarray(pi, dtype=np.float32)

            # Mask invalids
            pi *= valid.astype(np.float32)
            s = float(pi.sum())
            if s <= 1e-8:
                # fallback uniform over valid moves
                valids_idx = np.where(valid > 0)[0]
                a_idx = int(rng.choice(valids_idx))
            else:
                pi /= s
                a_idx = int(np.random.choice(np.arange(game.getActionSize()), p=pi))
            board, _ = game.getNextState(board, 1, a_idx)

        H, V = game.decode(board)
        out.append((H, V))

    return out
