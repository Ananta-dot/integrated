# src/backends/az_general.py
from __future__ import annotations
import sys, pathlib, math, random, importlib.util
from typing import List, Tuple
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
azg_dir = (ROOT / "external" / "alpha-zero-general")

# Robust import of MCTS
try:
    sys.path.append(str(azg_dir.resolve()))
    from MCTS import MCTS  # type: ignore
except Exception:
    mcts_py = azg_dir / "MCTS.py"
    spec = importlib.util.spec_from_file_location("azg_MCTS", mcts_py)
    azg_MCTS = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(azg_MCTS)  # type: ignore
    MCTS = azg_MCTS.MCTS  # type: ignore

from ..data import SPECIAL, BASE_VOCAB, MAX_N, tokens_to_seq
from ..misr_core import score_ratio

Seq = List[int]
Instance = Tuple[Seq, Seq]

class MISRTokenGame:
    def __init__(self, n: int, vocab_size: int):
        self.n = n
        self.vocab_size = vocab_size
        self.sep_id = SPECIAL["SEP"]
        self.bos_id = SPECIAL["BOS"]
        self.eos_id = SPECIAL["EOS"]
        self.first_label = BASE_VOCAB
        self.last_label = BASE_VOCAB + n - 1
        self.target_len = 1 + 2*n + 1 + 2*n  # BOS + 2n + SEP + 2n

    def getInitBoard(self):
        return [self.bos_id]

    def getBoardSize(self):
        return (self.target_len,)

    def getActionSize(self):
        return self.vocab_size

    # AlphaZero-General expects this
    def getCanonicalForm(self, board, player):
        # single-agent; no change needed
        return board

    def _counts_half(self, seq_ids):
        if self.sep_id in seq_ids:
            start = seq_ids.index(self.sep_id) + 1
        else:
            start = 1  # after BOS
        counts = [0] * (self.n + 1)
        for t in seq_ids[start:]:
            if t >= BASE_VOCAB:
                lab = t - BASE_VOCAB + 1
                if 1 <= lab <= self.n:
                    counts[lab] += 1
        return counts

    def _len_half(self, seq_ids):
        if self.sep_id in seq_ids:
            start = seq_ids.index(self.sep_id) + 1
        else:
            start = 1
        length = 0
        for t in seq_ids[start:]:
            if t >= BASE_VOCAB:
                length += 1
        return length

    def getValidMoves(self, board, player):
        L = len(board)
        valid = np.zeros(self.getActionSize(), dtype=np.uint8)

        if L >= self.target_len:
            return valid  # no moves

        has_sep = (self.sep_id in board)
        if not has_sep:
            counts = self._counts_half(board)
            total_labels = self._len_half(board)
            if total_labels < 2*self.n:
                for lab in range(1, self.n+1):
                    if counts[lab] < 2:
                        valid[BASE_VOCAB + (lab-1)] = 1
            if total_labels == 2*self.n:
                valid[self.sep_id] = 1
        else:
            counts = self._counts_half(board)
            total_labels = self._len_half(board)
            if total_labels < 2*self.n:
                for lab in range(1, self.n+1):
                    if counts[lab] < 2:
                        valid[BASE_VOCAB + (lab-1)] = 1
            # terminal when target_len reached

        return valid

    def getNextState(self, board, player, action):
        new_board = list(board)
        new_board.append(action)
        return new_board, 1

    def getGameEnded(self, board, player):
        if len(board) < self.target_len:
            return 0
        try:
            sep_idx = board.index(self.sep_id)
        except ValueError:
            return -1  # invalid terminal -> bad outcome

        H_tok = board[1:sep_idx]
        V_tok = board[sep_idx+1:]
        H = tokens_to_seq(H_tok)
        V = tokens_to_seq(V_tok)

        lp, ilp, ratio, blended = score_ratio(H, V)
        # squash to [-1,1] to fit AlphaZero value expectations
        return math.tanh(blended)

    def stringRepresentation(self, board):
        return ','.join(map(str, board))

class HeuristicNetWrapper:
    def __init__(self, game: MISRTokenGame):
        self.game = game
        self.action_size = game.getActionSize()

    def predict(self, board):
        valid = self.game.getValidMoves(board, 1)
        s = int(valid.sum())
        if s == 0:
            v = self.game.getGameEnded(board, 1)
            return np.zeros(self.action_size, dtype=np.float32), float(v)
        p = np.where(valid == 1, 1.0 / s, 0.0).astype(np.float32)
        v = 0.0
        return p, v

    def train(self, examples): pass
    def save_checkpoint(self, folder, filename): pass
    def load_checkpoint(self, folder, filename): pass

def propose_with_alphazero(n: int, k: int, rng: random.Random,
                           sims_per_move: int = 128, cpuct: float = 1.5) -> List[Instance]:
    vocab_size = BASE_VOCAB + MAX_N
    game = MISRTokenGame(n=n, vocab_size=vocab_size)
    nnet = HeuristicNetWrapper(game)
    args = type("Args", (), {
        "numMCTSSims": sims_per_move,
        "cpuct": cpuct,
        "dirichletAlpha": 0.3,
        "dirichletEps": 0.25,
        "tempThreshold": 0,
    })()
    mcts = MCTS(game, nnet, args=args)

    proposals: List[Instance] = []
    for _ in range(k):
        board = game.getInitBoard()
        while True:
            pi = mcts.getActionProb(game.getCanonicalForm(board, 1), temp=1)
            if float(np.sum(pi)) <= 0.0:
                valid = game.getValidMoves(board, 1)
                legal = np.flatnonzero(valid).tolist()
                if not legal: break
                action = rng.choice(legal)
            else:
                # sample from pi
                action = int(np.random.choice(len(pi), p=np.array(pi, dtype=np.float64)))

            board, _ = game.getNextState(board, 1, action)
            if game.getGameEnded(board, 1) != 0:
                try:
                    sep_idx = board.index(SPECIAL["SEP"])
                except ValueError:
                    break
                H_tok = board[1:sep_idx]
                V_tok = board[sep_idx+1:]
                H = tokens_to_seq(H_tok); V = tokens_to_seq(V_tok)
                if len(H)==2*n and len(V)==2*n:
                    proposals.append((H, V))
                break

        if len(proposals) >= k:
            break

    return proposals
