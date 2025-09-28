# src/main.py
from __future__ import annotations
import os, json, random, pathlib, pickle
from datetime import datetime
from typing import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

from .config import get_args
from .data import Batch, make_batch, BASE_VOCAB, MAX_N
from .sampler import sample_model
from .misr_core import (random_valid_seq, local_search, canonicalize)

# backends
from .backends.ngpt import NanoGPT
from .backends.makemore_backend import MakeMoreTransformer
from .backends.az_general import propose_with_alphazero

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def elites_for_n(elites, n):
    return [e for e in elites if e[1] and max(e[1])==n]

def recombine_seeds(elites, k, rng, n):
    pool = elites_for_n(elites, n)
    if not pool: return []
    out=[]
    for _ in range(k):
        _, H1, _ = rng.choice(pool)
        _, _, V2 = rng.choice(pool)
        h,v = canonicalize(H1, V2); out.append((h,v))
    return out

def ns_sequence(n_start, n_target, step):
    out=[]; n=n_start
    while n<=n_target: out.append(n); n+=step
    return out

def make_run_dirs(out_root, seed, n_list):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_root, f"misr_run-{ts}-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for n in n_list: os.makedirs(os.path.join(run_dir, f"n{n:02d}"), exist_ok=True)
    return run_dir

def train_one_step(model: nn.Module, opt: torch.optim.Optimizer, batch: Batch):
    model.train()
    # If the backend exposes a .gpt (nanoGPT), use its built-in loss path
    if hasattr(model, "gpt"):
        # nanoGPT expects ignore_index = -1 by default
        targets = batch.targets.clone()
        targets[targets == 2] = -1  # map EOS(2) padding to -1 ignore
        logits, loss = model.gpt(batch.tokens, targets=targets)
    else:
        # generic fallback (e.g., makemore_backend)
        logits = model(batch.tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.targets.reshape(-1),
            ignore_index=2  # EOS pad
        )

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())

def main():
    args = get_args()
    DEVICE = get_device()
    print(f"Device: {DEVICE} | backend={args.backend} | AlphaZero={args.use_az}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # model backend
    vocab = BASE_VOCAB + MAX_N
    if args.backend == "nanogpt":
        model = NanoGPT(vocab_size=vocab, d_model=192, n_head=6, n_layer=3, dropout=0.1, block_size=4096).to(DEVICE)
    else:
        model = MakeMoreTransformer(vocab_size=vocab, d_model=192, n_head=6, n_layer=3, dropout=0.1, block_size=4096).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    # dirs + args
    n_list = ns_sequence(args.n_start, args.n_target, args.lift_step)
    run_dir = make_run_dirs(args.out_root, args.seed, n_list)
    with open(os.path.join(run_dir, "run_args.json"), "w") as f: json.dump(vars(args), f, indent=2)

    elites: List[Tuple[float, List[int], List[int]]] = []
    def push_elite(score, H, V):
        elites.append((score, H[:], V[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    # initial seeds
    n = args.n_start; best_overall = 0.0
    seeds = [(random_valid_seq(n, rng), random_valid_seq(n, rng)) for _ in range(args.seeds_per_round)]

    while n <= args.n_target:
        n_dir = os.path.join(run_dir, f"n{n:02d}")
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")

        for r in range(args.rounds_per_n):
            # Optional: add AlphaZero proposals
            if args.use_az:
                az = propose_with_alphazero(n, k=max(1, args.seeds_per_round//8), rng=rng)
                seeds = seeds[:max(0, len(seeds)-len(az))] + az

            # 1) Local search
            # from .misr_core import local_search  # ensure fresh import if modified
            for (H,V) in seeds:
                es, best = local_search(
                    (H,V), rng=rng, time_budget_s=args.local_time_per_seed,
                    alpha_lp=args.alpha_lp, beta_ilp=args.beta_ilp,
                    threads=args.grb_threads, elite_size=64, neighbor_k=96
                )
                for (score, h, v) in es: push_elite(score, h, v)
                if best is not None: best_overall = max(best_overall, best)
            print(f"[round {r+1}/{args.rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f}")

            # 2) Train model
            topk = elites[:max(args.elites_to_train, min(32, len(elites)))]
            if topk:
                for _ in range(args.train_steps_per_round):
                    batch = make_batch(topk, min(args.batch_size, len(topk)), DEVICE, rng)
                    last_loss = train_one_step(model, opt, batch)
                print(f"   trained {args.train_steps_per_round} steps, last loss ~ {last_loss:.3f}")

            # 3) New seeds: recombine + model + jitter
            new_seeds = []
            new_seeds.extend(recombine_seeds(elites, k=max(1, args.seeds_per_round//4), rng=rng, n=n))
            while len(new_seeds) < args.seeds_per_round:
                pool_n = elites_for_n(elites, n)
                elite_mut = (rng.random() < 0.25)
                if elite_mut and pool_n:
                    _, h, v = rng.choice(pool_n[:min(64, len(pool_n))]); h=h[:]; v=v[:]
                    for S in (h, v):
                        if rng.random() < 0.6:
                            i, j = rng.randrange(len(S)), rng.randrange(len(S))
                            S[i], S[j] = S[j], S[i]
                    new_seeds.append((h, v))
                else:
                    h, v = sample_model(model, DEVICE, n, temperature=args.temperature, top_p=args.top_p)
                    if (not h) or (not v) or len(h)!=2*n or len(v)!=2*n:
                        # from .misr_core import random_valid_seq
                        h, v = random_valid_seq(n, rng), random_valid_seq(n, rng)
                    if rng.random() < 0.35:
                        for S in (h, v):
                            a = rng.randrange(len(S)); b = rng.randrange(len(S))
                            a,b = min(a,b), max(a,b)
                            if a!=b: S[a:b+1] = list(reversed(S[a:b+1]))
                    new_seeds.append((h, v))
            seeds = new_seeds

            # 4) Save elites for current n
            elites_n_only = elites_for_n(elites, n)
            ts_round = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pkl_name = f"round{r+1:02d}_elites_n{n:02d}_{ts_round}.pkl"
            with open(os.path.join(n_dir, pkl_name), "wb") as f: pickle.dump(elites_n_only, f)
            with open(os.path.join(n_dir, "LATEST.txt"), "w") as f: f.write(pkl_name + "\n")

        # curriculum lift (simple lift; you can swap in your richer lift later)
        n_next = n + args.lift_step
        if elites:
            lifted=[]
            for _, h, v in elites[:min(96, len(elites))]:
                H2, V2 = h[:], v[:]
                for lab in range(max(h)+1, n_next+1):
                    import random as pyrand
                    for S in (H2, V2):
                        i = pyrand.randrange(len(S)+1); S.insert(i, lab)
                        j = pyrand.randrange(len(S)+1); S.insert(j, lab)
                lifted.append((H2,V2))
            seeds = lifted + seeds[:max(0, args.seeds_per_round - len(lifted))]
            n = n_next
        else:
            seeds = [(random_valid_seq(n, rng), random_valid_seq(n, rng)) for _ in range(args.seeds_per_round)]

    final_path = os.path.join(run_dir, "final_elites.pkl")
    with open(final_path, "wb") as f: pickle.dump(elites[:256], f)
    print(f"Saved top elites to {final_path}")

if __name__ == "__main__":
    main()
