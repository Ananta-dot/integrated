# src/main.py
from __future__ import annotations
import argparse, hashlib, json, math, os, pickle, random, time, warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB

# Local (assumes these exist in your repo)
from src.constants import SPECIAL, BASE_VOCAB, MAX_N, N_BASE, VOCAB_SIZE
from src.backends.nanogpt_adapter import (
    NanoGPTWrapper, make_batch, sample_model, train_one_step
)
from src.backends.az_general import propose_with_alphazero

warnings.filterwarnings("ignore", message=r".*TF32 behavior.*", category=UserWarning)

# -----------------------
# Device
# -----------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -----------------------
# Types
# -----------------------
Seq = List[int]
Instance = Tuple[Seq, Seq]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

# -----------------------
# Seeding
# -----------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------
# Helpers (your original logic)
# -----------------------
def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    first = {}
    spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n + 1)]

def canonicalize(H: Seq, V: Seq) -> Instance:
    order = []
    seen = set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def instance_key(H: Seq, V: Seq) -> str:
    s = ','.join(map(str, H)) + '|' + ','.join(map(str, V))
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    seq = [i for i in range(1, n + 1) for _ in range(2)]
    rng.shuffle(seq)
    return seq

def motif_rainbow(n: int) -> Seq:
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    out=[]
    for i in range(1, n+1):
        out += [i,i]
    return out

def motif_interleave(n: int) -> Seq:
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1<=n else i
        out += [i, j, i, j]
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_zipper(n: int) -> Seq:
    out=[]
    for i in range(1, (n//2)+1):
        j = n - i + 1
        out += [i, j, i, j]
    if n % 2 == 1:
        k = (n//2)+1
        out += [k,k]
    return out[:2*n]

def motif_ladder(n: int) -> Seq:
    out=[]
    a, b = 1, 2
    while len(out) < 2*n:
        out += [a, b if b<=n else a]
        a += 1
        b += 1
        if a > n: a = n
        if b > n: b = n
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_corner_combo(n: int) -> Tuple[Seq, Seq]:
    return motif_rainbow(n), motif_doubled(n)

def motif_seeds(n: int) -> List[Seq]:
    S = [
        motif_rainbow(n),
        motif_doubled(n),
        motif_interleave(n),
        motif_zipper(n),
        motif_ladder(n),
        list(range(1, n+1)) + list(range(1, n+1)),
        [x for pair in zip(range(1,n+1), range(1,n+1)) for x in pair],
    ]
    out=[]
    for s in S:
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)):
            out.append(s)
    return out

def seeded_pool(n: int, rng: random.Random, base_count: int) -> List[Instance]:
    seeds = []
    motifs = motif_seeds(n)
    for m in motifs:
        seeds.append((m[:], random_valid_seq(n, rng)))
        seeds.append((random_valid_seq(n, rng), m[:]))
    RH, RV = motif_corner_combo(n)
    seeds.append((RH[:], RV[:]))
    seeds.append((RV[:], RH[:]))
    for i in range(min(len(motifs) - 1, 3)):
        seeds.append((motifs[i][:], motifs[i + 1][:]))
    while len(seeds) < base_count:
        seeds.append((random_valid_seq(n, rng), random_valid_seq(n, rng)))
    return seeds[:base_count]

# Depth-weighted curriculum lift
def lift_instance(H: Seq, V: Seq, n_new: int, rng: random.Random) -> Instance:
    assert n_new >= max(H)
    H2, V2 = H[:], V[:]

    def depth(seq: Seq) -> List[int]:
        sp = seq_spans(seq)
        line = [0]*(len(seq)+1)
        for (l,r) in sp:
            if l < r:
                line[l] += 1
                if r+1 < len(line):
                    line[r+1] -= 1
        out=[]; cur=0
        for i in range(len(seq)):
            cur += line[i]; out.append(cur)
        return out

    def weighted_idx(weights: List[int]) -> int:
        tot = sum(w+1 for w in weights)
        x = rng.randrange(tot); s=0
        for i,w in enumerate(weights):
            s += (w+1)
            if x < s: return i
        return len(weights)-1

    for lab in range(max(H)+1, n_new+1):
        for seq in (H2, V2):
            d = depth(seq)
            i = weighted_idx(d); j = min(i+1, len(seq))
            seq.insert(i, lab); seq.insert(j, lab)
    return canonicalize(H2, V2)

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def grid_points(rects: List[Rect]) -> List[Tuple[int,int]]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts: List[Tuple[int,int]]) -> List[List[int]]:
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

def solve_lp_ilp(rects: List[Rect], grb_threads: int = 0) -> Tuple[float, float]:
    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    return lp, ilp

def score_ratio(H: Seq, V: Seq,
                alpha_lp: float = 0.0,
                beta_ilp: float = 0.0,
                grb_threads: int = 0) -> Tuple[float,float,float,float]:
    rects = build_rects(H,V)
    lp, ilp = solve_lp_ilp(rects, grb_threads=grb_threads)
    ratio = (lp/ilp) if ilp > 0 else 0.0
    n = max(H) if H else 1
    blended = ratio + alpha_lp * (lp / n) - beta_ilp * (ilp / n)
    return lp, ilp, ratio, blended

def neighbors(H: Seq, V: Seq, rng: random.Random, k: int = 96) -> List[Instance]:
    out=[]
    L = len(H)
    moves = ['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV','pairHV']
    for _ in range(k):
        which = rng.choice(moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'):
            A[i], A[j] = A[j], A[i]
        elif which.startswith('move'):
            if i!=j:
                x = A.pop(i); A.insert(j, x)
        elif which.startswith('block'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                blk = A[a:b+1]; del A[a:b+1]
                t = rng.randrange(len(A)+1); A[t:t]=blk
        elif which == 'pairHV':
            labs = list(set(H) & set(V))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                AH, AV = H[:], V[:]
                for S in (AH,):
                    pa = [idx for idx,x in enumerate(S) if x==a_lab]
                    pb = [idx for idx,x in enumerate(S) if x==b_lab]
                    for ia, ib in zip(pa, pb): S[ia], S[ib] = S[ib], S[ia]
                for S in (AV,):
                    pa = [i for i,x in enumerate(S) if x==a_lab]
                    pb = [i for i,x in enumerate(S) if x==b_lab]
                    for ia, ib in zip(pa, pb): S[ia], S[ib] = S[ib], S[ia]
                out.append(canonicalize(AH, AV)); continue
        elif which.startswith('rev'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b: A[a:b+1] = list(reversed(A[a:b+1]))
        else:
            labs = list(set(A))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                pa = [idx for idx,x in enumerate(A) if x==a_lab]
                pb = [idx for idx,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia, ib in zip(pa, pb): A[ia], A[ib] = A[ib], A[ia]
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

def local_search(seed: Instance,
                 time_budget_s: float,
                 rng: random.Random,
                 alpha_lp: float,
                 beta_ilp: float,
                 grb_threads: int = 0,
                 tabu_seconds: float = 20.0,
                 elite_size: int = 64,
                 neighbor_k: int = 96):
    start = time.time()
    H, V = canonicalize(*seed)
    seen: Dict[str, Tuple[float,float,float,float]] = {}
    elites: List[Tuple[float, Seq, Seq]] = []
    tabu: Dict[str, float] = {}
    best = -1.0

    def push(score: float, h: Seq, v: Seq):
        elites.append((score, h[:], v[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_size: elites.pop()

    while time.time() - start < time_budget_s:
        key = instance_key(H,V); now = time.time()
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elites: _, H, V = random.choice(elites)
            else: H, V = H[::-1], V[::-1]
            continue
        if key not in seen:
            lp, ilp, ratio, blended = score_ratio(H,V, alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads)
            seen[key] = (lp, ilp, ratio, blended)
            push(ratio, H, V)
            best = max(best, ratio)
        else:
            lp, ilp, ratio, blended = seen[key]
        cand = neighbors(H,V,rng,neighbor_k)
        best_nb=None; best_sc=-1e9
        for (h2,v2) in cand:
            k2 = instance_key(h2,v2)
            if k2 in seen:
                lp2, ilp2, r2, b2 = seen[k2]
            else:
                lp2, ilp2, r2, b2 = score_ratio(h2,v2, alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads)
                seen[k2] = (lp2, ilp2, r2, b2)
                push(r2, h2, v2)
            if b2 > best_sc:
                best_sc = b2; best_nb=(h2,v2,lp2,ilp2,r2,b2)
        if best_nb:
            _,_,lp2,ilp2,r2,b2 = best_nb
            if b2 >= blended:
                H, V = best_nb[0], best_nb[1]
            else:
                delta = b2 - blended
                T = 0.03
                if math.exp(delta/max(T,1e-6)) > random.random():
                    H, V = best_nb[0], best_nb[1]
                else:
                    tabu[key] = now
                    if elites: _, H, V = random.choice(elites)
                    else: H, V = H[::-1], V[::-1]
            best = max(best, r2)
    elites_sorted = sorted(elites, key=lambda x: -x[0])
    return elites_sorted, best

def elites_for_n(elites: List[Tuple[float, Seq, Seq]], n: int) -> List[Tuple[float, Seq, Seq]]:
    return [e for e in elites if e[1] and max(e[1]) == n]

def recombine_seeds(elites: List[Tuple[float, Seq, Seq]], k: int, rng: random.Random, n: int) -> List[Instance]:
    pool = elites_for_n(elites, n)
    if not pool: return []
    out=[]
    for _ in range(k):
        _, H1, _ = rng.choice(pool)
        _, _, V2 = rng.choice(pool)
        h, v = canonicalize(H1, V2); out.append((h,v))
    return out

def ns_sequence(n_start: int, n_target: int, step: int) -> List[int]:
    out=[]; n=n_start
    while n<=n_target: out.append(n); n+=step
    return out

def make_run_dirs(out_root: str, seed: int, n_list: List[int]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_root, f"misr_run-{ts}-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for n in n_list:
        os.makedirs(os.path.join(run_dir, f"n{n:02d}"), exist_ok=True)
    return run_dir

# -----------------------
# Runner
# -----------------------
def run_patternboost(
    seed: int = 123,
    n_start: int = 8,
    n_target: int = 32,
    rounds_per_n: int = 10,
    seeds_per_round: int = 32,
    local_time_per_seed: float = 3.0,
    elites_to_train: int = 96,
    batch_size: int = 32,
    train_steps_per_round: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9,
    alpha_lp: float = 0.15,
    beta_ilp: float = 0.10,
    grb_threads: int = 0,
    lift_step: int = 3,
    out_root: str = "runs",
    seed_pkl: str = "",
    seed_top_k: int = 4,
    use_az: bool = False,
    az_frac: float = 0.25,
):
    rng = random.Random(seed)
    seed_everything(seed)

    n_list = ns_sequence(n_start, n_target, lift_step)
    run_dir = make_run_dirs(out_root, seed, n_list)
    print(f"[run_dir] {run_dir}")

    # persist args
    with open(os.path.join(run_dir, "run_args.json"), "w") as f:
        json.dump({
            "seed": seed, "n_start": n_start, "n_target": n_target,
            "rounds_per_n": rounds_per_n, "seeds_per_round": seeds_per_round,
            "local_time_per_seed": local_time_per_seed,
            "elites_to_train": elites_to_train, "batch_size": batch_size,
            "train_steps_per_round": train_steps_per_round,
            "temperature": temperature, "top_p": top_p,
            "alpha_lp": alpha_lp, "beta_ilp": beta_ilp,
            "grb_threads": grb_threads, "lift_step": lift_step,
            "use_az": use_az, "az_frac": az_frac
        }, f, indent=2)

    # nanoGPT model
    model = NanoGPTWrapper(n_layer=3, n_head=6, n_embd=192, dropout=0.1, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    print(f"Device: {DEVICE} | backend=nanogpt | AlphaZero={use_az}")
    nparams = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {nparams/1e6:.2f}M")

    elites: List[Tuple[float, Seq, Seq]] = []
    def push_elite(score, H, V):
        elites.append((score, H[:], V[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    # initial seeds
    n = n_start
    best_overall = 0.0
    seeds = seeded_pool(n, rng, seeds_per_round)

    # seed injection (optional)
    if seed_pkl:
        try:
            with open(seed_pkl, "rb") as f:
                data = pickle.load(f)
            injected=[]
            for item in data:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 3 and isinstance(item[1], list) and isinstance(item[2], list):
                        ratio, H, V = item[:3]
                        if H and max(H) == n: injected.append((H,V))
                    elif len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
                        H, V = item
                        if H and max(H) == n: injected.append((H,V))
            take = min(len(injected), seed_top_k, seeds_per_round)
            if take > 0:
                seeds = injected[:take] + seeds[:max(0, seeds_per_round - take)]
                print(f"[seed_inject] loaded {take} seeds from {seed_pkl} for n={n}")
        except Exception as e:
            print(f"[seed_inject] failed to load {seed_pkl}: {e}")

    round_idx = 0
    while n <= n_target:
        n_dir = os.path.join(run_dir, f"n{n:02d}")
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")
        for r in range(rounds_per_n):
            round_idx += 1

            # 0) AlphaZero proposals (optionally gated after round 1)
            az_seeds: List[Instance] = []
            if use_az and round_idx >= 2:
                k_az = max(0, int(az_frac * seeds_per_round))
                if k_az > 0:
                    az_seeds = propose_with_alphazero(n, k=k_az, rng=rng)
                    az_seeds = [canonicalize(h,v) for (h,v) in az_seeds if h and v and len(h)==2*n and len(v)==2*n]

            # 1) Local search on seeds (+ AZ seeds)
            for (H,V) in seeds + az_seeds:
                es, best = local_search(
                    (H,V),
                    time_budget_s=local_time_per_seed,
                    rng=rng,
                    alpha_lp=alpha_lp,
                    beta_ilp=beta_ilp,
                    grb_threads=grb_threads,
                    elite_size=64,
                    neighbor_k=96
                )
                for (score, h, v) in es:
                    push_elite(score, h, v)
                if best is not None:
                    best_overall = max(best_overall, best)
            print(f"[round {r+1}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f}")

            # 2) Train NanoGPT on elites
            topk = elites[:max(elites_to_train, min(32, len(elites)))]
            if topk:
                last_loss = None
                for _ in range(train_steps_per_round):
                    tokens, targets = make_batch(topk, min(batch_size, len(topk)), DEVICE, rng)
                    last_loss = train_one_step(model, opt, tokens, targets)
                print(f"   trained {train_steps_per_round} steps, last loss ~ {last_loss:.3f}")

            # 3) New seeds (recombine + model + motif refresh + jitter)
            new_seeds: List[Instance] = []
            new_seeds.extend(recombine_seeds(elites, k=max(1, seeds_per_round//4), rng=rng, n=n))

            while len(new_seeds) < seeds_per_round:
                elite_mut = (rng.random() < 0.25)
                pool_n = [e for e in elites if e[1] and max(e[1]) == n]
                if elite_mut and pool_n:
                    _, h, v = rng.choice(pool_n[:min(64, len(pool_n))])
                    h = h[:]; v = v[:]
                    for S in (h, v):
                        if rng.random() < 0.6:
                            i, j = rng.randrange(len(S)), rng.randrange(len(S))
                            S[i], S[j] = S[j], S[i]
                    new_seeds.append((h, v))
                else:
                    h, v = sample_model(model, n, DEVICE, temperature=temperature, top_p=top_p)
                    if (not h) or (not v) or len(h)!=2*n or len(v)!=2*n:
                        h, v = random_valid_seq(n, rng), random_valid_seq(n, rng)
                    if rng.random() < 0.35:
                        for S in (h, v):
                            a = rng.randrange(len(S)); b = rng.randrange(len(S))
                            a, b = min(a,b), max(a,b)
                            if a != b:
                                S[a:b+1] = list(reversed(S[a:b+1]))
                    new_seeds.append((h, v))

            # motif refresh
            mix=[]
            RH, RV = motif_corner_combo(n)
            mix.append((RH[:], RV[:]))
            mix.append((RV[:], RH[:]))
            for m in motif_seeds(n)[:2]:
                mix.append((m[:], motif_rainbow(n)))
            for i in range(min(len(mix), max(2, seeds_per_round//8))):
                new_seeds[i] = mix[i]
            seeds = new_seeds

            # 4) SAVE elites for current n after *every* round (strict format)
            elites_n_only_raw = elites_for_n(elites, n)

            # coerce to clean tuples: (float_ratio, H_list[int], V_list[int])
            elites_n_only: List[Tuple[float, List[int], List[int]]] = []
            for (score, h, v) in elites_n_only_raw:
                if not h: continue
                if max(h) != n: continue
                H = [int(t) for t in h]
                V = [int(t) for t in v]
                elites_n_only.append((float(score), H, V))

            ts_round = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pkl_name = f"round{r+1:02d}_elites_n{n:02d}_{ts_round}.pkl"
            pkl_path = os.path.join(n_dir, pkl_name)

            if elites_n_only:
                with open(pkl_path, "wb") as f:
                    pickle.dump(elites_n_only, f, protocol=4)
                    f.flush(); os.fsync(f.fileno())
                # atomic LATEST update
                tmp_latest = os.path.join(n_dir, "LATEST.tmp")
                with open(tmp_latest, "w") as f:
                    f.write(pkl_name + "\n")
                    f.flush(); os.fsync(f.fileno())
                os.replace(tmp_latest, os.path.join(n_dir, "LATEST.txt"))
                print(f"[save] n={n:02d} round {r+1}/{rounds_per_n}: wrote {len(elites_n_only)} elites -> {pkl_path}")
            else:
                print(f"[save] n={n:02d} round {r+1}/{rounds_per_n}: no n-matching elites; skip writing.")

        # 5) lift
        if elites:
            n_next = n + lift_step
            lifted=[]
            for _, h, v in elites[:min(96, len(elites))]:
                h2, v2 = lift_instance(h, v, n_next, rng)
                lifted.append((h2,v2))
            seeds = lifted + seeds[:max(0, seeds_per_round - len(lifted))]
            n = n_next
        else:
            seeds = seeded_pool(n, rng, seeds_per_round)

    # summary + final save
    print("\n=== BEST ELITES ===")
    for i,(score,h,v) in enumerate(elites[:10]):
        print(f"#{i+1} ratio={score:.4f}  n={max(h)}")
    final_path = os.path.join(run_dir, "final_elites.pkl")
    # store final elites in the same strict format (top 256)
    final_clean = [(float(sc), [int(t) for t in h], [int(t) for t in v]) for (sc,h,v) in elites[:256]]
    with open(final_path, "wb") as f:
        pickle.dump(final_clean, f, protocol=4)
    print(f"Saved top elites to {final_path}")
    return elites, run_dir

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=32)
    ap.add_argument("--local_time_per_seed", type=float, default=3.0)
    ap.add_argument("--elites_to_train", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--alpha_lp", type=float, default=0.15)
    ap.add_argument("--beta_ilp", type=float, default=0.10)
    ap.add_argument("--grb_threads", type=int, default=8)
    ap.add_argument("--lift_step", type=int, default=3)
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--seed_pkl", type=str, default="")
    ap.add_argument("--seed_top_k", type=int, default=4)
    ap.add_argument("--use_az", action="store_true")
    ap.add_argument("--az_frac", type=float, default=0.25)
    args = ap.parse_args()

    run_patternboost(
        seed=args.seed,
        n_start=args.n_start,
        n_target=args.n_target,
        rounds_per_n=args.rounds_per_n,
        seeds_per_round=args.seeds_per_round,
        local_time_per_seed=args.local_time_per_seed,
        elites_to_train=args.elites_to_train,
        batch_size=args.batch_size,
        train_steps_per_round=args.train_steps_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        alpha_lp=args.alpha_lp,
        beta_ilp=args.beta_ilp,
        grb_threads=args.grb_threads,
        lift_step=args.lift_step,
        out_root=args.out_root,
        seed_pkl=args.seed_pkl,
        seed_top_k=args.seed_top_k,
        use_az=args.use_az,
        az_frac=args.az_frac,
    )

if __name__ == "__main__":
    main()
