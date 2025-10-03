# src/seeds.py
from __future__ import annotations
from typing import List, Tuple
import random, pickle

Seq = List[int]
Instance = Tuple[Seq, Seq]

def canonicalize(H: Seq, V: Seq) -> Instance:
    order, seen = [], set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    seq = [i for i in range(1,n+1) for _ in range(2)]
    rng.shuffle(seq); return seq

# ---------- motifs ----------
def motif_rainbow(n: int) -> Seq:
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    out=[]; 
    for i in range(1, n+1): out += [i,i]
    return out

def motif_interleave(n: int) -> Seq:
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1 <= n else i
        out += [i, j, i, j]
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2: fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2: fixed.append(i); cnt[i]+=1
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
    out=[]; a, b = 1, 2
    while len(out) < 2*n:
        out += [a, b if b<=n else a]
        a += 1; b += 1
        if a > n: a = n
        if b > n: b = n
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2: fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2: fixed.append(i); cnt[i]+=1
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
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)): out.append(s)
    return out

# ---------- seed pools ----------
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
    # canonicalize and trim
    out=[]
    for H,V in seeds[:base_count]:
        h,v = canonicalize(H,V); out.append((h,v))
    return out

def load_seeds_from_pkl(seed_pkl: str, n: int, top_k: int = 10) -> List[Instance]:
    try:
        with open(seed_pkl, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return []
    bag=[]
    for item in data:
        if isinstance(item, (list,tuple)):
            if len(item)>=3 and isinstance(item[1], list) and isinstance(item[2], list):
                ratio, H, V = item[:3]
                if H and max(H)==n: bag.append((float(ratio), H, V))
            elif len(item)==2 and isinstance(item[0],list) and isinstance(item[1],list):
                H,V = item
                if H and max(H)==n: bag.append((0.0, H, V))
    bag.sort(key=lambda t: -t[0])
    return [canonicalize(H,V) for _,H,V in bag[:top_k]]
