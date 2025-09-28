# src/misr_core.py
from __future__ import annotations
from typing import List, Tuple, Dict
import random, math, time
import gurobipy as gp
from gurobipy import GRB

Seq = List[int]
Instance = Tuple[Seq, Seq]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

def seq_spans(seq: Seq) -> List[Tuple[int,int]]:
    first, spans = {}, {}
    for idx, lab in enumerate(seq):
        if lab not in first: first[lab] = idx
        else: spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n+1)]

def canonicalize(H: Seq, V: Seq) -> Instance:
    order, seen = [], set()
    for x in H:
        if x not in seen: order.append(x); seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def build_rects(H: Seq, V: Seq):
    X, Y = seq_spans(H), seq_spans(V)
    rects=[]
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def _grid_points(rects):
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def _covers_closed(rects, pts):
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2): S.append(i)
        C.append(S)
    return C

def solve_lp_ilp(rects, threads=0):
    pts = _grid_points(rects); covers = _covers_closed(rects, pts)
    m = gp.Model("misr_lp"); m.setParam('OutputFlag', 0)
    if threads>0: m.setParam('Threads', threads)
    n = len(rects)
    x = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m.optimize()
    lp = float(m.objVal) if m.status == GRB.OPTIMAL else 0.0

    m2 = gp.Model("misr_ilp"); m2.setParam('OutputFlag', 0)
    if threads>0: m2.setParam('Threads', threads)
    y = m2.addVars(n, vtype=GRB.BINARY, name='y')
    m2.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m2.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m2.optimize()
    ilp = float(m2.objVal) if m2.status == GRB.OPTIMAL else 0.0
    return lp, ilp

def score_ratio(H: Seq, V: Seq, alpha_lp=0.15, beta_ilp=0.10, threads=0):
    rects = build_rects(H,V)
    lp, ilp = solve_lp_ilp(rects, threads=threads)
    ratio = (lp/ilp) if ilp>0 else 0.0
    n = max(H) if H else 1
    blended = ratio + alpha_lp*(lp/n) - beta_ilp*(ilp/n)
    return lp, ilp, ratio, blended

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    seq = [i for i in range(1,n+1) for _ in range(2)]
    rng.shuffle(seq); return seq

def neighbors(H: Seq, V: Seq, rng: random.Random, k=96) -> List[Instance]:
    out=[]; L=len(H)
    moves = ['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV','pairHV']
    for _ in range(k):
        which = rng.choice(moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'): A[i],A[j] = A[j],A[i]
        elif which.startswith('move'):
            if i!=j: x=A.pop(i); A.insert(j,x)
        elif which.startswith('block'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                blk=A[a:b+1]; del A[a:b+1]
                t=rng.randrange(len(A)+1); A[t:t]=blk
        elif which.startswith('rev'):
            a,b=(i,j) if i<j else (j,i)
            if a!=b: A[a:b+1] = list(reversed(A[a:b+1]))
        elif which=='pairHV':
            labs=list(set(H)&set(V))
            if len(labs)>=2:
                a_lab,b_lab = rng.sample(labs,2)
                AH, AV = H[:], V[:]
                for S in (AH,):
                    pa=[p for p,x in enumerate(S) if x==a_lab]
                    pb=[p for p,x in enumerate(S) if x==b_lab]
                    for ia,ib in zip(pa,pb): S[ia],S[ib]=S[ib],S[ia]
                for S in (AV,):
                    pa=[p for p,x in enumerate(S) if x==a_lab]
                    pb=[p for p,x in enumerate(S) if x==b_lab]
                    for ia,ib in zip(pa,pb): S[ia],S[ib]=S[ib],S[ia]
                out.append(canonicalize(AH,AV)); continue
        else:
            labs=list(set(A))
            if len(labs)>=2:
                a_lab,b_lab = rng.sample(labs,2)
                pa=[p for p,x in enumerate(A) if x==a_lab]
                pb=[p for p,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia,ib in zip(pa,pb): A[ia],A[ib]=A[ib],A[ia]
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

def local_search(seed: Instance, rng: random.Random, time_budget_s=3.0,
                 alpha_lp=0.15, beta_ilp=0.10, threads=0,
                 tabu_seconds=20.0, elite_size=64, neighbor_k=96):
    start=time.time()
    H,V = canonicalize(*seed)
    seen: Dict[str, Tuple[float,float,float,float]]={}
    elites: List[Tuple[float, Seq, Seq]]=[]; tabu: Dict[str,float]={}; best=-1.0

    def key(h,v): return ','.join(map(str,h))+'|'+','.join(map(str,v))
    def push(score,h,v):
        elites.append((score,h[:],v[:])); elites.sort(key=lambda x:-x[0])
        if len(elites)>elite_size: elites.pop()

    while time.time()-start < time_budget_s:
        k=key(H,V); now=time.time()
        if k in tabu and (now-tabu[k] < tabu_seconds):
            if elites: _,H,V = random.choice(elites)
            else: H,V = H[::-1], V[::-1]; continue

        if k not in seen:
            lp,ilp,ratio,blend = score_ratio(H,V,alpha_lp,beta_ilp,threads)
            seen[k]=(lp,ilp,ratio,blend); push(ratio,H,V); best=max(best,ratio)
        else:
            lp,ilp,ratio,blend = seen[k]

        best_nb=None; best_sc=-1e9
        for (h2,v2) in neighbors(H,V,rng,neighbor_k):
            k2=key(h2,v2)
            if k2 in seen: lp2,ilp2,r2,b2 = seen[k2]
            else:
                lp2,ilp2,r2,b2 = score_ratio(h2,v2,alpha_lp,beta_ilp,threads)
                seen[k2]=(lp2,ilp2,r2,b2); push(r2,h2,v2)
            if b2>best_sc: best_sc=b2; best_nb=(h2,v2,lp2,ilp2,r2,b2)

        if best_nb:
            _,_,lp2,ilp2,r2,b2=best_nb
            if b2>=blend: H,V = best_nb[0],best_nb[1]
            else:
                delta=b2-blend; T=0.03; import random as pyrand
                if math.exp(delta/max(T,1e-6)) > pyrand.random(): H,V = best_nb[0],best_nb[1]
                else:
                    tabu[k]=now
                    if elites: _,H,V = random.choice(elites)
                    else: H,V = H[::-1], V[::-1]
            best=max(best,r2)

    elites_sorted = sorted(elites, key=lambda x:-x[0])
    return elites_sorted, best
