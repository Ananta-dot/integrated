#!/usr/bin/env python3
# Verify & visualize MISR elites saved by PatternBoost runner.
# - Accepts either a .pkl file or a directory (follows LATEST.txt or picks newest .pkl)
# - Understands (ratio,H,V), (H,V), or dicts {"H":..., "V":..., ...}
# - Recomputes LP, ILP, LP/ILP using the same edge-grid cover as the main run
# - Plots rectangles; ILP-chosen rects have thicker outlines

from __future__ import annotations
import argparse, os, pickle, glob, math
from typing import List, Tuple, Any, Optional

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

Seq  = List[int]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

# -------------------------
# I/O helpers
# -------------------------
def resolve_input_path(p: str) -> str:
    """Return a concrete .pkl path from a direct path or a directory."""
    if os.path.isfile(p) and p.endswith(".pkl"):
        return p
    if os.path.isdir(p):
        latest = os.path.join(p, "LATEST.txt")
        if os.path.isfile(latest):
            with open(latest, "r") as f:
                name = f.read().strip()
            path = name if os.path.isabs(name) else os.path.join(p, name)
            if os.path.isfile(path):
                return path
        # otherwise pick newest .pkl under p
        pkls = []
        for root, _, files in os.walk(p):
            for fn in files:
                if fn.endswith(".pkl"):
                    pkls.append(os.path.join(root, fn))
        if not pkls:
            raise FileNotFoundError(f"No .pkl files found under {p}")
        pkls.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        return pkls[0]
    raise FileNotFoundError(f"Not a .pkl or directory: {p}")

def _is_int_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(t, int) for t in x)

def parse_elites(obj: Any):
    """
    Return a list of tuples (stored_ratio_or_None, H, V) from various formats:
      - [(ratio, H, V), ...]
      - [(H, V), ...]
      - [{"H":..., "V":..., "ratio":...}, ...]
    """
    if not isinstance(obj, (list, tuple)):
        raise ValueError("Pickle top-level not a list/tuple.")
    out = []
    for item in obj:
        ratio = None; H = None; V = None
        if isinstance(item, (list, tuple)):
            if len(item) >= 3 and _is_int_list(item[1]) and _is_int_list(item[2]):
                H, V = list(item[1]), list(item[2])
                try: ratio = float(item[0])
                except Exception: ratio = None
            elif len(item) == 2 and _is_int_list(item[0]) and _is_int_list(item[1]):
                H, V = list(item[0]), list(item[1])
            else:
                # try to find two int-lists inside
                ints = [x for x in item if _is_int_list(x)]
                if len(ints) >= 2:
                    H, V = list(ints[0]), list(ints[1])
        elif isinstance(item, dict):
            if "H" in item and "V" in item and _is_int_list(item["H"]) and _is_int_list(item["V"]):
                H, V = list(item["H"]), list(item["V"])
                if "ratio" in item:
                    try: ratio = float(item["ratio"])
                    except Exception: pass
        if H is not None and V is not None:
            out.append((ratio, H, V))
    if not out:
        raise ValueError("No (H,V) entries parsed from pickle.")
    return out

# -------------------------
# Geometry & MISR models (match runner)
# -------------------------
def seq_spans(seq: Seq) -> List[Tuple[int,int]]:
    first = {}; spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n+1)]

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def _grid_points_edges(rects: List[Rect]) -> List[Tuple[float,float]]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(float(x), float(y)) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts: List[Tuple[float,float]]) -> List[List[int]]:
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

def solve_lp_ilp_with_vals(rects: List[Rect], threads: int = 0):
    """
    Same edge-grid clique constraints as the runner. Returns (LP, ILP, x_vals, y_vals).
    """
    pts = _grid_points_edges(rects)
    covers = covers_grid_closed(rects, pts)
    n = len(rects)

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam("OutputFlag", 0)
    if threads > 0: m_lp.setParam("Threads", threads)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1.0)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0
    x_vals = [float(x[i].X) for i in range(n)]

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam("OutputFlag", 0)
    if threads > 0: m_ilp.setParam("Threads", threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY)
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1.0)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    y_vals = [int(round(y[i].X)) for i in range(n)]

    return lp, ilp, x_vals, y_vals

# -------------------------
# Plotting
# -------------------------
def plot_rects(rects: List[Rect], y_vals, title: str, outpng: Optional[str]):
    fig, ax = plt.subplots(figsize=(6,6))
    for i, ((x1,x2),(y1,y2)) in enumerate(rects, start=1):
        w = (x2 - x1); h = (y2 - y1)
        lw = 2.5 if (i-1) < len(y_vals) and y_vals[i-1] == 1 else 1.0
        rect = plt.Rectangle((x1, y1), w, h, fill=False, linewidth=lw)
        ax.add_patch(rect)
        cx = x1 + w/2.0; cy = y1 + h/2.0
        ax.text(cx, cy, str(i), ha='center', va='center', fontsize=9)
    xs = [x for r in rects for x in (r[0][0], r[0][1])]
    ys = [y for r in rects for y in (r[1][0], r[1][1])]
    if xs and ys:
        ax.set_xlim(min(xs)-0.5, max(xs)+0.5)
        ax.set_ylim(min(ys)-0.5, max(ys)+0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("H index"); ax.set_ylabel("V index")
    ax.set_title(title)
    ax.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    if outpng:
        plt.savefig(outpng, dpi=160, bbox_inches="tight")
        print(f"[plot] saved: {outpng}")
    plt.show()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="Path to a .pkl OR an nXX/ directory (will follow LATEST.txt).")
    ap.add_argument("--rank", type=int, default=1,
                    help="1-based elite rank to verify/plot (after recomputing).")
    ap.add_argument("--threads", type=int, default=0,
                    help="Gurobi threads (0 = default).")
    ap.add_argument("--list", type=int, default=0,
                    help="If >0, list top-K by recomputed LP/ILP before plotting rank.")
    ap.add_argument("--no_plot", action="store_true",
                    help="Skip plotting (only print numbers).")
    ap.add_argument("--out", type=str, default="",
                    help="Optional output PNG path (default: auto).")
    args = ap.parse_args()

    pkl_path = resolve_input_path(args.path)
    print(f"[verify] using: {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    bag = parse_elites(raw)  # [(stored_ratio_or_None, H, V), ...]

    rescored = []
    for stored_ratio, H, V in bag:
        rects = build_rects(H, V)
        lp, ilp, x_vals, y_vals = solve_lp_ilp_with_vals(rects, threads=args.threads)
        ratio = (lp/ilp) if ilp > 0 else float("nan")
        rescored.append((ratio, lp, ilp, H, V, x_vals, y_vals, stored_ratio))

    # Sort by recomputed ratio desc, then by higher LP, then by smaller ILP
    rescored.sort(key=lambda t: (-(t[0] if math.isfinite(t[0]) else -1e9), -t[1], t[2]))

    if args.list and args.list > 0:
        K = min(args.list, len(rescored))
        print(f"\n[top {K} (recomputed LP/ILP)]")
        for i in range(K):
            ratio, lp, ilp, H, V, *_ = rescored[i]
            n = max(H) if H else 0
            print(f"#{i+1:2d}  n={n:2d}  LP={lp:.3f}  ILP={ilp:.3f}  LP/ILP={ratio:.4f}")

    idx = max(0, min(len(rescored)-1, args.rank-1))
    ratio, lp, ilp, H, V, x_vals, y_vals, stored_ratio = rescored[idx]
    n = max(H) if H else 0
    print(f"\n[rank {args.rank}] n={n}  LP={lp:.3f}  ILP={ilp:.3f}  LP/ILP={ratio:.4f}"
          + (f"  (stored={stored_ratio:.4f})" if stored_ratio is not None else ""))

    if not args.no_plot:
        title = f"Rectangles (rank {args.rank}, n={n})  LP={lp:.2f}  ILP={ilp:.2f}  LP/ILP={ratio:.3f}"
        outpng = args.out or f"rect_plot_rank{args.rank}_n{n}.png"
        rects = build_rects(H, V)
        plot_rects(rects, y_vals, title, outpng)

if __name__ == "__main__":
    main()
