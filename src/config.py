# src/config.py
import argparse

def get_args():
    ap = argparse.ArgumentParser(description="MISR PatternBoost (framework-backed)")

    # run control
    ap.add_argument("--seed", type=int, default=123, help="global random seed")
    ap.add_argument("--out_root", type=str, default="runs", help="root directory for outputs")

    # problem size & curriculum
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--lift_step", type=int, default=3)

    # per-size loop
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=32)
    ap.add_argument("--local_time_per_seed", type=float, default=3.0)

    # training
    ap.add_argument("--elites_to_train", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    # LP/ILP blend for scoring
    ap.add_argument("--alpha_lp", type=float, default=0.15)
    ap.add_argument("--beta_ilp", type=float, default=0.10)
    ap.add_argument("--grb_threads", type=int, default=0)

    # backends / proposers
    ap.add_argument("--backend", type=str, default="nanogpt",
                    choices=["nanogpt", "makemore"], help="model backend")
    ap.add_argument("--use_az", action="store_true", help="enable AlphaZero-based proposer")

    # optional seed injection from pickle
    ap.add_argument("--seed_pkl", type=str, default="",
                    help="pickle of (ratio,H,V) or (H,V) to inject at n_start")
    ap.add_argument("--seed_top_k", type=int, default=4, help="how many seeds to inject")

    return ap.parse_args()
