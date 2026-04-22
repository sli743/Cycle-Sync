#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclesync.experiments import ExperimentConfig, run_methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uniform", choices=["uniform", "adversarial", "adv"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--alignment", default="robust", choices=["robust", "l1", "l2"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-fused-ta", action="store_true")
    ap.add_argument("--out-dir", default="results/demo_synthetic")
    args = ap.parse_args()
    cfg = ExperimentConfig(model=args.model, n=args.n, p=args.p, q=args.q, sigma=args.sigma,
                           seed=args.seed, alignment=args.alignment,
                           run_fused_ta=not args.skip_fused_ta, fast=args.fast)
    df, _ = run_methods(cfg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    print(df[["method", "median_error", "mean_error", "trimmed_mean_error", "runtime_sec"]].to_string(index=False))
    print(f"Saved {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
