#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclesync.experiments import ExperimentConfig, run_methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="uniform", choices=["uniform", "adversarial", "adv"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--qlist", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--alignment", default="robust", choices=["robust", "l1", "l2"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-fused-ta", action="store_true")
    ap.add_argument("--out-dir", default="results/synthetic_sweep")
    args = ap.parse_args()
    rows = []
    for q in args.qlist:
        for trial in range(args.trials):
            cfg = ExperimentConfig(model=args.model, n=args.n, p=args.p, q=q, sigma=args.sigma,
                                   seed=2025 + trial, alignment=args.alignment,
                                   run_fused_ta=not args.skip_fused_ta, fast=args.fast)
            df, _ = run_methods(cfg)
            df["trial"] = trial
            rows.append(df)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out / "synthetic_sweep_by_trial.csv", index=False)
    summary = all_df.groupby(["method", "q"]).agg(
        median_error_mean=("median_error", "mean"),
        median_error_std=("median_error", "std"),
        mean_runtime_sec=("runtime_sec", "mean"),
    ).reset_index()
    summary.to_csv(out / "synthetic_sweep_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
