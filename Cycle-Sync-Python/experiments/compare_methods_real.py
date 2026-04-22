#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclesync.real_data import list_real_dataset_paths
from cyclesync.real_experiments import run_real_method_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real_precomputed")
    ap.add_argument("--out-dir", default="results/real_method_comparison")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--alignment", default="robust", choices=["robust", "l1", "l2"])
    args = ap.parse_args()
    paths = list_real_dataset_paths(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_real_method_comparison(paths, seed=args.seed, fast=args.fast, alignment=args.alignment)
    df.to_csv(out_dir / "real_method_by_dataset.csv", index=False)
    summary = df.groupby("method", dropna=False).agg(
        avg_median_error=("median_error", "mean"),
        median_of_medians=("median_error", "median"),
        avg_mean_error=("mean_error", "mean"),
        avg_runtime_sec=("runtime_sec", "mean"),
    ).reset_index().sort_values("avg_median_error")
    summary.to_csv(out_dir / "real_method_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
