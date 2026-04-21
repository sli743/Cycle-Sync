#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclesync.real_data import load_real_dataset, evaluate_real_locations
from cyclesync.real_experiments import default_cycle_params
from cyclesync.cyclesync import cycle_sync_location


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat_file", nargs="?", default="data/real_precomputed/delivery_area_location(2).mat")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--alignment", default="robust", choices=["robust", "l1", "l2"])
    args = ap.parse_args()
    data = load_real_dataset(args.mat_file)
    res = cycle_sync_location(data.adj, data.edges, data.gamma, default_cycle_params(fast=args.fast))
    err = evaluate_real_locations(res.t, data, alignment=args.alignment)
    print(f"Dataset: {data.name}  n={data.n}  m={data.m}")
    print(f"Alignment: {args.alignment}")
    print(f"Median error: {err['median']:.6g}")
    print(f"Mean error:   {err['mean']:.6g}")
    print(f"Runtime:      {res.runtime:.3f} sec")


if __name__ == "__main__":
    main()
