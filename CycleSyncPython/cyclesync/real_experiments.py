from __future__ import annotations
import time
import pandas as pd
from .cyclesync import cycle_sync_location, CycleSyncParams
from .baselines import lud_location, shapefit_location, bata_location, fused_ta_location
from .real_data import load_real_dataset, evaluate_real_locations


def default_cycle_params(*, seed: int = 123, fast: bool = False) -> CycleSyncParams:
    samples = dict(taab_nsample=50, cycle_nsample=50, tmax=5) if fast else dict(taab_nsample=200, cycle_nsample=100)
    return CycleSyncParams(seed=seed, **samples)


def run_real_method_comparison(paths, *, seed: int = 123, fast: bool = True, alignment: str = "robust") -> pd.DataFrame:
    rows = []
    for path in paths:
        data = load_real_dataset(path)
        methods = [
            ("Cycle-Sync", lambda: cycle_sync_location(data.adj, data.edges, data.gamma, default_cycle_params(seed=seed, fast=fast))),
            ("LUD", lambda: lud_location(data.adj, data.edges, data.gamma, maxit=20, delt=1e-16)),
            ("ShapeFit", lambda: shapefit_location(data.edges, data.gamma, data.n, max_iters=1000)),
            ("BATA", lambda: bata_location(data.edges, data.gamma, data.n, seed=seed + 7)),
            ("FusedTA", lambda: fused_ta_location(data.edges, data.gamma, data.n, seed=seed + 13, **(dict(numofiterinit=5, numofouteriter=3, numofinneriter=2) if fast else dict()))),
        ]
        for method, func in methods:
            tic = time.perf_counter()
            try:
                res = func()
                err = evaluate_real_locations(res.t, data, alignment=alignment, apply_rotation=True)
                row = {
                    "dataset": data.name,
                    "method": method,
                    "median_error": err["median"],
                    "mean_error": err["mean"],
                    "trimmed_mean_error": err["trimmed_mean"],
                    "q25_error": err["q25"],
                    "q75_error": err["q75"],
                    "q90_error": err["q90"],
                    "alignment": alignment,
                    "n": data.n,
                    "m": data.m,
                    "runtime_sec": getattr(res, "runtime", time.perf_counter() - tic),
                }
            except Exception as exc:
                row = {
                    "dataset": data.name,
                    "method": method,
                    "median_error": float("nan"),
                    "mean_error": float("nan"),
                    "trimmed_mean_error": float("nan"),
                    "q25_error": float("nan"),
                    "q75_error": float("nan"),
                    "q90_error": float("nan"),
                    "alignment": alignment,
                    "n": data.n,
                    "m": data.m,
                    "runtime_sec": time.perf_counter() - tic,
                    "error": str(exc),
                }
            rows.append(row)
    return pd.DataFrame(rows)
