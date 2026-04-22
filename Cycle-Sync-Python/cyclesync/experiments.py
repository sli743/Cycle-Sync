from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .data import uniform_corruption_model, adversarial_corruption_model
from .cyclesync import cycle_sync_location, CycleSyncParams
from .baselines import lud_location, shapefit_location, bata_location, fused_ta_location
from .align import camera_errors


@dataclass
class ExperimentConfig:
    model: str = "uniform"
    n: int = 100
    p: float = 0.5
    q: float = 0.8
    sigma: float = 0.0
    seed: int = 2025
    alignment: str = "robust"
    run_fused_ta: bool = True
    fast: bool = False


def make_data(cfg: ExperimentConfig):
    if cfg.model == "uniform":
        return uniform_corruption_model(cfg.n, cfg.p, cfg.q, cfg.sigma, seed=cfg.seed)
    if cfg.model in {"adv", "adversarial"}:
        return adversarial_corruption_model(cfg.n, cfg.p, cfg.q, cfg.sigma, seed=cfg.seed)
    raise ValueError("model must be uniform or adversarial")


def run_methods(cfg: ExperimentConfig, methods=None):
    if methods is None:
        methods = ["Cycle-Sync", "LUD", "ShapeFit", "BATA"] + (["FusedTA"] if cfg.run_fused_ta else [])
    data = make_data(cfg)
    rows = []
    outputs = {"data": data}
    sample_opts = dict(taab_nsample=50, cycle_nsample=50, tmax=5) if cfg.fast else dict()
    for method in methods:
        if method == "Cycle-Sync":
            res = cycle_sync_location(data.adj, data.edges, data.gamma, CycleSyncParams(seed=cfg.seed + 1000, **sample_opts))
            t_est = res.t
            runtime = res.runtime
            outputs[method] = res
            extra = {"iterations": len(res.history)}
        elif method == "LUD":
            res = lud_location(data.adj, data.edges, data.gamma, maxit=20, delt=1e-16)
            t_est = res.t
            runtime = res.runtime
            outputs[method] = res
            extra = {"iterations": res.info.get("iterations")}
        elif method == "ShapeFit":
            res = shapefit_location(data.edges, data.gamma, cfg.n, max_iters=1000)
            t_est = res.t
            runtime = res.runtime
            outputs[method] = res
            extra = {"iterations": res.info.get("iterations")}
        elif method == "BATA":
            res = bata_location(data.edges, data.gamma, cfg.n, seed=cfg.seed + 2000)
            t_est = res.t
            runtime = res.runtime
            outputs[method] = res
            extra = {}
        elif method == "FusedTA":
            fta_opts = dict(numofiterinit=5, numofouteriter=3, numofinneriter=2) if cfg.fast else dict()
            res = fused_ta_location(data.edges, data.gamma, cfg.n, seed=cfg.seed + 3000, **fta_opts)
            t_est = res.t
            runtime = res.runtime
            outputs[method] = res
            extra = res.info
        else:
            raise ValueError(method)
        err = camera_errors(t_est, data.t_gt, method=cfg.alignment)
        extra = {k: v for k, v in extra.items() if k != "method"}
        rows.append({
            "method": method,
            "median_error": err["median"],
            "mean_error": err["mean"],
            "trimmed_mean_error": err["trimmed_mean"],
            "q25_error": err["q25"],
            "q75_error": err["q75"],
            "nrmse": err["nrmse"],
            "runtime_sec": runtime,
            "n": cfg.n,
            "p": cfg.p,
            "q": cfg.q,
            "sigma": cfg.sigma,
            "model": cfg.model,
            "seed": cfg.seed,
            "alignment": cfg.alignment,
            **extra,
        })
        outputs[f"{method}_errors"] = err
    return pd.DataFrame(rows), outputs
