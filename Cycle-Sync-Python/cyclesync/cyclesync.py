from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from .taab import truncated_aab_scores
from .graph_utils import edge_lookup, sample_common_neighbors
from .wls import solve_translation_wls


@dataclass
class CycleSyncParams:
    """Default parameters for the released Cycle-Sync location solver."""
    tmax: int = 20
    beta: float = 20.0
    delta: float = 1e-8
    loss_a: float = 4.0
    lambda_offset: float = 10.0
    init_weight_scale: float = 20.0
    taab_nsample: int = 200
    cycle_nsample: int = 100
    sinmin: float = 0.6
    taab_iters: int = 5
    wls_max_iter: int = 200
    wls_tol: float = 1e-8
    seed: int | None = None
    use_taab_init: bool = True
    normalize_taab_by_pi: bool = True
    no_cycle_value: float = 1.0
    score_clip_min: float = 1e-8
    score_clip_max: float = 10.0


@dataclass
class CycleSyncResult:
    t: np.ndarray
    alpha: np.ndarray
    weights: np.ndarray
    residual_norms: np.ndarray
    cycle_scores: np.ndarray
    init_scores: np.ndarray
    history: list[dict] = field(default_factory=list)
    runtime: float = 0.0
    params: CycleSyncParams | None = None


def _weights_from_scores(h: np.ndarray, p: CycleSyncParams) -> np.ndarray:
    h = np.clip(h, p.score_clip_min, p.score_clip_max)
    return np.exp(-p.loss_a * h) / (h + p.delta)


def cycle_scores_from_locations(adj, edges, gamma, t, residuals, *, beta, nsample, seed=None, no_cycle_value=1.0):
    """Distance-based 3-cycle scores for each measured edge."""
    rng = np.random.default_rng(seed)
    n, m = adj.shape[0], len(edges)
    lookup = edge_lookup(edges, n)
    samples, has = sample_common_neighbors(adj, edges, nsample, rng)
    if m == 0:
        return np.empty(0)
    scores = np.full(m, float(no_cycle_value))
    valid = has & np.all(samples >= 0, axis=0)
    if not np.any(valid):
        return scores

    K = samples.copy()
    K[K < 0] = 0
    i = edges[:, 0]
    j = edges[:, 1]
    ii = i[None, :]
    jj = j[None, :]
    l_ik = lookup[ii, K]
    l_jk = lookup[jj, K]
    l_ki = lookup[K, ii]

    idx_ik = np.maximum(np.abs(l_ik) - 1, 0)
    idx_jk = np.maximum(np.abs(l_jk) - 1, 0)
    idx_ki = np.maximum(np.abs(l_ki) - 1, 0)
    s_jk = np.sign(l_jk).astype(float)
    s_ki = np.sign(l_ki).astype(float)

    lens = np.linalg.norm(t[:, edges[:, 0]] - t[:, edges[:, 1]], axis=0)
    term_ij = gamma[:, :, None] * lens[None, :, None]
    term_ij = np.transpose(term_ij, (0, 2, 1))
    term_ij = np.broadcast_to(term_ij, (3, nsample, m))
    term_jk = gamma[:, idx_jk] * (s_jk[None, :, :] * lens[idx_jk][None, :, :])
    term_ki = gamma[:, idx_ki] * (s_ki[None, :, :] * lens[idx_ki][None, :, :])
    vals = np.linalg.norm(term_ij + term_jk + term_ki, axis=0)

    w = np.exp(-float(beta) * (residuals[idx_ik] + residuals[idx_jk]))
    mask = valid[None, :]
    w = np.where(mask, w, 0.0)
    vals = np.where(mask, vals, 0.0)
    sw = np.sum(w, axis=0)
    weighted = np.divide(np.sum(w * vals, axis=0), sw, out=np.full(m, np.nan), where=sw > 1e-300)
    mean_vals = np.where(valid, np.mean(vals, axis=0), float(no_cycle_value))
    out = np.where(np.isfinite(weighted), weighted, mean_vals)
    scores[valid] = out[valid]
    return scores


def cycle_sync_location(adj, edges, gamma, params: CycleSyncParams | None = None) -> CycleSyncResult:
    params = params or CycleSyncParams()

    try:
        tic = time.perf_counter()
        n, m = adj.shape[0], len(edges)
        taab = truncated_aab_scores(
            adj,
            edges,
            gamma,
            nsample=params.taab_nsample,
            sinmin=params.sinmin,
            niter=params.taab_iters,
            normalize_by_pi=params.normalize_taab_by_pi,
            seed=params.seed,
        )
        weights = np.exp(-params.init_weight_scale * taab.scores) if params.use_taab_init else np.ones(m)
        history = []
        alpha = np.ones(m)
        residuals = np.ones(m)
        sc = taab.scores.copy()
        t = np.zeros((3, n))

        for it in range(1, params.tmax + 1):
            sol = solve_translation_wls(
                edges, gamma, weights, n,
                alpha_lower=1.0,
                max_iter=params.wls_max_iter,
                tol=params.wls_tol,
            )
            t, alpha, residuals = sol.t, sol.alpha, sol.residual_norms
            sc = cycle_scores_from_locations(
                adj,
                edges,
                gamma,
                t,
                residuals,
                beta=params.beta,
                nsample=params.cycle_nsample,
                seed=None if params.seed is None else params.seed + 7919 * it,
                no_cycle_value=params.no_cycle_value,
            )
            lam = it / (it + params.lambda_offset)
            h = (1 - lam) * residuals + lam * sc
            weights = _weights_from_scores(h, params)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            if np.max(weights) <= 0:
                weights = np.ones_like(weights)

            history.append({
                "iter": it,
                "lambda": lam,
                "median_residual": float(np.median(residuals)),
                "median_cycle_score": float(np.median(sc)),
                "min_weight": float(np.min(weights)),
                "max_weight": float(np.max(weights)),
                "wls_status": sol.status,
                "wls_cost": sol.cost,
            })

        out = CycleSyncResult(
            t, alpha, weights, residuals, sc, taab.scores,
            history, time.perf_counter() - tic, params
        )

        if (
            not np.all(np.isfinite(out.t))
            or not np.all(np.isfinite(out.weights))
            or not np.all(np.isfinite(out.residual_norms))
        ):
            raise FloatingPointError("non-finite Cycle-Sync output")

        return out

    except Exception:
        if params.sinmin > 0:
            params0 = CycleSyncParams(**{**vars(params), "sinmin": 0.0})
            return cycle_sync_location(adj, edges, gamma, params0)
        raise
