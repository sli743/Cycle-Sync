from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .graph_utils import direction_tensor, sample_common_neighbors, edge_lookup

@dataclass
class TAABResult:
    scores: np.ndarray
    samples: np.ndarray
    has_cycle: np.ndarray
    raw_cycle_scores: np.ndarray

def truncated_aab_scores(adj, edges, gamma, *, nsample=200, sinmin=0.6, niter=5,
                          tau_start=1.0, tau_rate=2.0, tau_max=20.0,
                          normalize_by_pi=True, seed=None) -> TAABResult:
    """T-AAB initialization, porting the AAB block of LocationEstimByIRLUD_AAB.m."""
    rng = np.random.default_rng(seed)
    n, m = adj.shape[0], len(edges)
    G = direction_tensor(gamma, edges, n)
    samples, has = sample_common_neighbors(adj, edges, nsample, rng, require_well_shaped=True, gamma_tensor=G, sinmin=sinmin)
    SAAB = np.full((nsample,m), np.pi/2)
    for e,(i,j) in enumerate(edges):
        if not has[e]:
            continue
        ks = samples[:,e]
        Xki = G[:,i,ks]            # gamma_ki
        Xjk = -G[:,j,ks]           # gamma_jk
        gij = gamma[:,e:e+1]
        X = np.sum(Xki*gij, axis=0)
        Y = np.sum(Xjk*gij, axis=0)
        Z = np.sum(Xki*Xjk, axis=0)
        S = ((X < Y*Z) & (Y < X*Z)).astype(float)
        denom = np.where(np.abs(1-Z*Z)<1e-12, 1e-12, 1-Z*Z)
        arg = S*(X*X + Y*Y - 2*X*Y*Z)/denom + (S-1.0)*np.minimum(X,Y)
        SAAB[:,e] = np.abs(np.arccos(np.clip(arg, -1, 1)))
    scores = np.mean(SAAB, axis=0)
    lookup = edge_lookup(edges, n)
    tau = tau_start
    for _ in range(niter):
        tau = min(tau*tau_rate, tau_max)
        new = scores.copy()
        for e,(i,j) in enumerate(edges):
            if not has[e]:
                continue
            ks = samples[:,e]
            ik = np.abs(lookup[i,ks]) - 1
            jk = np.abs(lookup[j,ks]) - 1
            w = np.exp(-tau*(scores[ik]+scores[jk]))
            sw = np.sum(w)
            w = np.ones_like(w)/len(w) if sw <= 1e-300 or not np.isfinite(sw) else w/sw
            new[e] = float(np.sum(w*SAAB[:,e]))
        scores = new
    if normalize_by_pi:
        scores = scores/np.pi
        raw = SAAB/np.pi
    else:
        raw = SAAB
    return TAABResult(scores, samples, has, raw)
