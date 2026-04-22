from __future__ import annotations
import numpy as np

def edge_lookup(edges: np.ndarray, n: int) -> np.ndarray:
    """lookup[i,j]=e+1 if edges[e]=(i,j), lookup[j,i]=-(e+1)."""
    L = np.zeros((n,n), dtype=int)
    for e,(i,j) in enumerate(edges):
        L[i,j] = e+1
        L[j,i] = -(e+1)
    return L

def oriented_gamma(gamma: np.ndarray, lookup: np.ndarray, u:int, v:int) -> np.ndarray:
    """Return observed direction t_u-t_v for any ordered measured pair."""
    idx = lookup[u,v]
    if idx == 0:
        raise KeyError((u,v))
    return gamma[:,idx-1] if idx > 0 else -gamma[:,-idx-1]

def direction_tensor(gamma: np.ndarray, edges: np.ndarray, n:int) -> np.ndarray:
    """G[:,u,v] = observed direction t_v-t_u, matching the MATLAB tensor."""
    G = np.zeros((3,n,n))
    for e,(i,j) in enumerate(edges):
        G[:,j,i] = gamma[:,e]
        G[:,i,j] = -gamma[:,e]
    return G

def sample_common_neighbors(adj, edges, nsample, rng, *, require_well_shaped=False, gamma_tensor=None, sinmin=0.6):
    m = len(edges)
    samples = -np.ones((nsample,m), dtype=int)
    has = np.zeros(m, dtype=bool)
    thresh = np.sqrt(max(0.0, 1.0 - sinmin**2))
    for e,(i,j) in enumerate(edges):
        c = np.flatnonzero((adj[:,i] > 0) & (adj[:,j] > 0))
        c = c[(c!=i)&(c!=j)]
        if require_well_shaped and gamma_tensor is not None and c.size:
            a = gamma_tensor[:,i,c]
            b = gamma_tensor[:,j,c]
            good = c[np.abs(np.sum(a*b, axis=0)) < thresh]
            if good.size:
                c = good
        if c.size:
            samples[:,e] = rng.choice(c, size=nsample, replace=True)
            has[e] = True
    return samples, has
