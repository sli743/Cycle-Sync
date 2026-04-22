from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass
class SyntheticData:
    adj: np.ndarray
    edges: np.ndarray       # (m,2), zero-based, i<j
    gamma: np.ndarray       # (3,m), observed direction t_i-t_j
    gamma_gt: np.ndarray
    t_gt: np.ndarray        # (3,n)
    true_error: np.ndarray
    edge_len: np.ndarray
    corrupted: np.ndarray
    model: str

def normalize_cols(x, eps=1e-15):
    return x / np.maximum(np.linalg.norm(x, axis=0), eps)

def erdos_renyi_edges(n:int, p:float, rng:np.random.Generator):
    # MATLAB generators use tril(rand(n,n)<p,-1), then Ind_i=cols, Ind_j=rows.
    G = np.tril(rng.random((n,n)) < p, k=-1)
    rows, cols = np.nonzero(G)
    edges = np.column_stack([cols, rows]).astype(int)
    adj = np.zeros((n,n), dtype=float)
    adj[rows, cols] = 1.0
    adj[cols, rows] = 1.0
    return adj, edges

def uniform_corruption_model(n:int, p:float, q:float, sigma:float, *, dist:Literal["uniform","gaussian"]="uniform", seed:int|None=None) -> SyntheticData:
    """Port of UniformCorruptionModel.m."""
    rng = np.random.default_rng(seed)
    adj, edges = erdos_renyi_edges(n,p,rng)
    i, j = edges[:,0], edges[:,1]
    m = len(edges)
    t_gt = rng.standard_normal((3,n))
    raw = t_gt[:,i] - t_gt[:,j]
    edge_len = np.linalg.norm(raw, axis=0)
    gamma_gt = raw / np.maximum(edge_len, 1e-15)
    gamma = gamma_gt.copy()
    noise_ind = rng.random(m) >= q
    corrupted = ~noise_ind
    noise = rng.standard_normal((3,m))
    if dist == "uniform":
        noise = normalize_cols(noise)
    elif dist != "gaussian":
        raise ValueError("dist must be 'uniform' or 'gaussian'")
    gamma[:, noise_ind] = gamma_gt[:, noise_ind] + sigma * noise[:, noise_ind]
    gamma[:, corrupted] = noise[:, corrupted]
    gamma = normalize_cols(gamma)
    true_error = np.abs(np.arccos(np.clip(np.sum(gamma_gt*gamma, axis=0), -1, 1)))
    return SyntheticData(adj, edges, gamma, gamma_gt, t_gt, true_error, edge_len, corrupted, "uniform")

def adversarial_corruption_model(n:int, p:float, q:float, sigma:float, *, seed:int|None=None) -> SyntheticData:
    """Port of AdvCorruptionModel.m; corrupted edges are cycle-consistent."""
    rng = np.random.default_rng(seed)
    adj, edges = erdos_renyi_edges(n,p,rng)
    i, j = edges[:,0], edges[:,1]
    m = len(edges)
    tall = rng.standard_normal((3,2*n))
    t_gt, t_adv = tall[:,:n], tall[:,n:]
    raw = t_gt[:,i] - t_gt[:,j]
    raw_adv = t_adv[:,i] - t_adv[:,j]
    edge_len = np.linalg.norm(raw, axis=0)
    gamma_gt = raw / np.maximum(edge_len, 1e-15)
    gamma_adv = raw_adv / np.maximum(np.linalg.norm(raw_adv, axis=0), 1e-15)
    noise_ind = rng.random(m) >= q
    corrupted = ~noise_ind
    gamma = gamma_gt.copy()
    gamma[:, corrupted] = gamma_adv[:, corrupted]
    noise = normalize_cols(rng.standard_normal((3,m)))
    gamma = normalize_cols(gamma + sigma * noise)
    true_error = np.abs(np.arccos(np.clip(np.sum(gamma_gt*gamma, axis=0), -1, 1)))
    return SyntheticData(adj, edges, gamma, gamma_gt, t_gt, true_error, edge_len, corrupted, "adversarial")
