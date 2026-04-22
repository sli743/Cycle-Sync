from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class WLSSolution:
    t: np.ndarray
    alpha: np.ndarray
    residual_vec: np.ndarray
    residual_norms: np.ndarray
    status: int
    cost: float

def _solve_centered_system(H, b, n):
    """Solve H t=b with sum_i t_i=0 using a 3-row KKT system."""
    Aeq = np.kron(np.ones((1,n)), np.eye(3))
    K = np.block([[H, Aeq.T], [Aeq, np.zeros((3,3))]])
    rhs = np.r_[b, np.zeros(3)]
    try:
        x = np.linalg.solve(K, rhs)[:3*n]
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(K, rhs, rcond=1e-12)[0][:3*n]
    return x.reshape(n,3).T

def _assemble_active_system(edges, gamma, weights, n, active):
    """Assemble normal equations for the current active alpha set.

    active[e]=True means alpha_e is fixed to 1. Otherwise alpha_e is free and
    eliminated, producing a perpendicular-projection term.
    """
    H = np.zeros((3*n, 3*n))
    b = np.zeros(3*n)
    I = np.eye(3)
    for e,(i,j) in enumerate(edges):
        w = float(weights[e])
        if w <= 0 or not np.isfinite(w):
            continue
        g = gamma[:,e]
        if active[e]:
            P = I
            rhs = g
        else:
            P = I - np.outer(g,g)
            rhs = np.zeros(3)
        ii = slice(3*i,3*i+3); jj = slice(3*j,3*j+3)
        H[ii,ii] += w*P; H[jj,jj] += w*P
        H[ii,jj] -= w*P; H[jj,ii] -= w*P
        if active[e]:
            b[ii] += w*rhs
            b[jj] -= w*rhs
    return H, b

def solve_translation_wls(edges, gamma, weights, n, *, alpha_lower=1.0, max_iter=50, tol=1e-10) -> WLSSolution:
    """Solve min sum w_e ||t_i-t_j-alpha_e gamma_e||^2, alpha_e>=1, sum_i t_i=0.

    This active-set solver is algebraically equivalent to the MATLAB quadprog
    subproblem but much faster for synthetic experiments. For fixed active
    alpha_e=1 constraints, free alphas are eliminated in closed form by projecting
    t_i-t_j onto gamma_e_perp.
    """
    m = len(edges)
    active = np.ones(m, dtype=bool)  # start with alpha_e=1
    last = None
    t = np.zeros((3,n))
    status = 0
    for it in range(max_iter):
        H,b = _assemble_active_system(edges, gamma, weights, n, active)
        t = _solve_centered_system(H + 1e-12*np.eye(3*n), b, n)
        x = t[:,edges[:,0]] - t[:,edges[:,1]]
        alpha_star = np.sum(gamma*x, axis=0)
        new_active = alpha_star <= alpha_lower + tol
        if last is not None and np.array_equal(new_active, active):
            status = 1
            active = new_active
            break
        last = active
        active = new_active
    x = t[:,edges[:,0]] - t[:,edges[:,1]]
    alpha = np.maximum(alpha_lower, np.sum(gamma*x, axis=0))
    residual_vec = x - gamma*alpha[None,:]
    residual_norms = np.linalg.norm(residual_vec, axis=0)
    cost = float(np.sum(weights*residual_norms**2))
    return WLSSolution(t, alpha, residual_vec, residual_norms, status, cost)
