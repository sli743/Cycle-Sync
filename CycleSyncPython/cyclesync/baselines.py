from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp
import networkx as nx
from .wls import solve_translation_wls


@dataclass
class BaselineResult:
    t: np.ndarray
    runtime: float
    info: dict


def _safe_solve(A: np.ndarray, b: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.size == 0:
        return np.empty(0)
    A = np.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
    b = np.nan_to_num(b, nan=0.0, posinf=1e12, neginf=-1e12)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=rcond)[0]


def _solve_kkt(H: np.ndarray, Aeq: np.ndarray, rhs: np.ndarray, beq: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    Aeq = np.asarray(Aeq, dtype=float)
    rhs = np.asarray(rhs, dtype=float).ravel()
    beq = np.asarray(beq, dtype=float).ravel()
    K = np.block([[H, Aeq.T], [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
    sol = _safe_solve(K, np.r_[rhs, beq])
    return sol[: H.shape[0]]


def _coord_mask(node_mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(node_mask, dtype=bool), 3)


def _edge_mask3(edge_mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(edge_mask, dtype=bool), 3)


def _center_nan_vector(t: np.ndarray, n: int) -> np.ndarray:
    T = t.reshape(n, 3).T
    mu = np.nanmean(T, axis=1)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    T = T - mu[:, None]
    return T.T.reshape(-1)


def _scale_to_dot_constraint(t: np.ndarray, aeq1_full: np.ndarray, target: float) -> np.ndarray:
    denom = float(np.nansum(aeq1_full * t))
    if np.isfinite(denom) and abs(denom) > 1e-12:
        return t * (target / denom)
    return t


def _extract_largest_conn_comp(Svec: np.ndarray, edges: np.ndarray, node_mask: np.ndarray):
    m = edges.shape[0]
    edge_mask = np.ones(m, dtype=bool)
    s = np.asarray(Svec).ravel()
    edge_mask &= s > 0
    edge_mask &= np.isfinite(s)
    edge_mask &= node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
    if not np.any(edge_mask):
        return edge_mask, node_mask.copy()
    G = nx.Graph()
    G.add_edges_from((int(i), int(j)) for i, j in edges[edge_mask])
    comps = list(nx.connected_components(G))
    if not comps:
        return edge_mask, node_mask.copy()
    comp = max(comps, key=len)
    new_node_mask = np.zeros_like(node_mask, dtype=bool)
    new_node_mask[list(comp)] = True
    new_edge_mask = edge_mask & new_node_mask[edges[:, 0]] & new_node_mask[edges[:, 1]]
    return new_edge_mask, new_node_mask


def lud_location(adj, edges, gamma, *, maxit=20, delt=1e-16, tol=1e-5, seed=None):
    """Least Unsquared Deviations location solver using IRLS."""
    tic = time.perf_counter()
    n, m = adj.shape[0], len(edges)
    weights = np.ones(m)
    old = 1.0
    costs = []
    stag = 0
    sol = None
    for _ in range(maxit):
        sol = solve_translation_wls(edges, gamma, weights, n, max_iter=200, tol=1e-8)
        r = sol.residual_norms
        weights = 1.0 / np.sqrt(r * r + delt)
        cost = float(np.sum(r))
        costs.append(cost)
        if abs(old - cost) / max(abs(old), 1e-15) <= tol:
            stag += 1
        else:
            stag = 0
        old = cost
        if stag > 5:
            break
    if sol is None:
        sol = solve_translation_wls(edges, gamma, weights, n, max_iter=200, tol=1e-8)
    return BaselineResult(sol.t, time.perf_counter() - tic, {"iterations": len(costs), "costs": costs})


def shapefit_location(edges, gamma, n, *, max_iters=1000, verbose=False):
    """ShapeFit ADMM solver."""
    tic = time.perf_counter()
    m, d = len(edges), 3
    E = np.zeros((m, n))
    E[np.arange(m), edges[:, 0]] = 1.0
    E[np.arange(m), edges[:, 1]] = -1.0
    v = gamma.T.copy()
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-15)
    y = np.zeros((m, d))
    x = np.zeros((n, d))
    lam = np.zeros((m, d))
    tau = 1000.0 / n
    C = E.T @ v
    S = np.vstack([E, np.ones((1, n))])
    S = S.T @ S
    Sinv = np.linalg.pinv(S)
    SinvC = Sinv @ C
    schur = -float(np.sum(C * SinvC))
    if abs(schur) < 1e-15:
        schur = -1e-15
    store = SinvC / schur

    def solve(rhs_y):
        rhs = E.T @ rhs_y
        return Sinv @ rhs + store * (float(np.sum(SinvC * rhs)) - 1.0)

    max_res = 0.0
    error = np.zeros((m, d))
    error0 = np.zeros((m, d))
    resids = []
    for it in range(1, max_iters + 1):
        x0 = x.copy()
        errorm1 = error0.copy()
        error0 = error.copy()
        x = solve(y - lam)
        y = E @ x + lam
        prods = np.sum(y * v, axis=1, keepdims=True)
        Py = v * prods
        perp = y - Py
        nr = np.linalg.norm(perp, axis=1, keepdims=True)
        scale = np.maximum(nr - 1.0 / tau, 0.0) / (nr + (nr < 1e-13))
        y = Py + perp * scale
        error = E @ x - y
        lam = lam + 1.9 * error
        resid = float(
            np.linalg.norm(error) / max(np.linalg.norm(y), 1e-15)
            + tau * np.linalg.norm(x - x0) / max(np.linalg.norm(x), 1e-15)
        )
        resids.append(resid)
        max_res = max(max_res, resid)
        rel = resid / max(max_res, 1e-15)
        denom = max(np.linalg.norm(error), 1e-15)
        if it > 5 and tau < 1e6 and np.linalg.norm(error + errorm1 - 2 * error0) / denom < 1e-3:
            lam = lam / 10.0
            tau = tau * 10.0
        if rel < 1e-5:
            break
    return BaselineResult(x.T, time.perf_counter() - tic, {"iterations": len(resids), "resids": resids})


def at0_full(edges, n):
    m = len(edges)
    rows = []
    cols = []
    vals = []
    for e, (i, j) in enumerate(edges):
        for r in range(3):
            rows += [3 * e + r, 3 * e + r]
            cols += [3 * i + r, 3 * j + r]
            vals += [1.0, -1.0]
    return sp.coo_matrix((vals, (rows, cols)), shape=(3 * m, 3 * n)).tocsr()


def bata_location(
    edges,
    gamma,
    n,
    *,
    seed=None,
    delta=1e-6,
    numofiterinit=10,
    numofouteriter=10,
    numofinneriter=10,
    robustthre=1e-1,
):
    """BATA location solver."""
    tic = time.perf_counter()
    rng = np.random.default_rng(seed)
    m = len(edges)
    tij = -gamma.copy()
    tij_vec = tij.T.reshape(-1)
    At = at0_full(edges, n)
    Aeq1 = np.asarray(tij_vec @ At).ravel()
    Aeq2 = np.kron(np.ones((1, n)), np.eye(3))
    Aeq = np.vstack([Aeq1, Aeq2])
    beq = np.r_[float(m), np.zeros(3)]
    Svec = rng.random(m)
    Svec = Svec / max(np.sum(Svec), 1e-15) * m
    Wvec = np.ones(m)
    tvec = np.zeros(3 * n)
    for _ in range(numofiterinit):
        sw = np.repeat(np.sqrt(np.maximum(Wvec, 0.0)), 3)
        S = np.repeat(Svec, 3)
        A = At.multiply(sw[:, None])
        B = sw * S * tij_vec
        tvec = _solve_kkt(2 * (A.T @ A).toarray(), Aeq, 2 * (A.T @ B), beq)
        Aij = (At @ tvec).reshape(m, 3).T
        Svec = np.sum(Aij * tij, axis=0) / np.maximum(np.sum(tij * tij, axis=0), 1e-15)
        tmp = Aij - tij * Svec[None, :]
        Wvec = (np.sum(tmp * tmp, axis=0) + delta) ** (-0.5)
    tmat = tvec.reshape(n, 3).T
    tmat = tmat - tmat[:, [0]]
    At_red = At[:, 3:]
    t = tmat[:, 1:].T.reshape(-1)
    Svec = np.maximum(np.abs(Svec), 1e-15)
    for _ in range(1, numofouteriter):
        A = At_red.multiply(np.repeat(1.0 / Svec, 3)[:, None])
        tmp = (A @ t - tij_vec).reshape(m, 3)
        tmpn = np.linalg.norm(tmp, axis=1)
        Wvec = np.ones(m)
        mask = tmpn > robustthre
        Wvec[mask] = robustthre / np.maximum(tmpn[mask], 1e-15)
        for _ in range(numofinneriter):
            Aij = (At_red @ t).reshape(m, 3).T
            denom = np.sum(Aij * tij, axis=0)
            Svec = np.sum(Aij * Aij, axis=0) / np.maximum(denom, 1e-15)
            Svec[Svec < 0] = np.inf
            Ssafe = np.where(np.isfinite(Svec), Svec, 1e15)
            A2 = At_red.multiply(np.repeat(np.sqrt(Wvec) / Ssafe, 3)[:, None])
            B = np.repeat(np.sqrt(Wvec), 3) * tij_vec
            H = (A2.T @ A2).toarray() + 1e-12 * np.eye(3 * (n - 1))
            rhs = A2.T @ B
            t = _safe_solve(H, rhs)
    tout = np.zeros((3, n))
    tout[:, 1:] = t.reshape(n - 1, 3).T
    return BaselineResult(tout, time.perf_counter() - tic, {"method": "BATA"})


def fused_ta_location(
    edges,
    gamma,
    n,
    *,
    seed=None,
    delta=1e-5,
    numofiterinit=50,
    relerrthreinit=1e-5,
    numofouteriter=100,
    numofinneriter=5,
    robustthreRLUD=1e-1,
    robustthreBATA=1e-1,
    relerrthreouter=1e-6,
    relchangethreouter=1e-5,
    relchangethreinner=1e-3,
    t_init_given=None,
):
    """Full Fused Translation Averaging solver ported from the MATLAB baseline."""
    tic = time.perf_counter()
    rng = np.random.default_rng(seed)
    m = len(edges)
    if m == 0:
        return BaselineResult(np.zeros((3, n)), time.perf_counter() - tic, {"iterations": 0})
    tij = -gamma.copy()
    tij_vec = tij.T.reshape(-1)
    At = at0_full(edges, n)
    deg = np.bincount(edges.ravel(), minlength=n)
    idx_const = int(np.argmax(deg))

    aeq1_full = np.asarray(tij_vec @ At).ravel()
    aeq2_full = np.kron(np.ones((1, n)), np.eye(3))
    Aeq = np.vstack([aeq1_full, aeq2_full])
    beq = np.r_[float(m), np.zeros(3)]

    if t_init_given is None:
        Svec = rng.random(m) + 0.5
        Svec = Svec / max(np.sum(Svec), 1e-15) * m
        S = np.repeat(Svec, 3)
        W = np.ones(3 * m)
        t = np.zeros(3 * n)
    else:
        T0 = np.asarray(t_init_given, dtype=float)
        T0 = T0 - np.mean(T0, axis=1, keepdims=True)
        t = T0.T.reshape(-1)
        denom = float(aeq1_full @ t)
        if abs(denom) > 1e-12:
            t = t * (float(m) / denom)
        Aij = (At @ t).reshape(m, 3).T
        Svec = np.sum(Aij * tij, axis=0) / np.maximum(np.sum(tij * tij, axis=0), 1e-15)
        S = np.repeat(Svec, 3)
        tmp = (At @ t - S * tij_vec).reshape(m, 3)
        Wvec = (np.sum(tmp * tmp, axis=1) + delta) ** (-0.5)
        W = np.repeat(Wvec, 3)

    err_prev = 1.0
    err_curr = 0.0
    init_iter = 0
    while init_iter < numofiterinit and abs(err_prev - err_curr) / max(abs(err_prev), 1e-15) > relerrthreinit:
        A = At.multiply(W[:, None])
        B = W * S * tij_vec
        t = _solve_kkt((A.T @ A).toarray(), Aeq, A.T @ B, beq)
        Aij = (At @ t).reshape(m, 3).T
        Svec = np.sum(Aij * tij, axis=0) / np.maximum(np.sum(tij * tij, axis=0), 1e-15)
        Svec = np.where(Svec < 0, 0.0, Svec)
        S = np.repeat(Svec, 3)
        err_prev = err_curr
        tmp = (At @ t - S * tij_vec).reshape(m, 3)
        tmp_sq = np.sum(tmp * tmp, axis=1)
        Wvec = (tmp_sq + delta) ** (-0.5)
        err_curr = float(np.sum(np.sqrt(tmp_sq)))
        W = np.repeat(np.sqrt(Wvec), 3)
        init_iter += 1

    node_mask = np.ones(n, dtype=bool)
    err_prev_rlud = 1.0
    err_curr_rlud = 0.0
    err_prev_bata = 1.0
    err_curr_bata = 0.0
    tprev = np.full_like(t, np.inf)
    outer = 0
    edge_mask = np.ones(m, dtype=bool)
    while outer < numofouteriter:
        finite_prev = np.all(np.isfinite(tprev[_coord_mask(node_mask)]))
        rel_change_ok = True
        if finite_prev:
            rel_change_ok = np.linalg.norm(tprev[_coord_mask(node_mask)] - t[_coord_mask(node_mask)]) > relchangethreouter
        rlud_ok = abs(err_prev_rlud - err_curr_rlud) / max(abs(err_prev_rlud), 1e-15) > relerrthreouter
        bata_ok = abs(err_prev_bata - err_curr_bata) / max(abs(err_prev_bata), 1e-15) > relerrthreouter
        if not (rel_change_ok and rlud_ok and bata_ok):
            break
        tprev = t.copy()
        err_prev_rlud = err_curr_rlud
        err_prev_bata = err_curr_bata

        Aij = (At @ np.nan_to_num(t, nan=0.0)).reshape(m, 3).T
        Svec_rlud = np.sum(Aij * tij, axis=0) / np.maximum(np.sum(tij * tij, axis=0), 1e-15)
        S_rlud = np.repeat(np.where(Svec_rlud < 0, 0.0, Svec_rlud), 3)
        ResErrRLUD = np.linalg.norm((At @ np.nan_to_num(t, nan=0.0) - S_rlud * tij_vec).reshape(m, 3), axis=1)
        err_curr_rlud = float(np.nansum(ResErrRLUD))
        edge_mask, node_mask = _extract_largest_conn_comp(Svec_rlud, edges, node_mask)
        if not np.any(edge_mask) or np.sum(node_mask) < 2:
            break
        num_edges_ret = int(np.sum(edge_mask))
        num_nodes_ret = int(np.sum(node_mask))
        rows = _edge_mask3(edge_mask)
        cols = _coord_mask(node_mask)
        Svec_red = Svec_rlud[edge_mask]
        At_red = At[rows, :][:, cols]
        tij_red = tij[:, edge_mask]
        tmpSc = np.repeat(Svec_red, 3)
        Wvec = 1.0 / (1.0 + (ResErrRLUD * ResErrRLUD) / (robustthreRLUD ** 2))
        W = np.repeat(Wvec[edge_mask], 3)
        Ar = At_red.multiply(np.sqrt(W)[:, None])
        Br = np.sqrt(W) * tmpSc * tij_red.T.reshape(-1)
        Aeq1 = np.asarray(tij_red.T.reshape(-1) @ At_red).ravel()
        Aeq2 = np.kron(np.ones((1, num_nodes_ret)), np.eye(3))
        Aeq_red = np.vstack([Aeq1, Aeq2])
        beq_red = np.r_[float(num_edges_ret), np.zeros(3)]
        X = _solve_kkt(2 * (Ar.T @ Ar).toarray(), Aeq_red, 2 * (Ar.T @ Br), beq_red)
        t_rlud = np.full(3 * n, np.nan)
        t_rlud[cols] = X[: 3 * num_nodes_ret]

        t_prev = tprev.copy()
        t_prev[~_coord_mask(node_mask)] = np.nan
        t_rlud[~_coord_mask(node_mask)] = np.nan
        t_rlud = _center_nan_vector(t_rlud, n)
        t_prev = _center_nan_vector(t_prev, n)
        nodes_in = np.flatnonzero(node_mask)
        edge_mask = np.isin(edges[:, 0], nodes_in) & np.isin(edges[:, 1], nodes_in)
        num_edges_ret = int(np.sum(edge_mask))
        t_rlud = _scale_to_dot_constraint(t_rlud, aeq1_full, float(num_edges_ret))
        t_prev = _scale_to_dot_constraint(t_prev, aeq1_full, float(num_edges_ret))
        t_rlud = t_rlud - np.tile(t_rlud[3 * idx_const : 3 * idx_const + 3], n)
        t_prev = t_prev - np.tile(t_prev[3 * idx_const : 3 * idx_const + 3], n)
        node_mask_temp = node_mask.copy()
        node_mask_temp[idx_const] = False
        temp_cols = _coord_mask(node_mask_temp)
        rows = _edge_mask3(edge_mask)
        At_temp = At[rows, :][:, temp_cols]

        def _rlud_hessian(tvec):
            Aijh = (At @ np.nan_to_num(tvec, nan=0.0)).reshape(m, 3).T
            Svec_h = np.sum(Aijh * tij, axis=0) / np.maximum(np.sum(tij * tij, axis=0), 1e-15)
            S_h = np.repeat(np.where(Svec_h < 0, 0.0, Svec_h), 3)
            Res = np.linalg.norm((At @ np.nan_to_num(tvec, nan=0.0) - S_h * tij_vec).reshape(m, 3), axis=1)
            Wv = 1.0 / (1.0 + (Res * Res) / (robustthreRLUD ** 2))
            Wloc = np.repeat(Wv[edge_mask], 3)
            Arh = At_temp.multiply(np.sqrt(Wloc)[:, None])
            return (Arh.T @ Arh).toarray()

        H_rlud = _rlud_hessian(t_rlud)
        H_rlud_prev = _rlud_hessian(t_prev)
        t_fused_temp = _safe_solve(
            H_rlud + H_rlud_prev + 1e-12 * np.eye(H_rlud.shape[0]),
            H_rlud @ t_rlud[temp_cols] + H_rlud_prev @ t_prev[temp_cols],
        )
        t = np.full(3 * n, np.nan)
        t[temp_cols] = t_fused_temp
        t[3 * idx_const : 3 * idx_const + 3] = 0.0

        Aij = (At @ np.nan_to_num(t, nan=0.0)).reshape(m, 3).T
        denom = np.maximum(np.sum(Aij * Aij, axis=0), 1e-15)
        Svec_bata = np.sum(Aij * tij, axis=0) / denom
        S_bata = np.repeat(np.where(Svec_bata < 0, 0.0, Svec_bata), 3)
        ResErrBATA = np.linalg.norm((sp.diags(S_bata) @ At @ np.nan_to_num(t, nan=0.0) - tij_vec).reshape(m, 3), axis=1)
        err_curr_bata = float(np.nansum(ResErrBATA))
        Wvec_bata = 1.0 / (1.0 + (ResErrBATA / robustthreBATA) ** 2)
        t_bata = t.copy()
        inner = 0
        tprev_in = np.full_like(t_bata, np.inf)
        while inner < numofinneriter:
            finite_in = np.all(np.isfinite(tprev_in[_coord_mask(node_mask)]))
            if finite_in and np.linalg.norm(tprev_in[_coord_mask(node_mask)] - t_bata[_coord_mask(node_mask)]) <= relchangethreinner:
                break
            tprev_in = t_bata.copy()
            edge_mask, node_mask = _extract_largest_conn_comp(Svec_bata, edges, node_mask)
            if not np.any(edge_mask) or np.sum(node_mask) < 2:
                break
            node_mask_temp = node_mask.copy()
            node_mask_temp[idx_const] = False
            rows = _edge_mask3(edge_mask)
            cols_temp = _coord_mask(node_mask_temp)
            Svec_red = Svec_bata[edge_mask]
            At_red = At[rows, :][:, cols_temp]
            tij_red = tij[:, edge_mask]
            tmpSc = np.repeat(Svec_red, 3)
            W = np.repeat(Wvec_bata[edge_mask], 3)
            Ar = At_red.multiply((np.sqrt(W) * tmpSc)[:, None])
            Br = np.sqrt(W) * tij_red.T.reshape(-1)
            H = (Ar.T @ Ar).toarray() + 1e-12 * np.eye(Ar.shape[1])
            X = _safe_solve(H, Ar.T @ Br)
            t_bata = np.full(3 * n, np.nan)
            t_bata[cols_temp] = X
            t_bata[3 * idx_const : 3 * idx_const + 3] = 0.0
            Aij = (At @ np.nan_to_num(t_bata, nan=0.0)).reshape(m, 3).T
            denom = np.maximum(np.sum(Aij * Aij, axis=0), 1e-15)
            Svec_bata = np.sum(Aij * tij, axis=0) / denom
            inner += 1

        t_prior = t.copy()
        t_prior[~_coord_mask(node_mask)] = np.nan
        t_bata[~_coord_mask(node_mask)] = np.nan
        t_prior = _center_nan_vector(t_prior, n)
        t_bata = _center_nan_vector(t_bata, n)
        nodes_in = np.flatnonzero(node_mask)
        edge_mask = np.isin(edges[:, 0], nodes_in) & np.isin(edges[:, 1], nodes_in)
        num_edges_ret = int(np.sum(edge_mask))
        t_prior = _scale_to_dot_constraint(t_prior, aeq1_full, float(num_edges_ret))
        t_bata = _scale_to_dot_constraint(t_bata, aeq1_full, float(num_edges_ret))
        t_prior = t_prior - np.tile(t_prior[3 * idx_const : 3 * idx_const + 3], n)
        t_bata = t_bata - np.tile(t_bata[3 * idx_const : 3 * idx_const + 3], n)
        node_mask_temp = node_mask.copy()
        node_mask_temp[idx_const] = False
        temp_cols = _coord_mask(node_mask_temp)
        rows = _edge_mask3(edge_mask)
        At_temp = At[rows, :][:, temp_cols]

        def _bata_hessian(tvec):
            Aijh = (At @ np.nan_to_num(tvec, nan=0.0)).reshape(m, 3).T
            denom_h = np.maximum(np.sum(Aijh * Aijh, axis=0), 1e-15)
            Svec_h = np.sum(Aijh * tij, axis=0) / denom_h
            S_h = np.repeat(np.where(Svec_h < 0, 0.0, Svec_h), 3)
            Res = np.linalg.norm((sp.diags(S_h) @ At @ np.nan_to_num(tvec, nan=0.0) - tij_vec).reshape(m, 3), axis=1)
            Wv = 1.0 / (1.0 + (Res / robustthreBATA) ** 2)
            Sred = np.repeat(Svec_h[edge_mask], 3)
            Wloc = np.repeat(Wv[edge_mask], 3)
            Arh = At_temp.multiply((np.sqrt(Wloc) * Sred)[:, None])
            return (Arh.T @ Arh).toarray()

        H_bata_prior = _bata_hessian(t_prior)
        H_bata = _bata_hessian(t_bata)
        t_fused_temp = _safe_solve(
            H_bata_prior + H_bata + 1e-12 * np.eye(H_bata.shape[0]),
            H_bata_prior @ t_prior[temp_cols] + H_bata @ t_bata[temp_cols],
        )
        t = np.full(3 * n, np.nan)
        t[temp_cols] = t_fused_temp
        t[3 * idx_const : 3 * idx_const + 3] = 0.0
        outer += 1

    T = np.nan_to_num(t.reshape(n, 3).T, nan=0.0)
    T = T - np.mean(T, axis=1, keepdims=True)
    nodes = np.flatnonzero(node_mask)
    edge_mask = np.isin(edges[:, 0], nodes) & np.isin(edges[:, 1], nodes)
    return BaselineResult(
        T,
        time.perf_counter() - tic,
        {"method": "FusedTA", "iterations": outer, "retained_edges": int(np.sum(edge_mask)), "retained_nodes": int(np.sum(node_mask))},
    )
