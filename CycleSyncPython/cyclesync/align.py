from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def align_scale_translate_l2(t_est, t_ref):
    """Signed-scale plus translation alignment in least-squares sense."""
    mu_x = np.mean(t_est, axis=1, keepdims=True)
    mu_y = np.mean(t_ref, axis=1, keepdims=True)
    X = t_est - mu_x
    Y = t_ref - mu_y
    denom = float(np.sum(X * X))
    c = 0.0 if denom < 1e-15 else float(np.sum(X * Y) / denom)
    shift = (mu_y - c * mu_x).ravel()
    fit = c * t_est + shift[:, None]
    nrmse = float(np.sqrt(np.sum((t_ref - fit) ** 2) / max(np.sum(Y * Y), 1e-15)))
    return fit, shift, c, nrmse


def align_scale_translate_l1(t_est, t_ref, *, maxiter=300):
    """CVX-free equivalent of SimpleTransScaleRemove(...,'L1') over 4 variables."""
    _, shift0, c0, _ = align_scale_translate_l2(t_est, t_ref)
    x0 = np.r_[shift0, c0]

    def obj(x):
        r = t_ref - (x[3] * t_est + x[:3, None])
        return float(np.sum(np.sqrt(np.sum(r * r, axis=0) + 1e-12)))

    out = minimize(obj, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": maxiter})
    x = out.x if np.all(np.isfinite(out.x)) else x0
    fit = x[3] * t_est + x[:3, None]
    Y = t_ref - np.mean(t_ref, axis=1, keepdims=True)
    nrmse = float(np.sqrt(np.sum((t_ref - fit) ** 2) / max(np.sum(Y * Y), 1e-15)))
    return fit, x[:3], float(x[3]), nrmse


def _weighted_scale_translate_l2(t_est: np.ndarray, t_ref: np.ndarray, weights: np.ndarray):
    """Weighted signed scale + translation closed form."""
    w = np.asarray(weights, dtype=float).ravel()
    w = np.maximum(w, 0.0)
    if np.sum(w) <= 1e-15:
        w = np.ones(t_est.shape[1])
    w = w / np.sum(w)
    mu_x = t_est @ w[:, None]
    mu_y = t_ref @ w[:, None]
    X = t_est - mu_x
    Y = t_ref - mu_y
    denom = float(np.sum(w[None, :] * X * X))
    c = 0.0 if denom < 1e-15 else float(np.sum(w[None, :] * X * Y) / denom)
    shift = (mu_y - c * mu_x).ravel()
    fit = c * t_est + shift[:, None]
    Y0 = t_ref - np.mean(t_ref, axis=1, keepdims=True)
    nrmse = float(np.sqrt(np.sum((t_ref - fit) ** 2) / max(np.sum(Y0 * Y0), 1e-15)))
    return fit, shift, c, nrmse


def align_scale_translate_robust(
    t_est: np.ndarray,
    t_ref: np.ndarray,
    *,
    trim_fraction: float = 0.80,
    max_iter: int = 50,
    cauchy_scale: float = 2.5,
):
    """Robust signed scale + translation alignment for real-data evaluation.

    This routine starts from L1 alignment and adds a trimmed Cauchy IRLS loop
    so that a few extreme camera outliers
    cannot dominate the fitted global scale or translation.  The reported camera
    errors are still computed for all cameras; only the alignment parameters are
    estimated from robustly weighted inliers.
    """
    n = t_est.shape[1]
    if n <= 4:
        return align_scale_translate_l1(t_est, t_ref)

    # Start from L1; it is a strong deterministic initialization and matches the
    # L1 is a strong deterministic initialization.
    fit, shift, scale, _ = align_scale_translate_l1(t_est, t_ref)
    w = np.ones(n)
    min_keep = min(n, max(4, int(np.ceil(trim_fraction * n))))
    last_scale = None
    for _ in range(max_iter):
        residual = np.linalg.norm(t_ref - fit, axis=0)
        med = float(np.median(residual))
        mad = float(1.4826 * np.median(np.abs(residual - med)))
        sigma = max(mad, med / 2.0, 1e-8)
        keep_idx = np.argpartition(residual, min_keep - 1)[:min_keep]
        keep = np.zeros(n, dtype=bool)
        keep[keep_idx] = True
        # Cauchy weights with hard trimming.  This is smoother than pure
        # least-trimmed squares but still suppresses extreme cameras.
        w_new = 1.0 / (1.0 + (residual / (cauchy_scale * sigma)) ** 2)
        w_new[~keep] = 0.0
        fit_new, shift, scale, _ = _weighted_scale_translate_l2(t_est, t_ref, w_new)
        if last_scale is not None and np.linalg.norm(fit_new - fit) / max(np.linalg.norm(fit), 1e-12) < 1e-8:
            w = w_new
            fit = fit_new
            break
        last_scale = scale
        w = w_new
        fit = fit_new

    Y0 = t_ref - np.mean(t_ref, axis=1, keepdims=True)
    nrmse = float(np.sqrt(np.sum((t_ref - fit) ** 2) / max(np.sum(Y0 * Y0), 1e-15)))
    inlier_mask = w > 0
    return fit, shift, float(scale), nrmse, inlier_mask, w


def camera_errors(t_est, t_ref, *, method="l1", trim_fraction: float = 0.80):
    method_l = method.lower()
    extra = {}
    if method_l == "l1":
        fit, shift, c, nrmse = align_scale_translate_l1(t_est, t_ref)
    elif method_l == "l2":
        fit, shift, c, nrmse = align_scale_translate_l2(t_est, t_ref)
    elif method_l in {"robust", "trimmed", "trimmed_cauchy", "robust_trimmed"}:
        fit, shift, c, nrmse, inlier_mask, weights = align_scale_translate_robust(
            t_est, t_ref, trim_fraction=trim_fraction
        )
        extra["alignment_inlier_mask"] = inlier_mask
        extra["alignment_weights"] = weights
        extra["alignment_inlier_fraction"] = float(np.mean(inlier_mask))
    else:
        raise ValueError(f"Unknown alignment method {method!r}")
    err = np.linalg.norm(fit - t_ref, axis=0)
    out = {
        "t_fit": fit,
        "shift": shift,
        "scale": c,
        "nrmse": nrmse,
        "errors": err,
        "median": float(np.median(err)),
        "mean": float(np.mean(err)),
        "trimmed_mean": float(np.mean(np.sort(err)[: max(1, int(np.ceil(trim_fraction * len(err))))])),
        "q25": float(np.quantile(err, 0.25)),
        "q75": float(np.quantile(err, 0.75)),
        "q90": float(np.quantile(err, 0.90)),
        "max": float(np.max(err)),
    }
    out.update(extra)
    return out
