from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from .measures import Emp
from .drifts import DriftField


def epsilon_of_sigma(sigma: float) -> float:
    """eps = 2 sigma^2"""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive")
    return 2.0 * sigma ** 2


OVER_RELAX = 1.9      # only used in the exact (K < 0) branch; 1.0 = plain Sinkhorn


def _canonical_correction_count(K) -> int:
    if K is None:
        return -1
    value = float(K)
    if np.isnan(value):
        raise ValueError("K cannot be NaN")
    if not np.isfinite(value) or value < 0:
        return -1
    if not value.is_integer():
        raise ValueError("finite K must be a non-negative integer")
    return int(value)


def _positive_integer(value, name: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def sinkhorn_log_scalings(q: Emp, pi: Emp, sigma: float, K: int,
                          tol: float = 1e-8, max_iter: int = 4000,
                          log_b0: Optional[np.ndarray] = None) -> np.ndarray:
    """Run K column corrections and return `log b_K` (shape (pi.n,))."""
    K = _canonical_correction_count(K)
    exact = K < 0
    max_iter = _positive_integer(max_iter, "max_iter")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and strictly positive")
    eps = epsilon_of_sigma(sigma)
    minus_C_eps = -cdist(q.X, pi.X, "sqeuclidean") / eps      # (n_q, n_pi)
    log_wq = np.log(q.w + 1e-300)
    log_wpi = np.log(pi.w + 1e-300)

    if exact and log_b0 is not None and log_b0.shape == (pi.n,):
        log_b = log_b0.copy()
    else:
        log_b = np.zeros(pi.n)

    # The damped symmetric update g <- (g + T(g))/2 solves the
    # same fixed point in ~20x fewer iterations; the two differ by an additive
    # constant, which the final row softmax cancels.
    if exact and pi is q:
        converged = False
        for _ in range(max_iter):
            log_b_new = 0.5 * (log_b - logsumexp(minus_C_eps + (log_b + log_wq)[None, :],
                                                 axis=1))
            shift = np.max(np.abs(log_b_new - log_b))
            log_b = log_b_new
            if shift < tol:
                converged = True
                break
        if not converged:
            warnings.warn(
                f"self Sinkhorn scaling reached max_iter={max_iter} before tol={tol:g}",
                RuntimeWarning,
                stacklevel=2,
            )
        return log_b

    omega = OVER_RELAX if exact else 1.0
    converged = not exact
    for _ in range(max_iter if exact else int(K)):
        log_a = -logsumexp(minus_C_eps + (log_b + log_wpi)[None, :], axis=1)
        target = -logsumexp(minus_C_eps + (log_a + log_wq)[:, None], axis=0)
        log_b_new = log_b + omega * (target - log_b)
        shift = np.max(np.abs(log_b_new - log_b))
        log_b = log_b_new
        if exact and shift < tol:
            converged = True
            break
    if exact and not converged:
        warnings.warn(
            f"cross Sinkhorn scaling reached max_iter={max_iter} before tol={tol:g}",
            RuntimeWarning,
            stacklevel=2,
        )
    return log_b


def sinkhorn_conditional(q: Emp, pi: Emp, sigma: float, K: int,
                         log_b: Optional[np.ndarray] = None, **kw) -> np.ndarray:
    """Row-stochastic conditional K_K^pi(x_i, .) as an (n_q, n_pi) matrix."""
    if log_b is None:
        log_b = sinkhorn_log_scalings(q, pi, sigma, K, **kw)
    L = (-cdist(q.X, pi.X, "sqeuclidean") / epsilon_of_sigma(sigma)
         + (log_b + np.log(pi.w + 1e-300))[None, :])
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=1, keepdims=True)


def sinkhorn_barycentre(q: Emp, pi: Emp, sigma: float, K: int, **kw) -> np.ndarray:
    """B_K^pi(x_i) = sum_j K_K^pi(x_i, y_j) y_j.  Shape (n_q, d)."""
    return sinkhorn_conditional(q, pi, sigma, K, **kw) @ pi.X


def sinkhorn_drift(sigma: float = 1.0, K: int = 0, tol: float = 1e-8,
                   max_iter: int = 4000) -> DriftField:
    """V^(K) = B_K^p - B_K^q"""
    K_int = _canonical_correction_count(K)
    max_iter = _positive_integer(max_iter, "max_iter")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and strictly positive")
    warm: dict = {}

    def scalings(src: Emp, tgt: Emp, tag: str) -> np.ndarray:
        log_b = sinkhorn_log_scalings(src, tgt, sigma, K_int, tol=tol,
                                      max_iter=max_iter, log_b0=warm.get(tag))
        if K_int < 0:
            warm[tag] = log_b
        return log_b

    def at(pts, p, q):
        cross = sinkhorn_barycentre(q, p, sigma, K_int, log_b=scalings(q, p, "cross"))
        self_ = sinkhorn_barycentre(q, q, sigma, K_int, log_b=scalings(q, q, "self"))
        V = cross - self_
        return V if pts is q.X else V[cdist(pts, q.X).argmin(axis=1)]

    tag = r"\infty" if K_int < 0 else str(K_int)
    return DriftField(f"sinkhorn_K{'inf' if K_int < 0 else K_int}", at,
                      label=rf"Sinkhorn $K={tag}$",
                      meta=dict(sigma=sigma, K=K_int, epsilon=epsilon_of_sigma(sigma)))


def sinkhorn_divergence(q: Emp, p: Emp, sigma: float,
                        max_iter: int = 2000) -> float:
    """S_sigma(q,p) = OT_sigma(q,p) - 1/2 OT_sigma(q,q) - 1/2 OT_sigma(p,p)``
    with ``C(x,y)=1/2||x-y||^2`` and KL regularization ``sigma^2``.
    """
    
    reg = 0.5 * epsilon_of_sigma(sigma)
    max_iter = _positive_integer(max_iter, "max_iter")
    def regularized_objective(mu: Emp, nu: Emp) -> float:
        import ot

        
        keep_mu, keep_nu = mu.w > 0, nu.w > 0
        a, b = mu.w[keep_mu], nu.w[keep_nu]
        C = 0.5 * cdist(mu.X[keep_mu], nu.X[keep_nu], "sqeuclidean")
        plan = ot.sinkhorn(
            a, b, C, reg=reg, method="sinkhorn_log",
            numItermax=max_iter, stopThr=1e-11, warn=True,
        )
        
        base = a[:, None] * b[None, :]
        active = plan > 0
        kl = float(np.sum(plan[active] * (np.log(plan[active]) - np.log(base[active]))))
        return float(np.sum(plan * C) + reg * kl)

    return float(
        regularized_objective(q, p)
        - 0.5 * regularized_objective(q, q)
        - 0.5 * regularized_objective(p, p)
    )
