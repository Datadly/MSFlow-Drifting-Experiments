from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.linalg import sqrtm

ArrayLike = Union[float, np.ndarray]


# =========================================================================== #
# 1.  Gaussian-Gaussian mean-shift flow
# =========================================================================== #
def affine_drift(m: np.ndarray, Sigma: np.ndarray, m_star: np.ndarray,
                 Sigma_star: np.ndarray, sigma: float):
    """(A, b) such that the Gaussian mean-shift drift is x -> A x + b."""
    d = len(m)
    Iq = np.linalg.inv(Sigma + sigma ** 2 * np.eye(d))
    Ip = np.linalg.inv(Sigma_star + sigma ** 2 * np.eye(d))
    return sigma ** 2 * (Iq - Ip), sigma ** 2 * (Ip @ m_star - Iq @ m)


def exact_barycentre(x: np.ndarray, m: np.ndarray, Sigma: np.ndarray,
                     sigma: float) -> np.ndarray:
    """Used to test the particle mean-shift without any Monte-Carlo error."""
    x = np.atleast_2d(np.asarray(x, float))
    Si = np.linalg.inv(np.asarray(Sigma, float))
    M = np.linalg.inv(np.eye(len(Si)) / sigma ** 2 + Si)
    return (x / sigma ** 2 + np.asarray(m, float) @ Si.T) @ M.T


def exact_mean_shift(x, m_q, Sigma_q, m_p, Sigma_p, sigma) -> np.ndarray:
    """Gaussian mean-shift drift, computed exactly (no sampling)."""
    return (exact_barycentre(x, m_p, Sigma_p, sigma)
            - exact_barycentre(x, m_q, Sigma_q, sigma))


def moment_map_step(m, Sigma, m_star, Sigma_star, sigma, tau):
    """One discrete drift step, exactly as the particles see it."""
    A, b = affine_drift(m, Sigma, m_star, Sigma_star, sigma)
    M = np.eye(len(m)) + tau * A
    return M @ m + tau * b, M @ Sigma @ M.T


def moment_ode_step(m, Sigma, m_star, Sigma_star, sigma, tau):
    """One explicit-Euler step of the *continuous* moment system"""
    A, b = affine_drift(m, Sigma, m_star, Sigma_star, sigma)
    return m + tau * (A @ m + b), Sigma + tau * (A @ Sigma + Sigma @ A)


def moment_flow(m0, Sigma0, m_star, Sigma_star, sigma: float, tau: float,
                n_iter: int, scheme: str = "map") -> dict:
    """Integrate the moment system and record diagnostics."""
    step = {"map": moment_map_step, "ode": moment_ode_step}[scheme]
    m, Sigma = np.asarray(m0, float).copy(), np.asarray(Sigma0, float).copy()
    ms, Ss, w2s, phis = [m.copy()], [Sigma.copy()], [], []
    for _ in range(n_iter):
        w2s.append(W2_gaussian(m, Sigma, m_star, Sigma_star))
        phis.append(lyapunov_phi(Sigma, Sigma_star, sigma))
        m, Sigma = step(m, Sigma, m_star, Sigma_star, sigma, tau)
        ms.append(m.copy()); Ss.append(Sigma.copy())
    w2s.append(W2_gaussian(m, Sigma, m_star, Sigma_star))
    phis.append(lyapunov_phi(Sigma, Sigma_star, sigma))
    return dict(m=np.array(ms), Sigma=np.array(Ss),
                W2=np.array(w2s), Phi=np.array(phis), scheme=scheme)


def W2_gaussian(m1, S1, m2, S2) -> float:
    """Exact W_2 between two Gaussians (Bures-Wasserstein)."""
    s1 = np.real(sqrtm(S1))
    cross = np.real(sqrtm(s1 @ S2 @ s1))
    bures = np.trace(S1) + np.trace(S2) - 2 * np.trace(cross)
    return float(np.sqrt(max(np.sum((np.asarray(m1) - np.asarray(m2)) ** 2) + bures, 0.0)))


def lyapunov_phi(Sigma, Sigma_star, sigma: float) -> float:
    """Phi(Sigma) = 2 KL( N(0, Sigma + s^2 I) || N(0, Sigma* + s^2 I) )."""
    d = len(Sigma)
    S = np.asarray(Sigma, float) + sigma ** 2 * np.eye(d)
    T = np.asarray(Sigma_star, float) + sigma ** 2 * np.eye(d)
    Ti = np.linalg.inv(T)
    _, ld_S = np.linalg.slogdet(S)
    _, ld_T = np.linalg.slogdet(T)
    return float(np.trace(Ti @ S) - ld_S + ld_T - d)


def phi_dissipation(Sigma, Sigma_star, sigma: float) -> float:
    """d/dt Phi(Sigma_t) = -(2/sigma^2) || Sigma^{1/2} A_Sigma ||_F^2  <= 0."""
    d = len(Sigma)
    A = sigma ** 2 * (np.linalg.inv(Sigma + sigma ** 2 * np.eye(d))
                      - np.linalg.inv(Sigma_star + sigma ** 2 * np.eye(d)))
    root = np.real(sqrtm(Sigma))
    return float(-2.0 / sigma ** 2 * np.linalg.norm(root @ A, "fro") ** 2)


# =========================================================================== #
# 2.  Centered commuting Gaussians under K-step Sinkhorn drifting
# =========================================================================== #
def s_K(lam: ArrayLike, q: ArrayLike, sigma: float, K: int) -> np.ndarray:
    """s_K(lambda, q): the cross scaling after K Sinkhorn column corrections."""
    lam, q = np.asarray(lam, float), np.asarray(q, float)
    d = sigma ** 4 / q
    if K is None or K < 0 or not np.isfinite(K):
        return 0.5 * (-d + np.sqrt(d ** 2 + 4.0 * d * lam))
    x = sigma ** 2 * lam / (sigma ** 2 + lam)
    for _ in range(int(K)):
        x = lam * (d + x) / (lam + d + x)
    return x


def ell_K(lam: ArrayLike, q: ArrayLike, sigma: float, K: int) -> np.ndarray:
    """ell_K(lambda, q) = (s_K(lambda,q) - s_K(q,q)) / sigma^2."""
    return (s_K(lam, q, sigma, K) - s_K(q, q, sigma, K)) / sigma ** 2


def covariance_flow(q0: ArrayLike, lam: ArrayLike, sigma: float, K: int,
                    tau: float, n_iter: int) -> dict:
    """Integrate qdot_i = 2 q_i ell_K(lambda_i, q_i) with explicit Euler."""
    q = np.atleast_1d(np.asarray(q0, float)).copy()
    lam = np.atleast_1d(np.asarray(lam, float))
    traj = [q.copy()]
    for _ in range(n_iter):
        q = q + tau * 2.0 * q * ell_K(lam, q, sigma, K)
        q = np.maximum(q, 1e-12)                       # covariances stay positive
        traj.append(q.copy())
    traj = np.array(traj)
    return dict(q=traj, lam=lam, err=np.abs(traj - lam).sum(axis=1),
                S_eps=np.array([sinkhorn_divergence_1d(v, lam, sigma).sum() for v in traj]))


# --- local convergence rate Gamma_K ---------------------------------------- #
def gamma_K(alpha: ArrayLike, K: int) -> np.ndarray:
    """Local decay exponent Gamma_K as a function of alpha = lambda / sigma^2."""
    a = np.asarray(alpha, float)
    if K is None or K < 0 or not np.isfinite(K):
        return 2.0 * a / np.sqrt(1.0 + 4.0 * a ** 2)
    x = a / (1.0 + a)
    Dk = 1.0 / (1.0 + a) ** 2
    for _ in range(int(K)):
        num, den = 1.0 / a + x, a + 1.0 / a + x
        Dk = (num ** 2 + a ** 2 * Dk) / den ** 2
        x = a * num / den
    return 2.0 * a * Dk


def gamma_K_numeric(lam: float, sigma: float, K: int, h: float = 1e-5) -> float:
    """Gamma_K = -F_K'(lambda) by central differences on F_K(q) = 2 q ell_K."""
    F = lambda v: 2.0 * v * float(ell_K(lam, v, sigma, K))
    step = h * lam
    return -(F(lam + step) - F(lam - step)) / (2.0 * step)


def optimal_alpha(K: int, grid: Optional[np.ndarray] = None) -> tuple[float, float]:
    """(alpha*, Gamma_K(alpha*)) maximizing the local rate at fixed K."""
    if grid is None:
        grid = np.logspace(-2, 4, 4001)
    vals = gamma_K(grid, K)
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 2001)
    vfine = gamma_K(fine, K)
    j = int(np.argmax(vfine))
    return float(fine[j]), float(vfine[j])


def rate_table(Ks=(0, 1, 2, 5, 10, 20, 50), cost_overhead: float = 0.0):
    import pandas as pd

    rows = []
    for K in Ks:
        a_star, g_star = optimal_alpha(K)
        c_K = cost_overhead + 2 * K + 1
        rows.append({"K": K, "sigma_opt^2/lambda": 1.0 / a_star,
                     "Gamma_K*": g_star, "cost c_K": c_K,
                     "Gamma_K*/c_K": g_star / c_K})
    return pd.DataFrame(rows).set_index("K")


# --- Sinkhorn divergence between centered 1-D Gaussians --------------------- #
def sinkhorn_divergence_1d(v: ArrayLike, lam: ArrayLike, sigma: float) -> np.ndarray:
    """S_eps(N(0,v), N(0,lambda)) in closed form"""
    v, lam = np.asarray(v, float), np.asarray(lam, float)
    s2 = sigma ** 2
    A = np.sqrt(s2 ** 2 + 4.0 * v ** 2)
    B = np.sqrt(s2 ** 2 + 4.0 * lam * v)
    C = np.sqrt(s2 ** 2 + 4.0 * lam ** 2)
    return 0.25 * (A - 2.0 * B + C
                   + s2 * np.log((B + s2) ** 2 / ((A + s2) * (C + s2))))


def dissipation_fraction(lam: np.ndarray, q: np.ndarray, sigma: float,
                         K: int) -> float:
    """alpha_K(p,q) = <Vhat, V^{(K)}> / ||Vhat||^2"""
    lam, q = np.atleast_1d(lam), np.atleast_1d(q)
    active = np.abs(lam - q) > 1e-14
    if not active.any():
        return 1.0
    li = ell_K(lam[active], q[active], sigma, -1)
    lk = ell_K(lam[active], q[active], sigma, K)
    w = q[active] * li ** 2
    return float((w * (lk / li)).sum() / w.sum())


def alpha_0_lower_bound(lam_max: float, sigma: float) -> float:
    """Theoretical Uniform lower bound on alpha_0 (population K=0)"""
    s2 = sigma ** 2
    if lam_max <= s2:
        return (4.0 * s2 + lam_max) / (5.0 * (s2 + lam_max))
    return s2 / (s2 + lam_max)
