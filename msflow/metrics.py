from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .measures import Emp, W2, OT_plan, barycentric_map
from .kernels import Kernel, get_kernel, kde, mean_shift_numerator


# =========================================================================== #
# Convergence certificates
# =========================================================================== #
def drift_norm(q: Emp, V: np.ndarray) -> float:
    """||V||_{L2(q)} = ( sum_i w_i ||V(x_i)||^2 )^{1/2}."""
    return float(np.sqrt((q.w * (V ** 2).sum(-1)).sum()))


def dissipativity(p: Emp, q: Emp, V: np.ndarray,
                  plan: Optional[np.ndarray] = None) -> float:
    """Pairing for one selected optimal plan ``gamma*``."""
    T = barycentric_map(q, p, plan)
    return float((q.w * (V * (q.X - T)).sum(-1)).sum())


def step_diagnostics(p: Emp, q: Emp, V: np.ndarray, tau: float,
                     floor: float = 1e-9) -> Dict[str, float]:
    """All per-step certificates for the step q -> (Id + tau V)_# q."""
    plan = OT_plan(q, p)
    w2 = float(np.sqrt(max((plan * cdist(q.X, p.X, "sqeuclidean")).sum(), 0.0)))
    w2_next = W2(Emp(q.X + tau * V, q.w), p)
    D = dissipativity(p, q, V, plan)
    nV = drift_norm(q, V)

    alpha_hat = -D / w2 ** 2 if w2 > floor else np.nan
    L_hat = nV / w2 if w2 > floor else np.nan
    rho_bound = (np.sqrt(max(1.0 - 2.0 * alpha_hat * tau + (L_hat * tau) ** 2, 0.0))
                 if w2 > floor else np.nan)
    return dict(W2=w2, W2_next=w2_next,
                rho_NE=w2_next / w2 if w2 > floor else np.nan,
                rho_diss=D / w2 ** 2 if w2 > floor else np.nan,
                beta_star=-D / nV ** 2 if nV > floor else np.nan,
                ot_alignment=-D / (w2 * nV) if w2 > floor and nV > floor else np.nan,
                alpha_hat=alpha_hat, L_hat=L_hat, rho_bound=rho_bound,
                tau_opt=alpha_hat / L_hat ** 2 if L_hat > floor else np.nan,
                pairing=D, Vnorm=nV)


def gaussian_growth_constant(tau_q: float, tau_star: float, sigma: float) -> float:
    """Closed-form L(q) for centred isotropic Gaussians and a Gaussian kernel."""
    return (sigma ** 2 * tau_q * (tau_q + tau_star)
            / ((sigma ** 2 + tau_q ** 2) * (sigma ** 2 + tau_star ** 2)))


def beta_vs_sigma(p: Emp, q: Emp, sigmas: Sequence[float],
                  build: Callable[[float], Callable]) -> np.ndarray:
    """Realized star-cocoercivity beta at a *frozen* pair (p, q), sweeping sigma."""
    plan = OT_plan(q, p)
    disp = q.X - barycentric_map(q, p, plan)
    out = []
    for s in sigmas:
        V = build(float(s))(p, q)
        pairing = float((q.w * (V * disp).sum(-1)).sum())
        out.append(-pairing / (drift_norm(q, V) ** 2 + 1e-14))
    return np.array(out)


def two_dirac_dissipativity(R: float, sigma: float,
                            kernel_name: str = "Gaussian") -> dict:
    """1-D counterexample to global dissipativity"""
    if kernel_name != "Gaussian":
        raise ValueError("the stated threshold and bound are derived for the Gaussian kernel")
    from .drifts import mean_shift

    p = Emp(np.array([[-R], [R]]))
    q = Emp(np.array([[R / 2.0], [3.0 * R / 4.0]]))
    V = mean_shift(get_kernel(kernel_name, sigma))(p, q)
    # the monotone coupling is the optimal one in 1-D
    D = 0.5 * V[0, 0] * (q.X[0, 0] + R) + 0.5 * V[1, 0] * (q.X[1, 0] - R)
    return dict(R=R, sigma=sigma, D=float(D), V_half=float(V[0, 0]),
                V_three_quarter=float(V[1, 0]), bound=R ** 2 / 8.0,
                threshold=sigma * np.sqrt(np.log(15.0)))


# =========================================================================== #
# Energy certificates
# =========================================================================== #
def MMD2(p: Emp, q: Emp, kernel: Kernel) -> float:
    """MMD_k^2(q, p) = E_qq[k] - 2 E_qp[k] + E_pp[k]."""
    return float(q.w @ kernel.K(q.X, q.X) @ q.w
                 - 2.0 * q.w @ kernel.K(q.X, p.X) @ p.w
                 + p.w @ kernel.K(p.X, p.X) @ p.w)


def sharp_energy(q: Emp, p: Emp, kernel: Kernel,
                 Epp: Optional[float] = None) -> tuple[float, float]:
    """F(q) = 1/2 MMD^2_{k#}(q, p); returns (F, E_pp) caching the fixed term."""
    Ks = kernel.sharp_K
    if Ks is None:
        raise ValueError(f"{kernel.name} has no sharp companion, F is undefined")
    if Epp is None:
        Epp = float(p.w @ Ks(p.X, p.X) @ p.w)
    Eqq = float(q.w @ Ks(q.X, q.X) @ q.w)
    Eqp = float(q.w @ Ks(q.X, p.X) @ p.w)
    return 0.5 * (Eqq - 2.0 * Eqp + Epp), Epp


def mean_shift_branches(p: Emp, q: Emp, kernel: Kernel):
    """(V+, V-, p_k, q_k) at q.X: the two mean-shift branches and the densities."""
    pk, qk = kde(q.X, p, kernel), kde(q.X, q, kernel)
    Vp = mean_shift_numerator(q.X, p, kernel) / pk[:, None]
    Vq = mean_shift_numerator(q.X, q, kernel) / qk[:, None]
    return Vp, Vq, pk, qk


def f_dissipativity(p: Emp, q: Emp, kernel: Kernel,
                    V: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Alignment of a field V with the WGF velocity of F = 1/2 MMD^2_{k#}."""
    Vp, Vm, pk, qk = mean_shift_branches(p, q, kernel)
    V_ms = Vp - Vm
    if V is None:
        V = V_ms
    mubar, rho_half = 0.5 * (pk + qk), 0.5 * (pk - qk)
    quad = float((q.w * mubar * (V * V_ms).sum(-1)).sum())
    mech = float((q.w * rho_half * (V * (Vp + Vm)).sum(-1)).sum())
    nV2 = float((q.w * (V ** 2).sum(-1)).sum()) + 1e-14
    return dict(neg_diss=quad + mech, quad=quad, mechant=mech,
                beta_F=(quad + mech) / nV2, normV2=nV2)


# =========================================================================== #
# Conservativity: how much of a field is a gradient
# =========================================================================== #

def _grad_gaussian_basis(X, anchors, h):
    diff = anchors[None, :, :] - X[:, None, :]
    chi = np.exp(-(diff ** 2).sum(-1) / (2 * h ** 2))
    return chi[:, :, None] * diff / h ** 2


def gauge_fraction(q: Emp, R: np.ndarray, anchors=None, h=None,
                   ridge: float = 1e-4) -> Dict[str, float]:
    """Finite-basis gradient-projection diagnostic for a field ``R``."""
    X, w = q.X, q.w
    anchors = X if anchors is None else anchors
    h = 0.5 * float(np.median(pdist(X))) if h is None else h
    Gb = _grad_gaussian_basis(X, anchors, h)
    A = np.einsum("iad,ibd->ab", w[:, None, None] * Gb, Gb)
    b = np.einsum("id,iad->a", w[:, None] * R, Gb)
    c = np.linalg.solve(A + ridge * (np.trace(A) / len(A)) * np.eye(len(A)), b)
    grad_energy = float(c @ b)
    total = float((w * (R ** 2).sum(-1)).sum())
    return dict(hm1_norm=float(np.sqrt(max(grad_energy, 0.0))),
                gauge_frac=1.0 - grad_energy / (total + 1e-12),
                norm=float(np.sqrt(total)))


def curl_2d(Vx, Vy, g):
    h = g[1] - g[0]
    return np.gradient(Vy, h, axis=1) - np.gradient(Vx, h, axis=0)


def div_2d(Vx, Vy, g):
    h = g[1] - g[0]
    return np.gradient(Vx, h, axis=1) + np.gradient(Vy, h, axis=0)


def rotational_fraction(Vx, Vy, g) -> float:
    """Finite-grid curl/divergence ratio; zero is compatible with conservativity."""
    re = float((curl_2d(Vx, Vy, g) ** 2).sum())
    de = float((div_2d(Vx, Vy, g) ** 2).sum())
    return re / (re + de + 1e-12)


# =========================================================================== #
# Target quality
# =========================================================================== #
def mode_metrics(q: Emp, centers, r_cover: float = 1.4,
                 min_mass_frac: float = 0.3) -> dict:
    """Coverage of a multimodal target."""
    Dm = cdist(q.X, centers)
    nearest, mind = Dm.argmin(axis=1), Dm.min(axis=1)
    K = len(centers)
    mass = np.array([q.w[(nearest == j) & (mind < r_cover)].sum() for j in range(K)])
    pc = mass / (mass.sum() + 1e-12)
    nz = pc[pc > 0]
    return dict(n_covered=int((mass > min_mass_frac / K).sum()),
                coverage_entropy=float(-np.sum(nz * np.log(nz)) / np.log(K)) if K > 1 else 1.0,
                precision=float(q.w[mind < r_cover].sum()),
                mass=mass)


def mass_split_error(q: Emp, centers, target_mass, r_cover: float = 1.4) -> float:
    """L1 error between the captured per-mode mass and the target's."""
    return float(np.abs(mode_metrics(q, centers, r_cover)["mass"] - target_mass).sum())


def smoothed_densities(q: Emp, p: Emp, sigma: float, extent: float,
                       n_grid: int = 64, kernel_name: str = "Gaussian"):
    """(q_sigma, p_sigma) on a regular grid of [-extent, extent]^2, normalized."""
    g = np.linspace(-extent, extent, n_grid)
    xx, yy = np.meshgrid(g, g)
    G = np.column_stack([xx.ravel(), yy.ravel()])
    kern = get_kernel(kernel_name, sigma)
    qd, pd = kde(G, q, kern), kde(G, p, kern)
    return qd / qd.sum(), pd / pd.sum(), (g, xx, yy)


def smoothed_kl(q: Emp, p: Emp, sigma: float, extent: float, n_grid: int = 64,
                kernel_name: str = "Gaussian") -> float:
    """H(q) = KL(q_sigma || p_sigma) by grid quadrature."""
    Q, P, _ = smoothed_densities(q, p, sigma, extent, n_grid, kernel_name)
    return float((Q * np.log(Q / P)).sum())


def fit_exponential_rate(ks, values, lo: float = 0.3, hi: float = 0.9,
                         floor: float = 1e-12) -> float:
    """Least-squares rate r in values ~ exp(-r k), fitted on the middle window."""
    ks, values = np.asarray(ks, float), np.asarray(values, float)
    i0, i1 = int(lo * len(ks)), int(hi * len(ks))
    k, v = ks[i0:i1], values[i0:i1]
    m = v > floor
    if m.sum() < 3:
        return np.nan
    return float(-np.polyfit(k[m], np.log(v[m]), 1)[0])


# =========================================================================== #
# Coifman-Lafon: the operator behind the alpha family
# =========================================================================== #

def alpha_operator(mu: Emp, kernel: Kernel, alpha: float) -> dict:
    """The alpha-reweighted Markov operator on supp(mu) and its invariant law."""
    K = kernel.K(mu.X, mu.X)
    rho = (K * mu.w[None, :]).sum(1) + 1e-12
    ra = rho ** (-alpha)
    A = K * ra[:, None] * ra[None, :] * mu.w[None, :]
    d = A.sum(1) + 1e-12
    dis = 1.0 / np.sqrt(d)
    S = dis[:, None] * A * dis[None, :]
    ev = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))[::-1]
    pi = d / d.sum()
    return dict(P=A / d[:, None], pi=pi, rho=rho, eigenvalues=ev,
                spectral_gap=float(1.0 - abs(ev[1])),
                slope=float(np.polyfit(np.log(rho), np.log(pi + 1e-300), 1)[0]),
                predicted_slope=1.0 - 2.0 * alpha,
                stationary_exponent=2.0 * (1.0 - alpha))
