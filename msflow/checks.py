from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from . import gaussian as gs
from .measures import Emp, W2, load_dataset
from .kernels import KERNELS, get_kernel, smoothed_score
from .drifts import (mean_shift, mean_shift_unnormalized, sharp_core, mmd, ot,
                     alpha_mean_shift, pgr_family, decompose, common_normalization,
                     deng_official_batch_details,
                     sinkhorn_proxy_details)
from .metrics import dissipativity, drift_norm, sharp_energy, f_dissipativity, \
    two_dirac_dissipativity
from .sinkhorn import sinkhorn_drift, sinkhorn_divergence, epsilon_of_sigma


def _result(name, err, tol, detail="") -> dict:
    return dict(check=name, passed=bool(err <= tol), error=float(err),
                tolerance=float(tol), detail=detail)


def _pair(seed: int = 0, n: int = 120, d: int = 2):
    """A small, well-separated (p, q) pair to evaluate identities on."""
    rng = np.random.default_rng(seed)
    p = Emp(np.array([1.5, -0.5])[:d] + 0.7 * rng.standard_normal((n, d)))
    q = Emp(np.array([-1.0, 1.0])[:d] + 0.9 * rng.standard_normal((n, d)))
    return p, q


# =========================================================================== #
# 1. Kernels and their sharp companions
# =========================================================================== #
def check_kernel_gradient(kernel_name: str, sigma: float = 1.0,
                          h: float = 1e-6, tol: float = 1e-5) -> dict:
    """grad_x_K agrees with central differences of K."""
    kern = get_kernel(kernel_name, sigma)
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(12, 2)), rng.normal(scale=1.5, size=(9, 2))
    ana = kern.grad_x_K(X, Y)
    num = np.empty_like(ana)
    for a in range(2):
        e = np.zeros(2); e[a] = h
        num[:, :, a] = (kern.K(X + e, Y) - kern.K(X - e, Y)) / (2 * h)
    err = np.abs(ana - num).max() / (np.abs(ana).max() + 1e-12)
    return _result(f"grad k [{kernel_name}]", err, tol, "analytic vs finite differences")


def check_sharp_companion(kernel_name: str, sigma: float = 1.0,
                          h: float = 1e-6, tol: float = 1e-5) -> dict:
    """grad_x k#(x,y) = k(x,y) (y - x): the defining identity of the companion."""
    kern = get_kernel(kernel_name, sigma)
    if not kern.has_sharp:
        return _result(f"sharp companion [{kernel_name}]", 0.0, 1.0,
                       "no positive tail-normalized companion — local primitive not tested")
    rng = np.random.default_rng(2)
    X, Y = rng.normal(size=(12, 2)), rng.normal(scale=1.5, size=(9, 2))
    target = kern.K(X, Y)[:, :, None] * (Y[None] - X[:, None])
    num = np.empty_like(target)
    for a in range(2):
        e = np.zeros(2); e[a] = h
        num[:, :, a] = (kern.sharp_K(X + e, Y) - kern.sharp_K(X - e, Y)) / (2 * h)
    err = np.abs(target - num).max() / (np.abs(target).max() + 1e-12)
    return _result(f"sharp companion [{kernel_name}]", err, tol,
                   r"grad_x k# = k (y-x)")


# =========================================================================== #
# 2. Structure of the mean-shift field
# =========================================================================== #
def check_zero_drift(tol: float = 1e-10) -> dict:
    """V_{p,p} = 0 for every kernelized field, with the self term kept.

    This is the equilibrium condition of the drifting model: once the model
    matches the data, nothing moves.
    """
    p, _ = _pair()
    kern = get_kernel("Laplacian", 1.0)
    fields = {"mean_shift": mean_shift(kern), "sharp_core": sharp_core(kern),
              "ms_unnorm": mean_shift_unnormalized(kern), "mmd": mmd(kern),
              "alpha_0.5": alpha_mean_shift(kern, 0.5), "ot": ot(),
              "common_geo": common_normalization(kern, "geo")}
    errs = {name: float(np.abs(f(p, p)).max()) for name, f in fields.items()}
    worst = max(errs, key=errs.get)
    return _result("zero-drift V_{p,p}=0", errs[worst], tol,
                   f"worst field: {worst}")


def check_self_term_convention(tol: float = 1e-3) -> dict:
    """Dropping the self pair from V^- breaks the zero-drift condition.

    Reported as the *size* of the violation relative to a typical drift
    magnitude: this is what a leave-one-out implementation silently pays.
    """
    p, q = _pair()
    kern = get_kernel("Laplacian", 1.0)
    keep = float(np.abs(mean_shift(kern)(p, p)).max())
    drop = float(np.abs(mean_shift(kern, exclude_self=True)(p, p)).max())
    scale = drift_norm(q, mean_shift(kern)(p, q))
    return _result("self term: leave-one-out breaks V_{p,p}=0",
                   0.0 if drop > 10 * max(keep, 1e-12) else 1.0, tol,
                   f"|V_pp| = {keep:.2e} (kept) vs {drop:.3f} (dropped), "
                   f"typical |V| = {scale:.3f}")


def check_pgr_decomposition(kernel_name: str = "Laplacian", sigma: float = 1.0,
                            tol: float = 1e-8) -> dict:
    """V^MS = a_q G + R exactly, for any radial kernel with a companion."""
    p, q = _pair()
    kern = get_kernel(kernel_name, sigma)
    dec = decompose(q.X, p, q, kern)
    err = np.abs(dec["V"] - mean_shift(kern)(p, q)).max() / (np.abs(dec["V"]).max() + 1e-12)
    return _result(f"P.G+R = population row [{kernel_name}]", err, tol,
                   "decomposition reconstructs the field")


def check_gaussian_residual_vanishes(sigma: float = 1.0, tol: float = 1e-10) -> dict:
    """For the Gaussian kernel k# = sigma^2 k, hence a = sigma^2 and R = 0.

    This is the notes' statement that the Gaussian is the only radial kernel
    with a conservative mean-shift field.
    """
    p, q = _pair()
    kern = get_kernel("Gaussian", sigma)
    dec = decompose(q.X, p, q, kern)
    err = max(float(np.abs(dec["R"]).max()),
              float(np.abs(dec["a_q"] - sigma ** 2).max()))
    return _result("Gaussian: R = 0 and a = sigma^2", err, tol,
                   "conservativity of the Gaussian mean shift")


def check_gaussian_score_identity(sigma: float = 1.0, tol: float = 1e-8) -> dict:
    """V^MS_Gaussian = sigma^2 ( grad log p_sigma - grad log q_sigma ).

    The smoothed-score reading of Gaussian drifting, from which identifiability
    follows.
    """
    p, q = _pair()
    kern = get_kernel("Gaussian", sigma)
    lhs = mean_shift(kern)(p, q)
    rhs = sigma ** 2 * (smoothed_score(q.X, p, kern) - smoothed_score(q.X, q, kern))
    err = np.abs(lhs - rhs).max() / (np.abs(lhs).max() + 1e-12)
    return _result("Gaussian MS = sigma^2 (s_p - s_q)", err, tol,
                   "smoothed score discrepancy")


def check_alpha_invariant_measure(tol: float = 0.05) -> dict:
    """The alpha-walk fixes pi ~ rho^{1-2 alpha} (measure p^{2(1-alpha)}).

    Fitted on a Gaussian-mixture target; the identity is asymptotic, so the
    tolerance is the fit's own slack rather than machine precision.
    """
    from .metrics import alpha_operator

    data = load_dataset("mass_hierarchy", n=600, seed=3)
    kern = get_kernel("Gaussian", 1.0)
    errs = [abs(alpha_operator(data.p, kern, a)["slope"] - (1.0 - 2.0 * a))
            for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return _result("alpha walk fixes rho^{1-2a}", max(errs), tol,
                   "fitted vs predicted exponent of pi against rho")


def check_alpha_zero_is_population(tol: float = 1e-9) -> dict:
    """At alpha zero the Coifman-Lafon family is the population row field."""
    p, q = _pair()
    kern = get_kernel("Laplacian", 1.0)
    err = np.abs(alpha_mean_shift(kern, 0.0)(p, q) - mean_shift(kern)(p, q)).max()
    return _result("alpha = 0 is population row field", err, tol, "")


def check_pgr_endpoints(tol: float = 1e-9) -> dict:
    """(gamma, eta)=(1,1) is the population row field; (0,0) its core."""
    p, q = _pair()
    kern = get_kernel("Laplacian", 1.0)
    e1 = np.abs(pgr_family(kern, 1.0, 1.0)(p, q) - mean_shift(kern)(p, q)).max()
    e2 = np.abs(pgr_family(kern, 0.0, 0.0)(p, q) - sharp_core(kern)(p, q)).max()
    return _result("P.G+R family endpoints", max(e1, e2), tol,
                   "(1,1)=population row field, (0,0)=sharp core")


def check_deng_official_literal(tol: float = 1e-13) -> dict:
    """Appendix A.1 Algorithm 2, independently transcribed line by line."""
    q = np.array([[-1.1, 0.2], [0.4, 1.3], [1.7, -0.8]])
    p = np.array([[-0.7, 1.8], [0.2, -1.1], [1.1, 0.6], [2.0, 1.5]])
    temp = 0.37
    got = deng_official_batch_details(q, p, temperature=temp)

    dpos = cdist(q, p, "euclidean")
    dneg = cdist(q, q, "euclidean") + np.eye(len(q)) * 1e6
    logits = np.concatenate((-dpos / temp, -dneg / temp), axis=1)
    row = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    col = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
    affinity = np.sqrt(row * col)
    Apos, Aneg = affinity[:, :len(p)], affinity[:, len(p):]
    Wpos = Apos * Aneg.sum(1, keepdims=True)
    Wneg = Aneg * Apos.sum(1, keepdims=True)
    ref = Wpos @ p - Wneg @ q
    errors = [
        np.max(np.abs(got.row_attention - row)),
        np.max(np.abs(got.column_attention - col)),
        np.max(np.abs(got.affinity - affinity)),
        np.max(np.abs(got.positive_weights - Wpos)),
        np.max(np.abs(got.negative_weights - Wneg)),
        np.max(np.abs(got.field - ref)),
    ]
    return _result("Deng paper Algorithm 2: literal pseudocode",
                   float(max(errors)), tol,
                   "pos-first joint softmax, mask=1e6, no floor/RMS")




def check_sinkhorn_proxy_separate_affinities(tol: float = 1e-13) -> dict:
    """Proxy normalizes positive and negative matrices separately."""
    p, q = _pair(seed=19, n=18)
    kern = get_kernel("Gaussian", 0.9)
    got = sinkhorn_proxy_details(p, q, kern, cross_weight=True).field

    def affinity(X, Y):
        logits = kern.logK(X, Y)
        row = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        col = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
        return np.sqrt(row * col)

    Apos, Aneg = affinity(q.X, p.X), affinity(q.X, q.X)
    spos, sneg = Apos.sum(1, keepdims=True), Aneg.sum(1, keepdims=True)
    ref = sneg * (Apos @ p.X) - spos * (Aneg @ q.X)
    err = float(np.abs(got - ref).max())
    return _result("Sinkhorn Proxy: separate +/- affinities", err, tol,
                   "Gaussian sqrt(row softmax * column softmax) + cross-weighting")


def check_common_none_is_exact_endpoint(tol: float = 1e-14) -> dict:
    """A floor for density normalizers must not rescale the `none` endpoint."""
    p, q = _pair(seed=20, n=24)
    kern = get_kernel("Laplacian", 1.0)
    reference = mean_shift_unnormalized(kern)(p, q)
    got = common_normalization(kern, "none", floor=0.37)(p, q)
    return _result("common normalization: none endpoint", float(np.abs(got - reference).max()),
                   tol, "density floor ignored when A=1")


# =========================================================================== #
# 3. Variational identities
# =========================================================================== #
def check_dissipativity_is_W2_derivative(h: float = 1e-4, tol: float = 5e-3) -> dict:
    """D_p^V(q) = 1/2 d/dt W_2^2((Id+tV)_# q, p) at t = 0+.

    The proposition that makes the dissipativity pairing meaningful.  Verified
    by a one-sided finite difference of the exact discrete W_2.
    """
    p, q = _pair(seed=3, n=60)
    V = mean_shift(get_kernel("Gaussian", 1.0))(p, q)
    D = dissipativity(p, q, V)
    w2_0 = W2(q, p) ** 2
    w2_h = W2(Emp(q.X + h * V, q.w), p) ** 2
    err = abs(0.5 * (w2_h - w2_0) / h - D) / (abs(D) + 1e-12)
    return _result("D = 1/2 d/dt W_2^2", err, tol,
                   f"D = {D:.4f}, finite difference = {0.5 * (w2_h - w2_0) / h:.4f}")


def check_unnormalized_is_wgf_of_F(h: float = 1e-5, tol: float = 1e-3) -> dict:
    """The unnormalized field is the WGF velocity of F = 1/2 MMD^2_{k#}.

    Equivalent to d/dt F((Id+tV)_# q)|_0 = -||V||^2_{L2(q)} when V = N_p - N_q,
    checked by finite differences.  This is why the unnormalized field is
    1-cocoercive for F, and the baseline against which the normalization is
    judged.
    """
    p, q = _pair(seed=4, n=80)
    kern = get_kernel("Laplacian", 1.0)
    V = mean_shift_unnormalized(kern)(p, q)
    F0, Epp = sharp_energy(q, p, kern)
    Fh, _ = sharp_energy(Emp(q.X + h * V, q.w), p, kern, Epp)
    lhs, rhs = (Fh - F0) / h, -drift_norm(q, V) ** 2
    err = abs(lhs - rhs) / (abs(rhs) + 1e-12)
    return _result("unnormalized MS = -grad_W F", err, tol,
                   f"dF/dt = {lhs:.5f}, -||V||^2 = {rhs:.5f}")


def check_f_dissipativity_split(h: float = 1e-5, tol: float = 5e-3) -> dict:
    """quad + mechant reproduces -d/dt F along the *normalized* field.

    Validates the lemma decomposing -D^{V,F} into a non-negative term and the
    sign-indefinite term created by the normalization.
    """
    p, q = _pair(seed=5, n=80)
    kern = get_kernel("Laplacian", 1.0)
    V = mean_shift(kern)(p, q)
    F0, Epp = sharp_energy(q, p, kern)
    Fh, _ = sharp_energy(Emp(q.X + h * V, q.w), p, kern, Epp)
    dec = f_dissipativity(p, q, kern)
    err = abs(-(Fh - F0) / h - dec["neg_diss"]) / (abs(dec["neg_diss"]) + 1e-12)
    return _result("quad + mechant = -dF/dt", err, tol,
                   f"quad = {dec['quad']:.4f}, mechant = {dec['mechant']:.4f}")


def check_f_cocoercivity_of_unnormalized(tol: float = 1e-9) -> dict:
    """The unnormalized field is exactly 1-cocoercive for F: beta_F = 1.

    Sanity check that `f_dissipativity` really evaluates the *given* field
    against v_F, and not the mean-shift field regardless of the argument.
    """
    p, q = _pair(seed=15, n=70)
    kern = get_kernel("Laplacian", 1.0)
    V = mean_shift_unnormalized(kern)(p, q)
    beta = f_dissipativity(p, q, kern, V)["beta_F"]
    return _result("unnormalized field: beta_F = 1", abs(beta - 1.0), tol, "")


def check_H_constants_identity(tol: float = 1e-10) -> dict:
    """beta_star = alpha_hat / L_hat^2 pointwise, and rho_NE <= the (H) bound.

    The first is the *exact* form of the implication (H) => (star): the constant
    beta = alpha/L^2 is not an extra assumption, it is an identity on realized
    values.  The second is the coupling inequality of the convergence proof,
    W_2^2(q^{k+1},p) <= (1 - 2 alpha tau + L^2 tau^2) W_2^2(q^k,p), checked step
    by step rather than assumed.
    """
    from .metrics import step_diagnostics

    p, q = _pair(seed=16, n=90)
    worst_id, worst_bound = 0.0, 0.0
    for tau in (0.01, 0.05, 0.2):
        for fld in (mean_shift(get_kernel("Laplacian", 1.0)), ot()):
            d = step_diagnostics(p, q, fld(p, q), tau)
            worst_id = max(worst_id, abs(d["beta_star"] - d["alpha_hat"] / d["L_hat"] ** 2))
            worst_bound = max(worst_bound, d["rho_NE"] - d["rho_bound"])
    return _result("(H): beta = alpha/L^2, rho_NE <= bound",
                   max(worst_id, max(worst_bound, 0.0)), tol,
                   "identity on realized constants + the coupling bound")


def check_gaussian_growth_constant(tol: float = 0.05, n: int = 2000) -> dict:
    """The closed-form L(q) of (H2) for centred isotropic Gaussians.

        L(q) = s^2 t_q (t_q + t_*) / ( (s^2 + t_q^2)(s^2 + t_*^2) )

    which the notes use to argue that the anchored-Lipschitz constant is
    **dimension-free** — the sqrt(d) in ||V||_{L2(q)} and in W_2 cancel — and
    stays below 1.  Checked in d = 2 and d = 3, so a stray sqrt(d) would show.

    Two finite-sample corrections, both of which just make the comparison the
    one the formula is about: the clouds are re-centred (the formula is for
    *centred* Gaussians, and an empirical mean of size t/sqrt(n) adds a constant
    to V), and W_2 and the formula both use the *sample* standard deviations
    rather than the nominal ones.  Bandwidths are kept comparable to the cloud
    radius: at t_q >> sigma in higher d the mean-shift KDE has too few
    neighbours and needs far more samples, which is a property of the estimator,
    not of the identity.
    """
    from .metrics import gaussian_growth_constant, drift_norm

    rng = np.random.default_rng(17)
    sigma, errs = 1.0, []
    for d in (2, 3):
        for t_q, t_star in ((0.6, 1.5), (1.2, 0.5)):
            Xq = t_q * rng.standard_normal((n, d))
            Xp = t_star * rng.standard_normal((n, d))
            q, p = Emp(Xq - Xq.mean(0)), Emp(Xp - Xp.mean(0))
            tq = np.sqrt(np.trace(q.cov()) / d)
            ts = np.sqrt(np.trace(p.cov()) / d)
            V = mean_shift(get_kernel("Gaussian", sigma))(p, q)
            measured = drift_norm(q, V) / (np.sqrt(d) * abs(tq - ts))
            predicted = gaussian_growth_constant(tq, ts, sigma)
            errs.append(abs(measured - predicted) / predicted)
    return _result("Gaussian L(q) closed form", max(errs), tol,
                   f"dimension-free, checked in d = 2 and 3 (n = {n:,})")


def check_ot_drift_constants(tol: float = 1e-8) -> dict:
    """The OT drift saturates (H) on this equal-size permutation-plan fixture.

    Here alpha = L = 1 exactly.  This is not asserted for a general weighted
    discrete plan, whose barycentric projection can discard conditional
    variance.
    """
    p, q = _pair(seed=6, n=60)
    V = ot()(p, q)
    w2sq = W2(q, p) ** 2
    err = max(abs(dissipativity(p, q, V) + w2sq), abs(drift_norm(q, V) ** 2 - w2sq)) / w2sq
    return _result("OT permutation fixture: D = -W_2^2, L = 1", err, tol, "")


# =========================================================================== #
# 4. The Gaussian sector
# =========================================================================== #
def check_gaussian_affine_drift(sigma: float = 1.2, tol: float = 1e-10) -> dict:
    """On Gaussians the mean-shift drift is exactly x -> A_Sigma x + b_Sigma."""
    m_q, m_p = np.array([0.3, -0.4]), np.array([1.0, 0.5])
    S_q = np.array([[1.2, 0.3], [0.3, 0.7]])
    S_p = np.array([[0.5, -0.1], [-0.1, 2.0]])
    x = np.random.default_rng(7).normal(size=(20, 2))
    A, b = gs.affine_drift(m_q, S_q, m_p, S_p, sigma)
    err = np.abs(gs.exact_mean_shift(x, m_q, S_q, m_p, S_p, sigma) - (x @ A.T + b)).max()
    return _result("Gaussian drift is affine", err, tol, "A x + b vs exact barycentres")


def check_particles_match_moment_flow(sigma: float = 1.2, tau: float = 0.05,
                                      n: int = 4000, n_iter: int = 60,
                                      tol: float = 0.08) -> dict:
    """Particle simulation reproduces the closed-form moment flow.

    Monte-Carlo, so the tolerance is a few percent on the terminal W_2; what
    matters is that the two agree far better than either agrees with a wrong
    scheme.
    """
    rng = np.random.default_rng(8)
    m0, S0 = np.array([-2.0, 1.0]), np.diag([1.5, 0.4])
    mS, SS = np.array([1.0, 0.0]), np.array([[0.8, 0.3], [0.3, 0.5]])
    q = Emp(m0 + rng.standard_normal((n, 2)) @ np.linalg.cholesky(S0).T)
    p = Emp(mS + rng.standard_normal((n, 2)) @ np.linalg.cholesky(SS).T)
    field = mean_shift(get_kernel("Gaussian", sigma))
    for _ in range(n_iter):
        q = Emp(q.X + tau * field(p, q), q.w)
    ref = gs.moment_flow(m0, S0, mS, SS, sigma, tau, n_iter, scheme="map")
    got = gs.W2_gaussian(q.mean(), q.cov(), mS, SS)
    err = abs(got - ref["W2"][-1]) / (ref["W2"][-1] + 1e-12)
    return _result("particles = moment flow", err, tol,
                   f"W_2 particles {got:.4f} vs theory {ref['W2'][-1]:.4f}")


def check_lyapunov_decreases(sigma: float = 1.0, tol: float = 1e-3) -> dict:
    """Phi(Sigma_t) is non-increasing, and its rate matches the LaSalle formula.

    Two claims at once: Phi never goes up along the whole trajectory (exact,
    zero tolerance), and its initial slope equals the closed-form dissipation
    -(2/s^2)||Sigma^{1/2} A||_F^2.  The slope is read off a tiny Euler step, so
    the residual is the O(tau) discretization error and shrinks with tau.
    """
    S0 = np.diag([2.5, 0.2])
    SS = np.array([[0.9, 0.2], [0.2, 0.6]])
    m = np.zeros(2)
    flow = gs.moment_flow(m, S0, m, SS, sigma, tau=0.01, n_iter=400, scheme="ode")
    if np.diff(flow["Phi"]).max() > 0:
        return _result("Lyapunov Phi decreases", np.inf, tol, "Phi increased along the flow")

    h = 1e-6
    short = gs.moment_flow(m, S0, m, SS, sigma, tau=h, n_iter=1, scheme="ode")
    rate_fd = (short["Phi"][1] - short["Phi"][0]) / h
    rate_cf = gs.phi_dissipation(S0, SS, sigma)
    err = abs(rate_fd - rate_cf) / (abs(rate_cf) + 1e-12)
    return _result("Lyapunov Phi decreases", err, tol,
                   f"dPhi/dt: {rate_fd:.6f} (flow) vs {rate_cf:.6f} (formula)")


# =========================================================================== #
# 5. K-step Sinkhorn drifting
# =========================================================================== #
def check_sinkhorn_K0_is_population(sigma: float = 1.0, tol: float = 1e-9) -> dict:
    """K=0 is the Gaussian population row field, not paper Algorithm 2.

    Also pins the bandwidth convention: eps = 2 sigma^2.
    """
    p, q = _pair(seed=9, n=70)
    err = np.abs(sinkhorn_drift(sigma, K=0)(p, q)
                 - mean_shift(get_kernel("Gaussian", sigma))(p, q)).max()
    return _result("Sinkhorn K=0 is population row field", err, tol,
                   f"row-only; eps = 2 sigma^2 = {epsilon_of_sigma(sigma):g}")


def check_sinkhorn_gaussian_closed_form(sigma: float = 1.0, K: int = 3,
                                        n: int = 1200, tol: float = 0.06) -> dict:
    """Particle K-step Sinkhorn drift matches the closed-form slope ell_K.

    Centered isotropic 1-D Gaussians, where the notes predict V(x) = ell_K x.
    Monte-Carlo, hence a percent-level tolerance.
    """
    rng = np.random.default_rng(10)
    lam, v = 1.6, 0.5
    p = Emp(np.sqrt(lam) * rng.standard_normal((n, 1)))
    q = Emp(np.sqrt(v) * rng.standard_normal((n, 1)))
    V = sinkhorn_drift(sigma, K=K)(p, q)
    slope = float(np.polyfit(q.X[:, 0], V[:, 0], 1)[0])            # regress V on x
    predicted = float(gs.ell_K(lam, v, sigma, K))
    err = abs(slope - predicted) / (abs(predicted) + 1e-12)
    return _result(f"Sinkhorn K={K} matches ell_K", err, tol,
                   f"slope {slope:.4f} vs ell_K {predicted:.4f}")


def check_gamma_recursion(lam: float = 1.3, sigma: float = 0.9, tol: float = 1e-4) -> dict:
    """The closed-form recursion for Gamma_K agrees with -F_K'(lambda)."""
    errs = []
    for K in (0, 1, 3, 8, -1):
        ana = float(gs.gamma_K(lam / sigma ** 2, K))
        num = gs.gamma_K_numeric(lam, sigma, K)
        errs.append(abs(ana - num) / (abs(num) + 1e-12))
    return _result("Gamma_K recursion", max(errs), tol,
                   "analytic recursion vs finite differences of F_K")


def check_gamma_endpoints(lam: float = 2.0, sigma: float = 1.1, tol: float = 1e-12) -> dict:
    """Gamma_0 = 2 lam s^2/(lam+s^2)^2 and Gamma_inf = 2 lam / sqrt(s^4+4 lam^2)."""
    a = lam / sigma ** 2
    e0 = abs(gs.gamma_K(a, 0) - 2 * lam * sigma ** 2 / (lam + sigma ** 2) ** 2)
    ei = abs(gs.gamma_K(a, -1) - 2 * lam / np.sqrt(sigma ** 4 + 4 * lam ** 2))
    return _result("Gamma_K endpoints", max(e0, ei), tol, "")


def check_gamma_monotone_in_K(tol: float = 0.0) -> dict:
    """0 < Gamma_0 < Gamma_1 < ... < Gamma_inf < 1, for every bandwidth."""
    alphas = np.logspace(-2, 2, 40)
    G = np.array([gs.gamma_K(alphas, K) for K in range(0, 12)])
    bad = float(np.maximum(G[:-1] - G[1:], 0).max())
    ceiling = float(np.maximum(G[-1] - gs.gamma_K(alphas, -1), 0).max())
    return _result("Gamma_K increasing in K", max(bad, ceiling), 1e-12,
                   f"sup Gamma_inf = {gs.gamma_K(alphas, -1).max():.8f} < 1")


def check_sinkhorn_divergence_derivative(tol: float = 1e-6) -> dict:
    """d/dv S_eps(N(0,v), N(0,lam)) = -1/2 ell_inf(lam, v).

    Confirms that the K = inf drift is minus the Wasserstein gradient of the
    Sinkhorn divergence, in the exactly solvable case.
    """
    lam, sigma, h = 1.4, 0.8, 1e-6
    vs = np.array([0.2, 0.7, 1.4, 3.0])
    num = (gs.sinkhorn_divergence_1d(vs + h, lam, sigma)
           - gs.sinkhorn_divergence_1d(vs - h, lam, sigma)) / (2 * h)
    ana = -0.5 * gs.ell_K(lam, vs, sigma, -1)
    err = float(np.abs(num - ana).max() / (np.abs(ana).max() + 1e-12))
    return _result("dS/dv = -1/2 ell_inf", err, tol, "")


def check_discrete_sinkhorn_energy_gradient(tol: float = 2e-7) -> dict:
    """The discrete entropic objective has ``B_cross-B_self`` as -W-gradient."""
    rng = np.random.default_rng(31)
    p = Emp(rng.normal(loc=(0.7, -0.2), scale=0.8, size=(13, 2)))
    q = Emp(rng.normal(loc=(-0.5, 0.4), scale=0.9, size=(11, 2)))
    sigma = 0.9
    V = sinkhorn_drift(sigma, K=-1, tol=1e-12, max_iter=10000)(p, q)
    norm_sq = float(np.mean(np.sum(V * V, axis=1)))
    h = 1e-4
    numeric = (
        sinkhorn_divergence(Emp(q.X + h * V), p, sigma)
        - sinkhorn_divergence(Emp(q.X - h * V), p, sigma)
    ) / (2 * h)
    err = abs(numeric + norm_sq) / (norm_sq + 1e-14)
    return _result("discrete Sinkhorn energy gradient", err, tol,
                   "full KL objective with half-squared transport cost")


def check_alpha_K_ordering(tol: float = 0.0) -> dict:
    """alpha_K increases in K and alpha_inf = 1, and alpha_0 clears its bound."""
    rng = np.random.default_rng(11)
    lam = np.exp(rng.normal(size=6))
    q = np.exp(rng.normal(size=6))
    sigma = 1.0
    a = np.array([gs.dissipation_fraction(lam, q, sigma, K) for K in range(0, 10)])
    bound = gs.alpha_0_lower_bound(float(lam.max()), sigma)
    err = max(float(np.maximum(a[:-1] - a[1:], 0).max()),
              float(max(bound - a[0], 0.0)),
              abs(gs.dissipation_fraction(lam, q, sigma, -1) - 1.0))
    return _result("alpha_K increasing, alpha_inf = 1", err, 1e-9,
                   f"alpha_0 = {a[0]:.3f} >= bound {bound:.3f}")


# =========================================================================== #
# 6. Parameterized maps (natural-gradient experiments)
# =========================================================================== #
def check_parametric_jacobian(h: float = 1e-6, tol: float = 1e-6) -> dict:
    """The analytic parameter Jacobians J_theta(z) match finite differences."""
    from .parametric import AffineMap, MLPMap

    rng = np.random.default_rng(12)
    errs = {}
    for model, theta in ((AffineMap(), np.array([0.3, np.log(0.7), np.log(1.4), -0.5, 0.4])),
                         (MLPMap(6), MLPMap(6).init(np.random.default_rng(1)))):
        z = rng.standard_normal((7, 2))
        J = model.jacobian(theta, z)
        num = np.empty_like(J)
        for j in range(model.dim):
            e = np.zeros(model.dim); e[j] = h
            num[:, :, j] = (model.forward(theta + e, z) - model.forward(theta - e, z)) / (2 * h)
        errs[type(model).__name__] = float(np.abs(J - num).max())
    worst = max(errs, key=errs.get)
    return _result("parametric Jacobians", errs[worst], tol, f"worst: {worst}")


def check_affine_metric_closed_form(n: int = 400_000) -> dict:
    """G(theta) = diag(s1^2+s2^2, s1^2, s2^2, 1, 1) for the affine map.

    The reference is a Monte-Carlo average of J^T J, so the tolerance is set to
    a few standard errors of a fourth-moment estimate, ~ |G| / sqrt(n).
    """
    from .parametric import AffineMap

    model = AffineMap()
    theta = np.array([0.3, np.log(0.7), np.log(1.4), -0.5, 0.4])
    z = np.random.default_rng(13).standard_normal((n, 2))
    J = model.jacobian(theta, z)
    G_exact = model.metric(theta)
    G_mc = np.einsum("nij,nik->jk", J, J) / n
    tol = 8.0 * float(np.abs(G_exact).max()) / np.sqrt(n)
    return _result("affine metric tensor", float(np.abs(G_exact - G_mc).max()), tol,
                   f"closed form vs Monte-Carlo (n={n:,}, tol = 8|G|/sqrt(n))")


def check_moment_kl_is_wasserstein_gradient(tol: float = 1e-10) -> dict:
    """The field returned by `moment_kl` is grad log q - grad log p.

    That is the Wasserstein gradient of KL(q || p) at the Gaussian with the
    empirical moments, which is what makes `lifted_gradient` the chain rule of
    eq. (lifted-gradient) rather than a per-sample derivative.
    """
    from .parametric import moment_kl

    rng = np.random.default_rng(14)
    x = rng.standard_normal((400, 2)) @ np.array([[1.2, 0.3], [0.0, 0.8]]) + [1.0, 2.0]
    mu_p, S_p = np.array([5.0, 3.0]), np.array([[2.0, 0.4], [0.4, 0.6]])
    _, wgrad = moment_kl(x, mu_p, S_p)
    mu, S = x.mean(0), np.cov(x.T, bias=True) + 1e-6 * np.eye(2)
    ref = (x - mu_p) @ np.linalg.inv(S_p).T - (x - mu) @ np.linalg.inv(S).T
    formula_error = float(np.abs(wgrad - ref).max())

    # Independent derivative check: moment_kl returns N*dL/dx, because the
    # lifted gradient averages J^T*wgrad over the empirical measure.
    h, i, a = 1e-6, 7, 1
    xp, xm = x.copy(), x.copy()
    xp[i, a] += h
    xm[i, a] -= h
    numeric = (moment_kl(xp, mu_p, S_p)[0] - moment_kl(xm, mu_p, S_p)[0]) / (2 * h)
    derivative_error = abs(numeric - wgrad[i, a] / len(x))
    return _result("moment KL gradient is grad_W KL",
                   max(formula_error, derivative_error), 2e-8,
                   "closed form + finite difference of the Gaussian-moment surrogate")


# =========================================================================== #
# 7. The counterexample
# =========================================================================== #
def check_two_dirac_counterexample(tol: float = 0.0) -> dict:
    """Past R > sigma sqrt(log 15) the pairing is positive and exceeds R^2/8."""
    d = two_dirac_dissipativity(R=3.0, sigma=1.0)
    err = max(0.0, d["bound"] - d["D"])
    return _result("two-Dirac counterexample", err, 1e-12,
                   f"D = {d['D']:.4f} >= R^2/8 = {d['bound']:.4f} "
                   f"(threshold R/sigma = {d['threshold']:.3f})")


# =========================================================================== #
# Runner
# =========================================================================== #
def run_all(verbose: bool = False) -> list[dict]:
    """Run every check and return the list of result dicts."""
    checks: list[Callable[[], dict]] = []
    for kn in KERNELS:
        checks.append(lambda kn=kn: check_kernel_gradient(kn))
        checks.append(lambda kn=kn: check_sharp_companion(kn))
    checks += [
        check_zero_drift, check_self_term_convention,
        lambda: check_pgr_decomposition("Laplacian"),
        lambda: check_pgr_decomposition("Matern"),
        check_gaussian_residual_vanishes, check_gaussian_score_identity,
        check_alpha_zero_is_population, check_alpha_invariant_measure, check_pgr_endpoints,
        check_deng_official_literal,
        check_sinkhorn_proxy_separate_affinities,
        check_common_none_is_exact_endpoint,
        check_dissipativity_is_W2_derivative, check_unnormalized_is_wgf_of_F,
        check_f_dissipativity_split, check_f_cocoercivity_of_unnormalized,
        check_H_constants_identity, check_gaussian_growth_constant,
        check_ot_drift_constants,
        check_gaussian_affine_drift, check_particles_match_moment_flow,
        check_lyapunov_decreases,
        check_sinkhorn_K0_is_population, check_sinkhorn_gaussian_closed_form,
        check_gamma_recursion, check_gamma_endpoints, check_gamma_monotone_in_K,
        check_sinkhorn_divergence_derivative, check_discrete_sinkhorn_energy_gradient,
        check_alpha_K_ordering,
        check_parametric_jacobian, check_affine_metric_closed_form,
        check_moment_kl_is_wasserstein_gradient,
        check_two_dirac_counterexample,
    ]
    out = []
    for fn in checks:
        try:
            res = fn()
        except Exception as exc:                                   # noqa: BLE001
            res = dict(check=getattr(fn, "__name__", "check"), passed=False,
                       error=np.inf, tolerance=0.0, detail=f"raised {type(exc).__name__}: {exc}")
        out.append(res)
        if verbose:
            print(f"[{'PASS' if res['passed'] else 'FAIL'}] {res['check']:44s} "
                  f"err={res['error']:.2e}  {res['detail']}")
    return out


def main() -> int:
    results = run_all(verbose=True)
    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
