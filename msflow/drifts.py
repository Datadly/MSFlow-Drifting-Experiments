from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from .measures import Emp, barycentric_map
from .kernels import (Kernel, get_kernel, kde, sharp_kde, mean_shift_numerator,
                      softmax_barycentre, smoothed_score)

_EPS = 1e-12


@dataclass
class DriftField:
    """A drift field, evaluable anywhere. """

    name: str
    at: Callable                     # (pts, p, q) -> (n_pts, d)
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.label = self.label or self.name

    def __call__(self, p: Emp, q: Emp) -> np.ndarray:
        return self.at(q.X, p, q)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.meta.items())
        return f"DriftField({self.name}{', ' + args if args else ''})"


# --------------------------------------------------------------------------- #
# Kernelized fields
# --------------------------------------------------------------------------- #
def mean_shift(kernel: Kernel, exclude_self: bool = False) -> DriftField:
    """Population mean-shift drift  V = V^+(x) - V^-(x)."""
    def at(pts, p, q):
        bar_p = softmax_barycentre(pts, p, kernel)
        # the self pair only exists when we evaluate on q's own support
        bar_q = (_barycentre_no_diag(pts, q, kernel) if exclude_self and pts is q.X
                 else softmax_barycentre(pts, q, kernel))
        return bar_p - bar_q

    return DriftField("mean_shift", at, label=r"population $m_p-m_q$",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma,
                                exclude_self=exclude_self))


# Explicit alias used in notebooks and reports whenever the distinction with
# the paper's finite-batch Algorithm 2 matters.
deng_population = mean_shift


def _barycentre_no_diag(pts, mu: Emp, kernel: Kernel) -> np.ndarray:
    """Leave-one-out barycentre (self pair removed). See the module docstring."""
    Kw = kernel.K(pts, mu.X) * mu.w[None, :]
    np.fill_diagonal(Kw, 0.0)
    return (Kw @ mu.X) / (Kw.sum(axis=1, keepdims=True) + _EPS)


def _batch_points(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or min(out.shape) == 0:
        raise ValueError(f"{name} must have non-empty shape (n, d)")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _require_uniform_empirical_weights(*measures: Emp) -> None:
    """Reject weighted clouds where a published mini-batch formula uses samples."""
    for measure in measures:
        uniform = np.full(measure.n, 1.0 / measure.n)
        if not np.allclose(measure.w, uniform, rtol=1e-12, atol=1e-15):
            raise ValueError(
                "this canonical batch estimator assumes uniform samples; "
                "encode mass by sample multiplicity instead"
            )


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    return np.exp(values - logsumexp(values, axis=axis, keepdims=True))


# --------------------------------------------------------------------------- #
# Deng et al., paper Algorithm 2 (the transport-only batch field)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DengOfficialBatchResult:

    row_attention: np.ndarray
    column_attention: np.ndarray
    affinity: np.ndarray
    positive_affinity: np.ndarray
    negative_affinity: np.ndarray
    positive_weights: np.ndarray
    negative_weights: np.ndarray
    positive_drift: np.ndarray
    negative_drift: np.ndarray
    field: np.ndarray


def deng_official_batch_details(x, y_pos, y_neg=None, *,
                                temperature: float = 1.0,
                                self_mask: bool = True,
                                self_mask_distance: float = 1e6
                                ) -> DengOfficialBatchResult:
    """NumPy implementation of Deng et al., Algorithm 2."""
    negative_is_x = y_neg is None or y_neg is x
    x = _batch_points(x, "x")
    y_pos = _batch_points(y_pos, "y_pos")
    if x.shape[1] != y_pos.shape[1]:
        raise ValueError("x and y_pos must have the same dimension")
    if y_neg is None:
        y_neg = x
    else:
        y_neg = _batch_points(y_neg, "y_neg")
        if x.shape[1] != y_neg.shape[1]:
            raise ValueError("x and y_neg must have the same dimension")

    temperature = float(temperature)
    self_mask_distance = float(self_mask_distance)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if not np.isfinite(self_mask_distance) or self_mask_distance <= 0:
        raise ValueError("self_mask_distance must be finite and strictly positive")
    if self_mask and negative_is_x and len(y_neg) != len(x):
        raise ValueError("the self-negative batch must have the same length as x")

    dist_pos = cdist(x, y_pos, "euclidean")
    dist_neg = cdist(x, y_neg, "euclidean")
    if self_mask and negative_is_x:
        diagonal = np.arange(len(x))
        dist_neg[diagonal, diagonal] += self_mask_distance

    logits = np.concatenate((-dist_pos / temperature,
                             -dist_neg / temperature), axis=1)
    row_attention = _softmax(logits, axis=1)
    column_attention = _softmax(logits, axis=0)
    # Keep the literal Algorithm 2 expression: there is no max(..., 1e-6)
    # before the square root.
    affinity = np.sqrt(row_attention * column_attention)
    split = len(y_pos)
    A_pos, A_neg = affinity[:, :split], affinity[:, split:]
    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)
    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    field_value = drift_pos - drift_neg

    return DengOfficialBatchResult(
        row_attention=row_attention,
        column_attention=column_attention,
        affinity=affinity,
        positive_affinity=A_pos,
        negative_affinity=A_neg,
        positive_weights=W_pos,
        negative_weights=W_neg,
        positive_drift=drift_pos,
        negative_drift=drift_neg,
        field=field_value,
    )


def deng_official_batch(x, y_pos, y_neg=None, **kwargs) -> np.ndarray:
    """Return only ``V`` from the paper Algorithm 2."""
    return deng_official_batch_details(x, y_pos, y_neg, **kwargs).field


def deng_official(*, temperature: float = 1.0, self_mask: bool = True,
                  self_mask_distance: float = 1e6) -> DriftField:
    """Paper Algorithm 2 as a batch-dependent transport field."""
    options = dict(temperature=float(temperature), self_mask=bool(self_mask),
                   self_mask_distance=float(self_mask_distance))
    # Validate eagerly so bad config cells fail at construction time.
    if not np.isfinite(options["temperature"]) or options["temperature"] <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if (not np.isfinite(options["self_mask_distance"])
            or options["self_mask_distance"] <= 0):
        raise ValueError("self_mask_distance must be finite and strictly positive")

    def at(pts, p, q):
        _require_uniform_empirical_weights(p, q)
        # Omitting y_neg declares the paper's canonical y_neg=x relation and
        # therefore activates its conditional diagonal mask.
        V = deng_official_batch(q.X, p.X, **options)
        return V if pts is q.X else V[cdist(pts, q.X).argmin(axis=1)]

    return DriftField(
        "deng_official", at,
        label=r"Deng Algorithm 2 $\sqrt{A_{row}A_{col}}$",
        meta=dict(
            implementation="paper_algorithm_2",
            kernel="Laplacian", sigma=options["temperature"],
            temperature=options["temperature"],
            self_mask=options["self_mask"],
            self_mask_distance=options["self_mask_distance"],
            distance="euclidean",
            row_scope="joint_positive_negative",
            column_scope="joint_query_batch",
            affinity="sqrt(A_row*A_col)",
            affinity_product_floor=0.0,
            cross_weight=True,
            feature_normalization=False,
            drift_normalization=False,
            temperatures=(options["temperature"],),
            weights="uniform_samples_required",
            output="raw_algorithm2_field",
            evaluation_scope="batch_support",
            grid_extension="nearest_neighbor_visualization_only",
            supports_grid_diagnostics=False,
            source="Deng et al. 2026, Appendix A.1, Algorithm 2",
        ),
    )



# --------------------------------------------------------------------------- #
# Sinkhorn Proxy (separate positive/negative geometric normalizations)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SinkhornProxyResult:
    positive_affinity: np.ndarray
    negative_affinity: np.ndarray
    positive_mass: np.ndarray
    negative_mass: np.ndarray
    core: np.ndarray
    mobility: np.ndarray
    field: np.ndarray


def _geometric_softmax_affinity(logits: np.ndarray,
                                product_floor: float = 0.0) -> np.ndarray:
    if product_floor < 0 or not np.isfinite(product_floor):
        raise ValueError("product_floor must be finite and non-negative")
    # The proxy has no release-style product floor, so keep the geometric mean
    # in log space: row*column can underflow while its square root is still
    # representable.
    log_row = logits - logsumexp(logits, axis=1, keepdims=True)
    log_col = logits - logsumexp(logits, axis=0, keepdims=True)
    log_affinity = 0.5 * (log_row + log_col)
    if product_floor > 0:
        log_affinity = np.maximum(log_affinity, 0.5 * np.log(product_floor))
    return np.exp(log_affinity)


def sinkhorn_proxy_details(p: Emp, q: Emp, kernel: Kernel, *,
                           exclude_self: bool = False,
                           cross_weight: bool = True,
                           product_floor: float = 0.0) -> SinkhornProxyResult:
    """Compute the finite-batch Sinkhorn Proxy at ``q.X``."""
    _require_uniform_empirical_weights(p, q)
    log_pos = (kernel.logK(q.X, p.X) if kernel.logK is not None
               else np.log(kernel.K(q.X, p.X) + 1e-300))
    log_neg = (kernel.logK(q.X, q.X) if kernel.logK is not None
               else np.log(kernel.K(q.X, q.X) + 1e-300))
    if exclude_self:
        log_neg = log_neg.copy()
        np.fill_diagonal(log_neg, -np.inf)
        if q.n < 2:
            raise ValueError("exclude_self=True needs at least two generated samples")

    A_pos = _geometric_softmax_affinity(log_pos, product_floor)
    A_neg = _geometric_softmax_affinity(log_neg, product_floor)
    if exclude_self:
        np.fill_diagonal(A_neg, 0.0)
    mass_pos = A_pos.sum(axis=1, keepdims=True)
    mass_neg = A_neg.sum(axis=1, keepdims=True)
    B_pos = (A_pos @ p.X) / (mass_pos + 1e-300)
    B_neg = (A_neg @ q.X) / (mass_neg + 1e-300)
    core = B_pos - B_neg
    mobility = mass_pos * mass_neg
    field = mobility * core if cross_weight else core
    return SinkhornProxyResult(A_pos, A_neg, mass_pos[:, 0], mass_neg[:, 0],
                               core, mobility[:, 0], field)


def sinkhorn_proxy(kernel: Kernel, *, exclude_self: bool = False,
                   cross_weight: bool = True,
                   product_floor: float = 0.0) -> DriftField:
    """Sinkhorn Proxy field with separate geometric normalizations (Gretton et al.)."""
    def at(pts, p, q):
        V = sinkhorn_proxy_details(p, q, kernel, exclude_self=exclude_self,
                                   cross_weight=cross_weight,
                                   product_floor=product_floor).field
        return V if pts is q.X else V[cdist(pts, q.X).argmin(axis=1)]

    suffix = "full" if cross_weight else "core"
    qualifier = "" if kernel.name == "Gaussian" else " (generalized)"
    return DriftField(
        f"sinkhorn_proxy_{suffix}", at,
        label=f"Sinkhorn Proxy {suffix}{qualifier}",
        meta=dict(kernel=kernel.name, sigma=kernel.sigma,
                  tau=2.0 * kernel.sigma ** 2 if kernel.name == "Gaussian" else None,
                  row_scope="separate",
                  distance=("squared_euclidean" if kernel.name == "Gaussian"
                            else "kernel_defined"),
                  exclude_self=exclude_self, cross_weight=cross_weight,
                  product_floor=product_floor,
                  weights="uniform_samples_required",
                  source_exact=(kernel.name == "Gaussian" and not exclude_self
                                and cross_weight and product_floor == 0.0),
                  batch_size_factor_omitted=True,
                  evaluation_scope="batch_support",
                  grid_extension="nearest_neighbor_visualization_only",
                  supports_grid_diagnostics=False),
    )


def mean_shift_unnormalized(kernel: Kernel) -> DriftField:
    """N_p - N_q, i.e. the mean-shift numerator without any normalization.

    It is a spatial gradient for every radial profile.  When the kernel has a
    registered positive tail-normalized companion, it also equals the
    Wasserstein-gradient-flow velocity of
    F(q) = 1/2 MMD^2_{k#}(q,p) and is 1-cocoercive for that energy.  IMQ has a
    local primitive but not that positive sharp-MMD interpretation.
    """
    def at(pts, p, q):
        return mean_shift_numerator(pts, p, kernel) - mean_shift_numerator(pts, q, kernel)

    return DriftField("ms_unnorm", at, label=r"unnormalized $N_p-N_q$",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma))


def mmd(kernel: Kernel) -> DriftField:
    """MMD gradient-flow velocity  E_p[grad_x k] - E_q[grad_x k]  for kernel k."""
    def at(pts, p, q):
        return (np.einsum("nmd,m->nd", kernel.grad_x_K(pts, p.X), p.w)
                - np.einsum("nmd,m->nd", kernel.grad_x_K(pts, q.X), q.w))

    return DriftField("mmd", at, label=r"MMD-WGF $\nabla_W \frac{1}{2}\mathrm{MMD}^2_k$",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma))


def sharp_core(kernel: Kernel) -> DriftField:
    """The gradient part of the P.G + R decomposition."""
    def at(pts, p, q):
        return (mean_shift_numerator(pts, p, kernel) / sharp_kde(pts, p, kernel)[:, None]
                - mean_shift_numerator(pts, q, kernel) / sharp_kde(pts, q, kernel)[:, None])

    return DriftField("sharp_core", at, label=r"sharp core $G$",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma))


def decompose(pts: np.ndarray, p: Emp, q: Emp, kernel: Kernel) -> dict:
    """The structural decomposition  V^MS = P G + R. """
    s_p = mean_shift_numerator(pts, p, kernel) / sharp_kde(pts, p, kernel)[:, None]
    s_q = mean_shift_numerator(pts, q, kernel) / sharp_kde(pts, q, kernel)[:, None]
    a_p = sharp_kde(pts, p, kernel) / kde(pts, p, kernel)
    a_q = sharp_kde(pts, q, kernel) / kde(pts, q, kernel)
    G = s_p - s_q
    R = (a_p - a_q)[:, None] * s_p
    return dict(s_p=s_p, s_q=s_q, a_p=a_p, a_q=a_q, G=G, R=R, P=a_q,
                V=a_q[:, None] * G + R)


def pgr_family(kernel: Kernel, gamma: float = 1.0, eta: float = 1.0,
               mobility_reference: Optional[float] = None) -> DriftField:
 
    if mobility_reference is not None:
        mobility_reference = float(mobility_reference)
        if not np.isfinite(mobility_reference) or mobility_reference <= 0:
            raise ValueError("mobility_reference must be finite and positive")

    def at(pts, p, q):
        dec = decompose(pts, p, q, kernel)
        mobility = (dec["a_q"] ** gamma if mobility_reference is None else
                    mobility_reference * (dec["a_q"] / mobility_reference) ** gamma)
        return mobility[:, None] * dec["G"] + eta * dec["R"]

    mobility_expr = (rf"a_q^{{{gamma:g}}}" if mobility_reference is None else
                     rf"a_0(a_q/a_0)^{{{gamma:g}}}")
    return DriftField(f"pgr_g{gamma:g}_e{eta:g}", at,
                      label=rf"${mobility_expr}G+{eta:g}R$",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma,
                                gamma=gamma, eta=eta,
                                mobility_reference=mobility_reference))


def power_family(kernel: Kernel, gamma: float = 1.0,
                 mobility_reference: Optional[float] = None) -> DriftField:
    """Separate-normalization exponent  V = a_p^gamma s_p^# - a_q^gamma s_q^#."""

    if mobility_reference is not None:
        mobility_reference = float(mobility_reference)
        if not np.isfinite(mobility_reference) or mobility_reference <= 0:
            raise ValueError("mobility_reference must be finite and positive")

    def at(pts, p, q):
        dec = decompose(pts, p, q, kernel)
        if mobility_reference is None:
            ap, aq = dec["a_p"] ** gamma, dec["a_q"] ** gamma
        else:
            ap = mobility_reference * (dec["a_p"] / mobility_reference) ** gamma
            aq = mobility_reference * (dec["a_q"] / mobility_reference) ** gamma
        return ap[:, None] * dec["s_p"] - aq[:, None] * dec["s_q"]

    return DriftField(f"power_g{gamma:g}", at, label=rf"$a^{{{gamma:g}}}$ separate",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma, gamma=gamma,
                                mobility_reference=mobility_reference))


NORMALIZERS: dict[str, Callable] = {
    "row_separate": None,                                        # separate p_k, q_k
    "deng":   None,                                              
    "p":      lambda pk, qk: pk,
    "q":      lambda pk, qk: qk,
    "geo":    lambda pk, qk: np.sqrt(pk) * np.sqrt(qk),
    "arith":  lambda pk, qk: 0.5 * (pk + qk),
    "harm":   lambda pk, qk: 2.0 * pk * qk / (pk + qk + _EPS),
    "none":   lambda pk, qk: np.ones_like(pk),
}


def common_normalization(kernel: Kernel, mode: str = "geo",
                         floor: float = 0.0) -> DriftField:
    """Same numerator N_p - N_q, one *shared* denominator A(p_k, q_k) + floor."""
    if mode not in NORMALIZERS:
        raise KeyError(f"unknown normalization {mode!r}; available: {sorted(NORMALIZERS)}")
    floor = float(floor)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and non-negative")

    def at(pts, p, q):
        Np = mean_shift_numerator(pts, p, kernel)
        Nq = mean_shift_numerator(pts, q, kernel)
        pk, qk = kde(pts, p, kernel), kde(pts, q, kernel)
        if mode in {"row_separate", "deng"}:
            return Np / (pk[:, None] + floor) - Nq / (qk[:, None] + floor)
        denominator = NORMALIZERS[mode](pk, qk)
        if mode != "none":
            denominator = denominator + floor
        return (Np - Nq) / denominator[:, None]

    return DriftField(f"norm_{mode}", at, label=f"common {mode}",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma,
                                mode=mode, floor=floor))


def density_floor(target: Emp, kernel: Kernel, frac: float = 0.3) -> float:
    """An absolute floor set from the target's own density scale."""
    frac = float(frac)
    if not np.isfinite(frac) or frac < 0:
        raise ValueError("frac must be finite and non-negative")
    return float(frac * np.median(kde(target.X, target, kernel)))


# --------------------------------------------------------------------------- #
# Coifman-Lafon alpha-normalization
# --------------------------------------------------------------------------- #
def _log_self_density(mu: Emp, kernel: Kernel) -> np.ndarray:
    """log rho_mu(y_l) = log E_{y~mu}[k(y_l, y)] at the support of mu."""
    if kernel.logK is not None:
        L = kernel.logK(mu.X, mu.X) + np.log(mu.w + 1e-300)[None, :]
    else:
        L = np.log(kernel.K(mu.X, mu.X) + 1e-300) + np.log(mu.w + 1e-300)[None, :]
    m = L.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(L - m).sum(axis=1))


def alpha_shift(pts: np.ndarray, mu: Emp, kernel: Kernel, alpha: float,
                log_rho: Optional[np.ndarray] = None) -> np.ndarray:
    """Coifman-Lafon alpha-reweighted mean shift"""
    if log_rho is None:
        log_rho = _log_self_density(mu, kernel)
    base = (kernel.logK(pts, mu.X) if kernel.logK is not None
            else np.log(kernel.K(pts, mu.X) + 1e-300))
    L = base + np.log(mu.w + 1e-300)[None, :] - alpha * log_rho[None, :]
    Kw = np.exp(L - L.max(axis=1, keepdims=True))
    return (Kw @ mu.X) / (Kw.sum(axis=1, keepdims=True) + _EPS) - pts


def alpha_mean_shift(kernel: Kernel, alpha: float = 0.0) -> DriftField:
    """Alpha-normalized mean shift (alpha=0 is the population row field)."""
    def at(pts, p, q):
        return alpha_shift(pts, p, kernel, alpha) - alpha_shift(pts, q, kernel, alpha)

    return DriftField(f"alpha_{alpha:g}", at, label=rf"$\alpha={alpha:g}$",
                      meta=dict(kernel=kernel.name, sigma=kernel.sigma, alpha=alpha))


# --------------------------------------------------------------------------- #
# Transport-based fields (defined at q.X; extended to a grid by nearest neighbour)
# --------------------------------------------------------------------------- #
def _nn_extend(values_at_q: np.ndarray, pts: np.ndarray, q: Emp) -> np.ndarray:
    """Nearest-neighbour extension. Only for plotting: it is not the true field."""
    return values_at_q[cdist(pts, q.X).argmin(axis=1)]


def ot() -> DriftField:
    """Barycentric projection of one optimal plan T-Id."""
    def at(pts, p, q):
        V = barycentric_map(q, p) - q.X
        return V if pts is q.X else _nn_extend(V, pts, q)

    return DriftField("ot", at, label=r"OT $T_{q,p}-\mathrm{Id}$")


def linear_potential(kappa: float = 1.0) -> DriftField:
    """V = -kappa (x - mean(p))."""
    def at(pts, p, q):
        return -kappa * (pts - p.mean())

    return DriftField("linear", at, label=r"$-\kappa(x-\bar p)$", meta=dict(kappa=kappa))


# --------------------------------------------------------------------------- #
# Flow-matching induced drift
# --------------------------------------------------------------------------- #
def _fm_barycentre(pts, mu: Emp, t: float) -> np.ndarray:
    """E[Y | X_t = x] for X_t = (1-t) X_0 + t Y, X_0 ~ N(0, I), Y ~ mu"""
    s = 1.0 - t
    L = -cdist(pts, t * mu.X, "sqeuclidean") / (2.0 * s * s) + np.log(mu.w + 1e-300)[None, :]
    Kw = np.exp(L - L.max(axis=1, keepdims=True))
    return (Kw @ mu.X) / (Kw.sum(axis=1, keepdims=True) + _EPS)


def flow_matching(t: float, t_max: float = 0.995) -> DriftField:
    """FM velocity drift:  V = (E[Y|X_t] - E[X|X_t]) / (1-t)."""
    t, t_max = float(t), float(t_max)
    if not np.isfinite(t) or not np.isfinite(t_max):
        raise ValueError("t and t_max must be finite")
    if not 0.0 <= t <= t_max < 1.0:
        raise ValueError("flow_matching requires 0 <= t <= t_max < 1")

    def at(pts, p, q):
        return (_fm_barycentre(pts, p, t) - _fm_barycentre(pts, q, t)) / (1.0 - t)

    return DriftField(f"fm_t{t:.3f}", at, label=rf"FM $t={t:.2f}$", meta=dict(t=t))


# --------------------------------------------------------------------------- #
# Diffusion / score regularization  (notes, sec. "Diffusion regularized flows")
# --------------------------------------------------------------------------- #
def score_regularized(base: DriftField, beta: float, kernel: Kernel,
                      score_p: Optional[Callable] = None) -> DriftField:
    """V^beta = V_base + beta v_KL,  v_KL[q] = grad log p - grad log q.

    Neither score is available on a point cloud, so both are plugged in:
      * grad log q  by the KDE score at bandwidth `kernel.sigma`;
      * grad log p  by `score_p(pts)` when the target has an analytic score
        (the Gaussian dataset does), else by the same KDE.
    """
    def at(pts, p, q):
        s_p = score_p(pts) if score_p is not None else smoothed_score(pts, p, kernel)
        s_q = smoothed_score(pts, q, kernel)
        return base.at(pts, p, q) + beta * (s_p - s_q)

    return DriftField(f"{base.name}+kl{beta:g}", at,
                      label=rf"{base.label} $+{beta:g}\,v_{{KL}}$",
                      meta=dict(**base.meta, beta=beta,
                                score_sigma=kernel.sigma,
                                analytic_score_p=score_p is not None))


def gaussian_score(mean: np.ndarray, cov: np.ndarray) -> Callable:
    """Analytic grad log N(mean, cov), for the exactly-solvable dataset."""
    mean = np.asarray(mean, float)
    prec = np.linalg.inv(np.asarray(cov, float))
    return lambda pts: -(pts - mean) @ prec


# --------------------------------------------------------------------------- #
# Norm matching: isolate the *direction* of a field from its magnitude
# --------------------------------------------------------------------------- #
def norm_matched(field: DriftField, reference: DriftField) -> DriftField:
    """Rescale `field` at every step to the L2(q) norm of `reference`."""
    def _l2q(V, q):
        return np.sqrt((q.w * (V ** 2).sum(-1)).sum()) + 1e-14

    def at(pts, p, q):
        gain = _l2q(reference.at(q.X, p, q), q) / _l2q(field.at(q.X, p, q), q)
        return gain * field.at(pts, p, q)

    return DriftField(f"{field.name}_nm", at, label=f"{field.label} (norm-matched)",
                      meta=dict(**field.meta, norm_matched_to=reference.name))


# --------------------------------------------------------------------------- #
# One-line construction from a spec
# --------------------------------------------------------------------------- #
def make_drift(kind: str, kernel_name: str = "Laplacian", sigma: float = 1.0,
               **kwargs) -> DriftField:
    """Build any drift from strings, for config cells and sweeps.

    >>> make_drift("mean_shift", "Gaussian", 1.0)
    DriftField(mean_shift, kernel='Gaussian', sigma=1.0, exclude_self=False)
    >>> make_drift("ot")
    DriftField(ot)
    """
    from .sinkhorn import sinkhorn_drift          # local import: avoids a cycle

    if kind == "ot":
        return ot()
    if kind == "linear":
        return linear_potential(**kwargs)
    if kind == "flow_matching":
        return flow_matching(**kwargs)
    if kind == "sinkhorn":
        return sinkhorn_drift(sigma=sigma, **kwargs)
    if kind == "deng_official":
        if kernel_name != "Laplacian":
            raise ValueError("Deng Algorithm 2 uses a Laplacian affinity; "
                             "set KERNEL='Laplacian'")
        if "temperature" in kwargs:
            temperature = float(kwargs.pop("temperature"))
            if float(sigma) != 1.0 and not np.isclose(float(sigma), temperature):
                raise ValueError(
                    "specify the Algorithm 2 temperature either through sigma "
                    "or `temperature`, not two different values"
                )
        else:
            temperature = float(sigma)
        return deng_official(temperature=temperature, **kwargs)

    kern = get_kernel(kernel_name, sigma)
    builders = {
        "mean_shift":           mean_shift,
        "ms_unnorm":            mean_shift_unnormalized,
        "mmd":                  mmd,
        "sharp_core":           sharp_core,
        "pgr":                  pgr_family,
        "power":                power_family,
        "common_normalization": common_normalization,
        "alpha":                alpha_mean_shift,
        "sinkhorn_proxy":       sinkhorn_proxy,
    }
    if kind not in builders:
        special = ["ot", "linear", "flow_matching", "sinkhorn",
                   "deng_official"]
        raise KeyError(f"unknown drift {kind!r}; available: "
                       f"{sorted(list(builders) + special)}")
    return builders[kind](kern, **kwargs)
