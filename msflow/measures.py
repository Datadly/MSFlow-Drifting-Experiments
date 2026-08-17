from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import ot
from scipy.spatial.distance import cdist

D = 2  # ambient dimension of every 2-D dataset below


# --------------------------------------------------------------------------- #
# Data Structure
# --------------------------------------------------------------------------- #
@dataclass
class Emp:
    """Empirical distribution: a weighted point cloud in R^d.

    Parameters
    ----------
    X : (n, d) support points.
    w : (n,) weights; defaults to uniform 1/n.
    """

    X: np.ndarray
    w: Optional[np.ndarray] = None

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim == 1:                       # allow 1-D input
            self.X = self.X[:, None]
        if self.X.ndim != 2 or self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError("X must have non-empty shape (n, d)")
        if not np.isfinite(self.X).all():
            raise ValueError("X must contain only finite values")
        self.n, self.d = self.X.shape
        if self.w is None:
            self.w = np.full(self.n, 1.0 / self.n)
        self.w = np.asarray(self.w, dtype=float)
        if self.w.shape != (self.n,):
            raise ValueError(f"got {self.n} points but weights have shape {self.w.shape}")
        if not np.isfinite(self.w).all() or np.any(self.w < 0):
            raise ValueError("weights must be finite and non-negative")
        total = float(self.w.sum())
        if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"weights must sum to one (got {total:.16g})")

    def copy(self) -> "Emp":
        return Emp(self.X.copy(), self.w.copy())

    def mean(self) -> np.ndarray:
        return self.w @ self.X

    def cov(self) -> np.ndarray:
        Xc = self.X - self.mean()
        return (self.w[:, None] * Xc).T @ Xc


# --------------------------------------------------------------------------- #
# Optimal transport
# --------------------------------------------------------------------------- #
_EMD_MAX_ITER = 1_000_000   # POT's default (100k) trips on clouds of a few thousand


def OT_plan(mu: Emp, nu: Emp) -> np.ndarray:
    """Exact optimal plan gamma* in Gamma_o(mu, nu) for the squared cost."""
    return ot.emd(mu.w, nu.w, cdist(mu.X, nu.X, "sqeuclidean"),
                  numItermax=_EMD_MAX_ITER)


def W2(mu: Emp, nu: Emp) -> float:
    """Exact 2-Wasserstein distance W_2(mu, nu)."""
    cost = ot.emd2(mu.w, nu.w, cdist(mu.X, nu.X, "sqeuclidean"),
                   numItermax=_EMD_MAX_ITER)
    return float(np.sqrt(max(cost, 0.0)))


def barycentric_map(mu: Emp, nu: Emp, plan: Optional[np.ndarray] = None) -> np.ndarray:
    """Barycentric projection T(x_i) = E_{gamma*}[y | x = x_i] of the OT plan.

    For an absolutely continuous mu this is the Brenier map; for discrete
    measures it is its standard surrogate, and it is what makes the OT drift
    `V_OT = T - Id` computable on clouds.
    """
    if plan is None:
        plan = OT_plan(mu, nu)
    return (plan @ nu.X) / (plan.sum(axis=1)[:, None] + 1e-15)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """A (source, target) pair plus everything the diagnostics need.

    Attributes
    ----------
    q0, p    : source and target empirical measures.
    centers  : (K, d) reference mode locations, used by the coverage metrics.
    p_mass   : (K,) target mass per mode when the target is deliberately
               imbalanced, else None.
    extent   : half-width of the plotting / quadrature box.
    obstacle : one line saying what this dataset is designed to break.
    """

    name: str
    q0: Emp
    p: Emp
    centers: np.ndarray
    extent: float
    obstacle: str = ""
    p_mass: Optional[np.ndarray] = None

    def summary(self) -> dict:
        return dict(dataset=self.name, n_source=self.q0.n, n_target=self.p.n,
                    modes=len(self.centers), extent=self.extent,
                    obstacle=self.obstacle)


def _blob(rng, center, std, n) -> np.ndarray:
    return np.asarray(center, float) + std * rng.standard_normal((n, D))


def _two_gaussians(rng, n):
    """Warm-up: a compact source facing a well-separated bimodal target."""
    centers = np.array([[-2.0, 2.0], [-2.0, -2.0]])
    X = np.vstack([_blob(rng, c, 0.35, n // 2) for c in centers])
    return Dataset("two_gaussians", Emp(_blob(rng, [3.5, 0.0], 0.5, n)), Emp(X),
                   centers, extent=5.0,
                   obstacle="none — sanity check that a drift transports at all")


def _ring(rng, n, n_modes=8, radius=4.0, mode_std=0.3):
    """Multimodality: the source must split into `n_modes` separated blobs."""
    ang = 2 * np.pi * np.arange(n_modes) / n_modes
    centers = radius * np.column_stack([np.cos(ang), np.sin(ang)])
    X = np.vstack([_blob(rng, c, mode_std, n // n_modes) for c in centers])
    return Dataset("ring", Emp(_blob(rng, [-9.0, 0.0], 0.5, n)), Emp(X),
                   centers, extent=10.0,
                   obstacle="mode splitting at long range: needs sigma >> mode gap early")


def _moons(rng, n, noise=0.1, scale=4.0, n_centers=4):
    """Curved, thin support: exposes the O(sigma^2) inward bias of mean-shift."""
    def moons(t1, t2):
        return np.vstack([np.column_stack([np.cos(t1), np.sin(t1)]),
                          np.column_stack([1.0 - np.cos(t2), 1.0 - np.sin(t2) - 0.5])])

    n1, n2 = n // 2, n - n // 2
    shift = np.array([0.5, 0.25])                       # centroid of the clean shape
    X = (moons(np.pi * rng.random(n1), np.pi * rng.random(n2)) - shift) * scale
    X = X + noise * scale * rng.standard_normal((n, D))
    tc = np.linspace(0.15, np.pi - 0.15, n_centers)
    centers = (moons(tc, tc) - shift) * scale
    return Dataset("moons", Emp(_blob(rng, [0.0, -7.0], 0.5, n)), Emp(X),
                   centers, extent=9.0,
                   obstacle="curved 1-D support: kernel smoothing contracts it inward")


def _checkerboard(rng, n, n_tiles=4, tile=1.6):
    """Many nearby modes: the field must resolve structure below the tile size."""
    cells = np.array([[i, j] for i in range(n_tiles) for j in range(n_tiles)
                      if (i + j) % 2 == 0], float) - (n_tiles - 1) / 2.0
    X = np.vstack([(c + rng.random((n // len(cells), D)) - 0.5) * tile for c in cells])
    return Dataset("checkerboard", Emp(_blob(rng, [-8.0, 0.0], 0.5, n)), Emp(X),
                   cells * tile, extent=8.0,
                   obstacle="fine multimodality: sigma must end below the tile size")


def _imbalanced(rng, n, weights=(0.85, 0.15), centers=((-5.0, 0.0), (5.0, 0.0)), std=0.4):
    """Wrong mass split, right supports."""
    centers = np.asarray(centers, float)
    m = np.asarray(weights, float); m = m / m.sum()
    X = np.vstack([_blob(rng, c, std, k) for c, k in zip(centers, rng.multinomial(n, m))])
    half = np.vstack([_blob(rng, c, std, k)
                      for c, k in zip(centers, rng.multinomial(n, [0.5, 0.5]))])
    return Dataset("imbalanced", Emp(half), Emp(X), centers, extent=8.0,
                   obstacle="mass reallocation across a gap the kernel cannot bridge",
                   p_mass=m)


def _thin_circle(rng, n, radius=4.0, width=0.06, n_centers=8):
    """Uniform mass on a thin circle: a pure curvature-bias probe."""
    th = 2 * np.pi * rng.random(n)
    r = radius + width * rng.standard_normal(n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    ang = 2 * np.pi * np.arange(n_centers) / n_centers
    return Dataset("thin_circle", Emp(_blob(rng, [0.0, -7.0], 0.5, n)), Emp(X),
                   radius * np.column_stack([np.cos(ang), np.sin(ang)]), extent=9.0,
                   obstacle="curvature bias: mean-shift pulls the ring inward by O(sigma^2/R)")


def _mass_hierarchy(rng, n, n_modes=4, radius=5.0, decay=0.5, std=0.35):
    """Geometrically decaying mode masses p_i ~ decay^i."""
    ang = 2 * np.pi * np.arange(n_modes) / n_modes
    centers = radius * np.column_stack([np.cos(ang), np.sin(ang)])
    m = decay ** np.arange(n_modes); m = m / m.sum()
    X = np.vstack([_blob(rng, c, std, k) for c, k in zip(centers, rng.multinomial(n, m))])
    # the source sits at the centre, equidistant from every mode: the experiment
    # is about *how the mass is allocated*, not about reaching the modes at all
    return Dataset("mass_hierarchy", Emp(_blob(rng, [0.0, 0.0], 1.0, n)), Emp(X),
                   centers, extent=8.0,
                   obstacle="light modes: density-dependent normalization decides if they survive",
                   p_mass=m)


def _gaussian_aniso(rng, n, mean=(2.0, 1.0), angle=np.pi / 5, scales=(2.0, 0.4)):
    """Anisotropic Gaussian target: solvable in closed form."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    Sigma = R @ np.diag(np.asarray(scales, float) ** 2) @ R.T
    L = np.linalg.cholesky(Sigma)
    X = np.asarray(mean, float) + rng.standard_normal((n, D)) @ L.T
    return Dataset("gaussian_aniso", Emp(_blob(rng, [-4.0, -3.0], 0.6, n)), Emp(X),
                   np.asarray(mean, float)[None, :], extent=8.0,
                   obstacle="none — exactly solvable reference (see msflow.gaussian)")


DATASETS: dict[str, Callable] = {
    "two_gaussians":  _two_gaussians,
    "ring":           _ring,
    "moons":          _moons,
    "checkerboard":   _checkerboard,
    "imbalanced":     _imbalanced,
    "thin_circle":    _thin_circle,
    "mass_hierarchy": _mass_hierarchy,
    "gaussian_aniso": _gaussian_aniso,
}


def load_dataset(name: str, n: int = 400, seed: int = 2026, **kwargs) -> Dataset:
    """Build a dataset from its own RNG, so the result is order-independent."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name](np.random.default_rng(seed), n, **kwargs)
