from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial.distance import cdist

_EPS_R = 1e-12  # keeps ||x-y|| differentiable at 0 for the non-smooth kernels


def _dist(X, Y, eps: float = _EPS_R) -> np.ndarray:
    return np.sqrt(cdist(X, Y, "sqeuclidean") + eps)


@dataclass
class Kernel:
    """A radial kernel of bandwidth sigma, with everything the drifts need."""

    name: str
    sigma: float
    K: Callable
    grad_x_K: Callable
    logK: Optional[Callable] = None
    sharp_K: Optional[Callable] = None

    def __repr__(self):
        return f"Kernel({self.name}, sigma={self.sigma:g})"

    @property
    def has_sharp(self) -> bool:
        return self.sharp_K is not None



def make_gaussian_kernel(sigma: float) -> Kernel:
    def logK(X, Y):
        return -cdist(X, Y, "sqeuclidean") / (2 * sigma ** 2)

    def K(X, Y):
        return np.exp(logK(X, Y))

    def grad_x_K(X, Y):
        return K(X, Y)[:, :, None] * (Y[None] - X[:, None]) / sigma ** 2

    def sharp_K(X, Y):
        return sigma ** 2 * K(X, Y)

    return Kernel("Gaussian", sigma, K, grad_x_K, logK, sharp_K)



def make_laplacian_kernel(sigma: float) -> Kernel:
    def logK(X, Y):
        return -_dist(X, Y) / sigma

    def K(X, Y):
        return np.exp(logK(X, Y))

    def grad_x_K(X, Y):
        diff = X[:, None] - Y[None]
        r = _dist(X, Y)
        return -(diff / (r[:, :, None] * sigma)) * np.exp(-r / sigma)[:, :, None]

    def sharp_K(X, Y):
        r = _dist(X, Y)
        return sigma * (r + sigma) * np.exp(-r / sigma)

    return Kernel("Laplacian", sigma, K, grad_x_K, logK, sharp_K)


def make_matern_kernel(sigma: float) -> Kernel:
    c = np.sqrt(5.0) / sigma

    def logK(X, Y):
        u = c * _dist(X, Y)
        return np.log1p(u + u ** 2 / 3.0) - u

    def K(X, Y):
        u = c * _dist(X, Y)
        return (1.0 + u + u ** 2 / 3.0) * np.exp(-u)

    def grad_x_K(X, Y):
        
        u = c * _dist(X, Y)
        coeff = -(5.0 / (3.0 * sigma ** 2)) * (1.0 + u) * np.exp(-u)
        return coeff[:, :, None] * (X[:, None] - Y[None])

    def sharp_K(X, Y):
        u = c * _dist(X, Y)
        return (sigma ** 2 / 5.0) * np.exp(-u) * (5.0 + 5.0 * u + 2.0 * u ** 2 + u ** 3 / 3.0)

    return Kernel("Matern", sigma, K, grad_x_K, logK, sharp_K)


def make_imq_kernel(sigma: float) -> Kernel:
    def logK(X, Y):
        return -0.5 * np.log1p(cdist(X, Y, "sqeuclidean") / sigma ** 2)

    def K(X, Y):
        return np.exp(logK(X, Y))

    def grad_x_K(X, Y):
        base = 1.0 + cdist(X, Y, "sqeuclidean") / sigma ** 2
        return -((X[:, None] - Y[None]) / sigma ** 2) * base[:, :, None] ** (-1.5)

    return Kernel("IMQ", sigma, K, grad_x_K, logK, sharp_K=None)


KERNELS: dict[str, Callable[[float], Kernel]] = {
    "Gaussian":  make_gaussian_kernel,
    "Laplacian": make_laplacian_kernel,
    "Matern":    make_matern_kernel,
    "IMQ":       make_imq_kernel,
}


def get_kernel(name: str, sigma: float) -> Kernel:
    if name not in KERNELS:
        raise KeyError(f"unknown kernel {name!r}; available: {sorted(KERNELS)}")
    return KERNELS[name](float(sigma))


# --------------------------------------------------------------------------- #
# Density / numerator building blocks shared by every kernelized drift
# --------------------------------------------------------------------------- #
def kde(pts: np.ndarray, mu, kernel: Kernel, eps: float = 1e-300) -> np.ndarray:
    """mu_k(x) = E_{y~mu}[k(x,y)], evaluated at `pts`.  Shape (n_pts,)."""
    return (kernel.K(pts, mu.X) * mu.w[None, :]).sum(1) + eps


def sharp_kde(pts: np.ndarray, mu, kernel: Kernel, eps: float = 1e-300) -> np.ndarray:
    """mu_{k#}(x) = E_{y~mu}[k#(x,y)].  Raises if the kernel has no companion."""
    if not kernel.has_sharp:
        raise ValueError(
            f"{kernel.name} has no registered positive tail-normalized sharp companion "
            "(see module docstring)"
        )
    return (kernel.sharp_K(pts, mu.X) * mu.w[None, :]).sum(1) + eps


def mean_shift_numerator(pts: np.ndarray, mu, kernel: Kernel) -> np.ndarray:
    """N_mu(x) = E_{y~mu}[k(x,y)(y-x)].  Shape (n_pts, d).

    Equal to grad_x mu_{k#}(x) whenever k# exists.
    """
    Kw = kernel.K(pts, mu.X) * mu.w[None, :]
    return np.einsum("nm,nmd->nd", Kw, mu.X[None] - pts[:, None])


def softmax_barycentre(pts: np.ndarray, mu, kernel: Kernel, eps: float = 1e-300) -> np.ndarray:
    """m_mu(x) = E_{y~mu}[k(x,y) y] / E_{y~mu}[k(x,y)], computed in log-space."""
    if kernel.logK is not None:
        L = kernel.logK(pts, mu.X) + np.log(mu.w + 1e-300)[None, :]
        Kw = np.exp(L - L.max(axis=1, keepdims=True))
    else:
        Kw = kernel.K(pts, mu.X) * mu.w[None, :]
    return (Kw @ mu.X) / (Kw.sum(axis=1, keepdims=True) + eps)


def smoothed_score(pts: np.ndarray, mu, kernel: Kernel) -> np.ndarray:
    """s_{mu,sigma}(x) = grad_x log mu_k(x), the score of the smoothed density."""
    grad = np.einsum("nmd,m->nd", kernel.grad_x_K(pts, mu.X), mu.w)
    return grad / kde(pts, mu, kernel)[:, None]


def plot_kernel_profiles(sigma: float = 1.0, ax=None, show_sharp: bool = True):
    """1-D radial profiles of every kernel (and its companion) in the registry."""
    import matplotlib.pyplot as plt

    X = np.linspace(-4, 4, 400)[:, None]
    Y = np.zeros((1, 1))
    if ax is None:
        _, ax = plt.subplots(1, 2 if show_sharp else 1,
                             figsize=(9 if show_sharp else 5, 3.5), squeeze=False)
        ax = ax.ravel()
    ax = np.atleast_1d(ax)

    for name, build in KERNELS.items():
        kern = build(sigma)
        ax[0].plot(X[:, 0], kern.K(X, Y)[:, 0], lw=2, label=name)
        if show_sharp and kern.has_sharp:
            ax[1].plot(X[:, 0], kern.sharp_K(X, Y)[:, 0], lw=2, label=f"{name}")
    ax[0].set(title=rf"$k_\sigma(x,0)$   ($\sigma={sigma:g}$)", xlabel="$x$")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    if show_sharp and len(ax) > 1:
        ax[1].set(title=r"sharp companion $k^\#(x,0)$", xlabel="$x$")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    return ax
