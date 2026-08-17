from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import ot
from scipy.linalg import inv
from scipy.sparse.linalg import cg
from scipy.spatial.distance import cdist

from .measures import Emp


# =========================================================================== #
# Parameterized maps
# =========================================================================== #
class ParametricMap:
    """T_theta : R^2 -> R^2, with its parameter Jacobian."""

    dim: int

    def forward(self, theta, z):
        raise NotImplementedError

    def jacobian(self, theta, z):
        raise NotImplementedError

    def metric(self, theta, z) -> np.ndarray:
        """G(theta) = E_{z~rho_0}[ J(z)^T J(z) ], the pullback metric tensor."""
        J = self.jacobian(theta, z)
        return np.einsum("nij,nik->jk", J, J) / len(z)

    def lifted_gradient(self, theta, z, wgrad: np.ndarray) -> np.ndarray:
        """grad_theta F(T_theta) (chain rule)"""
        return np.einsum("nij,ni->j", self.jacobian(theta, z), wgrad) / len(z)


class AffineMap(ParametricMap):
    """T(x) = R(phi) diag(s1, s2) x + mu,  theta = (phi, log s1, log s2, mu)."""

    dim = 5

    @staticmethod
    def _rot(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s], [s, c]])

    def linear_part(self, theta):
        return self._rot(theta[0]) @ np.diag(np.exp(theta[1:3]))

    def forward(self, theta, z):
        return z @ self.linear_part(theta).T + theta[3:5]

    def pushforward(self, theta):
        """(mean, covariance) of T_# N(0, I), exactly."""
        A = self.linear_part(theta)
        return theta[3:5].copy(), A @ A.T

    def jacobian(self, theta, z):
        phi, s = theta[0], np.exp(theta[1:3])
        R, dR = self._rot(phi), self._rot(phi + np.pi / 2)     # R'(phi) = R(phi + pi/2)
        N = len(z)
        J = np.zeros((N, 2, self.dim))
        J[:, :, 0] = (z * s) @ dR.T                            # d/dphi
        J[:, :, 1] = np.outer(s[0] * z[:, 0], R[:, 0])         # d/dlog s1
        J[:, :, 2] = np.outer(s[1] * z[:, 1], R[:, 1])         # d/dlog s2
        J[:, 0, 3] = J[:, 1, 4] = 1.0                          # d/dmu
        return J

    def metric(self, theta, z=None) -> np.ndarray:
        if z is not None:
            return super().metric(theta, z)
        s = np.exp(theta[1:3])
        return np.diag([s[0] ** 2 + s[1] ** 2, s[0] ** 2, s[1] ** 2, 1.0, 1.0])


class MLPMap(ParametricMap):
    """T(z) = W2 tanh(W1 z + c1) + c2, with h hidden units: 5h + 2 parameters."""

    def __init__(self, h: int = 8):
        self.h = h
        self.dim = 5 * h + 2

    def unpack(self, theta):
        h = self.h
        W1 = theta[:2 * h].reshape(h, 2)
        c1 = theta[2 * h:3 * h]
        W2 = theta[3 * h:5 * h].reshape(2, h)
        c2 = theta[5 * h:5 * h + 2]
        return W1, c1, W2, c2

    def forward(self, theta, z):
        W1, c1, W2, c2 = self.unpack(theta)
        return np.tanh(z @ W1.T + c1) @ W2.T + c2

    def jacobian(self, theta, z):
        W1, c1, W2, c2 = self.unpack(theta)
        h = self.h
        hid = np.tanh(z @ W1.T + c1)                 # (N, h)
        dhid = 1.0 - hid ** 2                        # (N, h)
        N = len(z)
        J = np.zeros((N, 2, self.dim))
        back = dhid[:, None, :] * W2[None, :, :]     # (N, 2, h) = dT/dpre-activation
        J[:, :, :2 * h] = (back[:, :, :, None] * z[:, None, None, :]).reshape(N, 2, 2 * h)
        J[:, :, 2 * h:3 * h] = back
        for i in range(2):                           # dT_i/dW2[i, j] = hid_j
            J[:, i, 3 * h + i * h:3 * h + (i + 1) * h] = hid
        J[:, 0, 5 * h] = J[:, 1, 5 * h + 1] = 1.0
        return J

    def init(self, rng, scale: float = 0.5) -> np.ndarray:
        theta = np.concatenate([scale * rng.standard_normal(2 * self.h),
                                0.3 * rng.standard_normal(self.h),
                                scale * rng.standard_normal(2 * self.h),
                                np.zeros(2)])
        return theta


# =========================================================================== #
# The moment-matching objective (experiments 1 and 2)
# =========================================================================== #
def kl_gaussian(mu_q, S_q, mu_p, S_p) -> float:
    """KL( N(mu_q, S_q) || N(mu_p, S_p) )."""
    d = len(mu_q)
    Pi = inv(S_p)
    diff = mu_p - mu_q
    _, ld_p = np.linalg.slogdet(S_p)
    _, ld_q = np.linalg.slogdet(S_q)
    return float(0.5 * (np.trace(Pi @ S_q) + diff @ Pi @ diff - d + ld_p - ld_q))


def moment_kl(x: np.ndarray, mu_p, S_p, jitter: float = 1e-6):
    """KL of the empirical Gaussian moments of `x` against N(mu_p, S_p), and its
    *Wasserstein gradient field* at those points."""
    N = len(x)
    mu = x.mean(0)
    Xc = x - mu
   
    S = Xc.T @ Xc / N + jitter * np.eye(x.shape[1])
    loss = kl_gaussian(mu, S, mu_p, S_p)

    Pi, Si = inv(S_p), inv(S)
    wgrad = Pi @ (x - mu_p).T - Si @ Xc.T
    return loss, wgrad.T


@dataclass
class OptTrace:
    """Parameter path plus the loss, for plotting."""

    theta: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    cond_G: list = field(default_factory=list)

    def as_arrays(self):
        return (np.array(self.theta), np.array(self.loss),
                np.array(self.grad_norm), np.array(self.cond_G))


def optimize(model: ParametricMap, theta0: np.ndarray, z: np.ndarray,
             mu_p, S_p, *, lr: float = 0.1, n_steps: int = 300,
             method: str = "euclidean", ridge: float = 1e-5,
             max_step: float = 10.0, track_cond: bool = True) -> OptTrace:
    """Minimize KL(q_theta || N(mu_p, S_p)) by Euclidean or natural gradient.

    method = "euclidean" : dtheta = -lr * grad
    method = "natural"   : dtheta = -lr * (G + ridge I)^{-1} grad
    """
    if method not in {"euclidean", "natural"}:
        raise ValueError("method must be 'euclidean' or 'natural'")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    for value, name, strict in ((lr, "lr", False), (ridge, "ridge", False),
                                (max_step, "max_step", True)):
        if not np.isfinite(float(value)) or (float(value) <= 0 if strict else float(value) < 0):
            relation = "positive" if strict else "non-negative"
            raise ValueError(f"{name} must be finite and {relation}")
    theta = np.asarray(theta0, float).copy()
    trace = OptTrace()
    for _ in range(n_steps):
        x = model.forward(theta, z)
        loss, wgrad = moment_kl(x, mu_p, S_p)
        grad = model.lifted_gradient(theta, z, wgrad)

        G = model.metric(theta, z)
        trace.theta.append(theta.copy())
        trace.loss.append(loss)
        trace.grad_norm.append(float(np.linalg.norm(grad)))
        if track_cond:
            ev = np.linalg.eigvalsh(G)
            trace.cond_G.append(float(ev[-1] / max(ev[0], 1e-14)))

        step = grad if method == "euclidean" else \
            inv(G + ridge * np.eye(model.dim)) @ grad
        norm = np.linalg.norm(step)
        if norm > max_step:
            step = step * max_step / norm
        theta = theta - lr * step

    x = model.forward(theta, z)
    loss, wgrad = moment_kl(x, mu_p, S_p)
    grad = model.lifted_gradient(theta, z, wgrad)
    G = model.metric(theta, z)
    trace.theta.append(theta.copy())
    trace.loss.append(loss)
    trace.grad_norm.append(float(np.linalg.norm(grad)))
    if track_cond:
        ev = np.linalg.eigvalsh(G)
        trace.cond_G.append(float(ev[-1] / max(ev[0], 1e-14)))
    return trace


def mc_noise_floor(model: ParametricMap, theta: np.ndarray, z: np.ndarray) -> dict:
    """Spectrum of G against the heuristic scale lambda_max / sqrt(N)."""
    ev = np.sort(np.linalg.eigvalsh(model.metric(theta, z)))[::-1]
    floor = ev[0] / np.sqrt(len(z))
    return dict(eigenvalues=ev, noise_floor=float(floor),
                n_below=int((ev < floor).sum()), n_params=len(ev),
                condition=float(ev[0] / max(ev[-1], 1e-14)))


# =========================================================================== #
# Experiment 3-4: transport-projection with a parameterized map
# =========================================================================== #
def free_drift(x0: np.ndarray, p: Emp, field, n_steps: int, tau: float) -> list:
    """Unconstrained particle flow x <- x + tau V(x): the ideal transport."""
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    for _ in range(n_steps):
        x = x + tau * field(p, Emp(x))
        traj.append(x.copy())
    return traj


def transport_project(model: ParametricMap, theta0: np.ndarray, z: np.ndarray,
                      p: Emp, field, *, n_outer: int = 80, tau: float = 0.3,
                      projection: str = "euclidean", n_proj: int = 20,
                      lr_proj: float = 0.05, ridge: float = 1e-4,
                      max_step: float = 5.0, cg_atol: float = 1e-10,
                      cg_rtol: float = 0.0) -> dict:
    """The drifting training loop with an explicit map."""
    if projection not in {"euclidean", "natural", "cg"}:
        raise ValueError("projection must be 'euclidean', 'natural' or 'cg'")
    if not isinstance(n_outer, (int, np.integer)) or n_outer < 0:
        raise ValueError("n_outer must be a non-negative integer")
    if not isinstance(n_proj, (int, np.integer)) or n_proj < 1:
        raise ValueError("n_proj must be a positive integer")
    for value, name, strict in ((ridge, "ridge", False), (max_step, "max_step", True),
                                (cg_atol, "cg_atol", False), (cg_rtol, "cg_rtol", False)):
        if not np.isfinite(float(value)) or (float(value) <= 0 if strict else float(value) < 0):
            relation = "positive" if strict else "non-negative"
            raise ValueError(f"{name} must be finite and {relation}")
    theta = np.asarray(theta0, float).copy()
    N = len(z)
    traj = [model.forward(theta, z).copy()]
    thetas, mse = [theta.copy()], []
    cg_iterations, cg_info = [], []

    for _ in range(n_outer):
        x = model.forward(theta, z)
        target = x + tau * field(p, Emp(x))            # frozen transport target

        if projection == "cg":
            J = model.jacobian(theta, z)
            G = np.einsum("nij,nik->jk", J, J) / N
            b = np.einsum("nij,ni->j", J, target - x) / N
            iteration_count = [0]

            def count_iteration(_):
                iteration_count[0] += 1

            delta, info = cg(
                G + ridge * np.eye(model.dim), b, x0=np.zeros(model.dim),
                maxiter=n_proj, atol=cg_atol, rtol=cg_rtol,
                callback=count_iteration,
            )
            cg_iterations.append(iteration_count[0])
            cg_info.append(int(info))
            theta = theta + delta
        else:
            for _ in range(n_proj):
                J = model.jacobian(theta, z)
                residual = model.forward(theta, z) - target
                grad = 2.0 * np.einsum("nij,ni->j", J, residual) / N
                if projection == "natural":
                    G = np.einsum("nij,nik->jk", J, J) / N
                    grad = inv(G + ridge * np.eye(model.dim)) @ grad
                norm = np.linalg.norm(grad)
                if norm > max_step:
                    grad = grad * max_step / norm
                theta = theta - lr_proj * grad

        traj.append(model.forward(theta, z).copy())
        thetas.append(theta.copy())
        mse.append(float(np.mean(np.sum((traj[-1] - target) ** 2, axis=1))))

    return dict(traj=traj, theta=np.array(thetas), mse=np.array(mse),
                projection=projection, n_proj=n_proj, compute=n_outer * n_proj,
                cg_iterations=np.asarray(cg_iterations, dtype=int),
                cg_info=np.asarray(cg_info, dtype=int), cg_atol=cg_atol,
                cg_rtol=cg_rtol)


def distance_to_free(traj_param: Sequence[np.ndarray],
                     traj_free: Sequence[np.ndarray]) -> np.ndarray:
    """Mean per-particle distance to the ideal flow at each k."""
    n = min(len(traj_param), len(traj_free))
    return np.array([np.mean(np.linalg.norm(traj_param[k] - traj_free[k], axis=1))
                     for k in range(n)])


def W1(x: np.ndarray, y: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between two uniform clouds."""
    a = np.full(len(x), 1.0 / len(x))
    b = np.full(len(y), 1.0 / len(y))
    return float(ot.emd2(a, b, cdist(x, y)))
