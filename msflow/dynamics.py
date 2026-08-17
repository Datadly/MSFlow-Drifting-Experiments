from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from .measures import Emp, W2, Dataset
from .kernels import Kernel
from .drifts import DriftField
from .metrics import (MMD2, sharp_energy, mode_metrics, mass_split_error,
                      smoothed_kl, drift_norm, step_diagnostics,
                      f_dissipativity, fit_exponential_rate)


# =========================================================================== #
# Schedules  s(k)
# =========================================================================== #
def constant(value: float) -> Callable[[int], float]:
    return lambda k: value


def exp_anneal(s0: float, s_inf: float, kappa: float) -> Callable[[int], float]:
    """s_k = s_inf + (s_0 - s_inf) e^{-k/kappa}."""
    return lambda k: s_inf + (s0 - s_inf) * np.exp(-k / kappa)


def linear_anneal(s0: float, s_inf: float, n_iter: int) -> Callable[[int], float]:
    return lambda k: s_inf + (s0 - s_inf) * (1.0 - min(max(k / n_iter, 0.0), 1.0))


def cosine_anneal(s0: float, s_inf: float, n_iter: int) -> Callable[[int], float]:
    def sched(k):
        u = min(max(k / n_iter, 0.0), 1.0)
        return s_inf + 0.5 * (s0 - s_inf) * (1.0 + np.cos(np.pi * u))
    return sched


def poly_anneal(s0: float, s_inf: float, n_iter: int, power: float = 1.0):
    """power < 1 keeps the bandwidth large longer; power > 1 cools early."""
    def sched(k):
        u = min(max(k / n_iter, 0.0), 1.0)
        return s_inf + (s0 - s_inf) * (1.0 - u) ** power
    return sched


def plateau_then_exp(s0: float, s_inf: float, n_iter: int,
                     plateau_frac: float = 0.25, kappa_frac: float = 0.25):
    """Hold s_0 while the mass spreads, then cool exponentially."""
    k0, kappa = int(plateau_frac * n_iter), kappa_frac * n_iter
    return lambda k: s0 if k <= k0 else s_inf + (s0 - s_inf) * np.exp(-(k - k0) / kappa)


def fm_time(n_iter: int, t_max: float = 0.995) -> Callable[[int], float]:
    """Flow-matching time t_k = min(k/n_iter, t_max)."""
    return lambda k: min(k / n_iter, t_max)


SCHEDULES = {"constant": constant, "exp": exp_anneal, "linear": linear_anneal,
             "cosine": cosine_anneal, "poly": poly_anneal,
             "plateau_exp": plateau_then_exp, "fm_time": fm_time}


# =========================================================================== #
# Result containers
# =========================================================================== #
@dataclass
class Run:
    """One trajectory plus its probe curves."""

    name: str
    label: str
    X: list                       # trajectory, list of (n, d) arrays
    w: np.ndarray                 # particle weights (constant along the flow)
    probe_k: np.ndarray           # iterations at which probes were evaluated
    probes: Dict[str, np.ndarray]
    schedule: np.ndarray          # the value of s at each step (empty if static)
    tau: float
    n_iter: int
    store_every: int = 1          # frame i of X is iteration i * store_every
    meta: dict = field(default_factory=dict)

    @property
    def q_final(self) -> Emp:
        return Emp(self.X[-1], self.w)

    def iteration_of_frame(self, i: int) -> int:
        return min(i * self.store_every, self.n_iter)

    def q_at_frame(self, i: int) -> Emp:
        return Emp(self.X[min(i, len(self.X) - 1)], self.w)

    def final(self, key: str) -> float:
        v = self.probes.get(key)
        return float(v[-1]) if v is not None and len(v) else np.nan

    def first_k_where(self, key: str, predicate: Callable[[float], bool]) -> int:
        """First probed iteration where `predicate(probe)` holds, else -1."""
        v = self.probes.get(key)
        if v is None:
            return -1
        hits = np.where([predicate(x) for x in v])[0]
        return int(self.probe_k[hits[0]]) if len(hits) else -1

    def rate(self, key: str = "W2") -> float:
        """Fitted exponential decay rate of a probe over its middle window."""
        return fit_exponential_rate(self.probe_k, self.probes[key])


class Runs(dict):
    """A dict of `Run`s, keyed by a short name.

    `runs.summary(...)` is a shortcut for `report.runs_table(runs, ...)`; the
    formatting lives there so there is only one implementation of it.
    """

    def summary(self, **kwargs):
        from .report import runs_table
        return runs_table(self, **kwargs)


# =========================================================================== #
# The engine
# =========================================================================== #
def _clip(V: np.ndarray, cap: Optional[float]) -> np.ndarray:
    """Cap ||V(x)|| at `cap`, preserving direction.

    Norm clipping (rather than per-coordinate clipping) matters here: box
    clipping bends large velocities towards the diagonals, which corrupts every
    direction-based diagnostic (dissipativity, cocoercivity, curl).
    """
    if cap is None:
        return V
    cap = float(cap)
    if not np.isfinite(cap) or cap < 0:
        raise ValueError("clip cap must be finite and non-negative, or None")
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V * np.minimum(1.0, cap / (n + 1e-12))


def run(q0: Emp, p: Emp, drift: Union[DriftField, Callable], *,
        n_iter: int = 500, tau: float = 0.04,
        schedule: Optional[Callable[[int], float]] = None,
        clip: Optional[float] = 10.0,
        probes: Optional[Dict[str, Callable]] = None, probe_every: int = 10,
        store_every: int = 1, name: str = "", label: str = "") -> Run:
    """Iterate the explicit drift scheme and record probes.

    Parameters
    ----------
    drift      : a `DriftField`, or a builder `s -> DriftField` (then `schedule`
                 is required and the field is rebuilt at every step).
    tau        : step size of the explicit Euler / KM iteration.
    clip       : cap on ||V(x)||; None disables it.
    probes     : name -> `probe(k, p, q, V) -> float`, evaluated (after clipping)
                 every `probe_every` steps.  See `standard_probes`.
    store_every: keep one frame out of `store_every` in the trajectory.

    Returns a `Run`.
    """
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer")
    if not isinstance(probe_every, (int, np.integer)) or probe_every < 1:
        raise ValueError("probe_every must be a positive integer")
    if not isinstance(store_every, (int, np.integer)) or store_every < 1:
        raise ValueError("store_every must be a positive integer")
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if clip is not None and (not np.isfinite(float(clip)) or float(clip) < 0):
        raise ValueError("clip must be finite and non-negative, or None")

    if isinstance(drift, DriftField):
        get_field = lambda k: drift
        base_meta, base_label = dict(drift.meta), drift.label
    else:
        if schedule is None:
            raise ValueError("a drift builder needs a `schedule`; pass a DriftField "
                             "for a static run")
        get_field = lambda k: drift(schedule(k))
        probe_field = drift(schedule(0))
        base_meta, base_label = dict(probe_field.meta), probe_field.label

    probes = probes or {}
    q = q0.copy()
    X = [q.X.copy()]
    sched_vals, probe_k = [], []
    curves: Dict[str, list] = {key: [] for key in probes}

    for k in range(n_iter):
        s = schedule(k) if schedule is not None else np.nan
        sched_vals.append(s)
        V = _clip(get_field(k)(p, q), clip)

        if k % probe_every == 0:
            probe_k.append(k)
            for key, fn in probes.items():
                curves[key].append(float(fn(k, p, q, V)))

        q = Emp(q.X + tau * V, q.w)
        if (k + 1) % store_every == 0 or k == n_iter - 1:
            X.append(q.X.copy())

    # one last probe on the final state (V recomputed at the terminal schedule)
    V = _clip(get_field(n_iter)(p, q), clip) if probes else None
    probe_k.append(n_iter)
    for key, fn in probes.items():
        curves[key].append(float(fn(n_iter, p, q, V)))

    return Run(name=name or base_label, label=label or base_label, X=X, w=q0.w,
               probe_k=np.array(probe_k),
               probes={key: np.array(v) for key, v in curves.items()},
               schedule=np.array(sched_vals), tau=tau, n_iter=n_iter,
               store_every=store_every,
               meta=dict(base_meta, tau=tau, n_iter=n_iter, clip=clip,
                         scheduled=schedule is not None))


# =========================================================================== #
# Standard probe sets
# =========================================================================== #
def standard_probes(data: Dataset, kernel: Optional[Kernel] = None, *,
                    with_certificates: bool = False,
                    eval_sigma: Optional[float] = None,
                    r_cover: float = 1.4, tau: Optional[float] = None
                    ) -> Dict[str, Callable]:
    """The usual diagnostics for a dataset.

    Always on
        W2         exact W_2(q^k, p)
        Vnorm      ||V||_{L2(q^k)}, the drifting loss in value
        n_covered  number of target modes covered
        entropy    evenness of the captured mass
        precision  mass within r_cover of some mode
        H          KL(q_sigma || p_sigma) at the *fixed* evaluation bandwidth

    With a kernel
        F          1/2 MMD^2_{k#}(q^k, p): a candidate sharp-MMD diagnostic
        beta_F     realized alignment with F, and its `quad` / `mechant` split

    Only the raw unnormalized field is the exact WGF velocity of ``F`` by
    construction.  For population mean shift, the Proxy or another field,
    ``beta_F`` measures whether the candidate happens to decrease at the probed
    state; the presence of a kernel does not make ``F`` a Lyapunov function.

    With `with_certificates` (2 extra OT solves per probe, so keep the stride up)
        rho_NE, rho_diss, beta_star, ot_alignment, alpha_hat, L_hat,
        rho_bound, tau_opt

    `eval_sigma` fixes the bandwidth at which H is measured; it must be the same
    across runs for the numbers to be comparable, so it defaults to a value tied
    to the dataset rather than to the run.
    """
    s_eval = eval_sigma if eval_sigma is not None else 0.5
    box, energy_cache = data.extent, {}

    coverage = _shared(lambda k, p, q, V: mode_metrics(q, data.centers, r_cover))
    P: Dict[str, Callable] = {
        "W2":        lambda k, p, q, V: W2(q, p),
        "Vnorm":     lambda k, p, q, V: drift_norm(q, V),
        "n_covered": coverage("n_covered"),
        "entropy":   coverage("coverage_entropy"),
        "precision": coverage("precision"),
        "H":         lambda k, p, q, V: smoothed_kl(q, p, s_eval, box),
    }
    if data.p_mass is not None:
        P["mass_err"] = lambda k, p, q, V: mass_split_error(q, data.centers,
                                                            data.p_mass, r_cover)
    if kernel is not None:
        P["MMD2"] = lambda k, p, q, V: MMD2(p, q, kernel)
    if kernel is not None and kernel.has_sharp:
        def energy(k, p, q, V):
            F, energy_cache["Epp"] = sharp_energy(q, p, kernel, energy_cache.get("Epp"))
            return F
        fdiss = _shared(lambda k, p, q, V: f_dissipativity(p, q, kernel, V))
        P["F"] = energy
        P["beta_F"] = fdiss("beta_F")
        P["quad"] = fdiss("quad")
        P["mechant"] = fdiss("mechant")
    if with_certificates:
        # all of them share one OT solve per probed iteration, so asking for the
        # (H) constants on top of the ratios costs nothing extra
        P.update(certificate_probes(tau if tau is not None else 0.04,
                                    keys=("rho_NE", "rho_diss", "beta_star",
                                          "ot_alignment", "alpha_hat", "L_hat",
                                          "rho_bound", "tau_opt")))
    return P


def _shared(compute: Callable) -> Callable[[str], Callable]:
    """Turn a dict-valued computation into several probes sharing one evaluation.

    Probes are called once per key per iteration; `mode_metrics` and
    `f_dissipativity` each yield three of them, and `step_diagnostics` costs an
    OT solve, so recomputing per key would triple the run time for nothing.
    """
    cache: dict = {}

    def factory(key):
        def probe(k, p, q, V):
            # ``k`` alone collides when callers reuse one probe dictionary for
            # several runs.  Object identity keeps sharing within a probe step
            # while invalidating the cache for another cloud/field.
            signature = (k, id(p), id(q), id(V))
            if cache.get("signature") != signature:
                cache.clear()
                cache.update(signature=signature, value=compute(k, p, q, V))
            return cache["value"][key]
        return probe
    return factory


def certificate_probes(tau: float, keys: Sequence[str] = (
        "W2", "rho_NE", "rho_diss", "beta_star", "Vnorm",
        "ot_alignment", "alpha_hat", "L_hat", "rho_bound", "tau_opt")) -> Dict[str, Callable]:
    """Convergence certificates, all sharing a single OT solve per iteration."""
    shared = _shared(lambda k, p, q, V: step_diagnostics(p, q, V, tau))
    return {key: shared(key) for key in keys}


def replay(run_obj: Run, probe: Callable[[Emp], float],
           stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a state-only probe along a stored trajectory, after the fact.

    Returns (iterations, values) — useful to add a diagnostic without re-running.
    """
    frames = np.arange(0, len(run_obj.X), stride)
    ks = np.array([run_obj.iteration_of_frame(i) for i in frames])
    return ks, np.array([probe(run_obj.q_at_frame(i)) for i in frames])


# =========================================================================== #
# Persistence
# =========================================================================== #
def save_run(r: Run, name: str, directory: str = "runs") -> str:
    """Write a `Run` to `<directory>/<name>.npz` (+ a small JSON sidecar).

    Worth doing for the expensive ones — an exact-Sinkhorn run costs minutes,
    and reloading it lets the plots be re-made without recomputing.
    """
    import json
    import os

    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, name)
    np.savez_compressed(base + ".npz", X=np.stack(r.X), w=r.w, probe_k=r.probe_k,
                        schedule=r.schedule,
                        **{f"probe__{k}": v for k, v in r.probes.items()})
    with open(base + ".json", "w") as f:
        json.dump(dict(name=r.name, label=r.label, tau=r.tau, n_iter=r.n_iter,
                       store_every=r.store_every, probes=list(r.probes),
                       meta={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                 else str(v))
                             for k, v in r.meta.items()}), f, indent=2)
    return base + ".npz"


def load_run(name: str, directory: str = "runs") -> Run:
    """Read back a `Run` written by `save_run`."""
    import json
    import os

    base = os.path.join(directory, name)
    npz = np.load(base + ".npz")
    with open(base + ".json") as f:
        meta = json.load(f)
    return Run(name=meta["name"], label=meta["label"],
               X=[npz["X"][i] for i in range(len(npz["X"]))], w=npz["w"],
               probe_k=npz["probe_k"],
               probes={k: npz[f"probe__{k}"] for k in meta["probes"]},
               schedule=npz["schedule"], tau=meta["tau"], n_iter=meta["n_iter"],
               store_every=meta["store_every"], meta=meta["meta"])
