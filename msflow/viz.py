"""Plots and animations.

Two audiences are served here:

  * the screen — `animate_particles` / `animate_field` show the dynamics while
    exploring;
  * the report — `plot_snapshots`, `plot_runs`, `plot_field` produce static,
    self-contained figures.  Anything meant for the write-up should go through
    those, and be saved with `save`.

Colours are assigned per run key and memoized, so the same variant keeps its
colour across every figure of a notebook.
"""
from __future__ import annotations

import itertools
import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .measures import Emp, Dataset
from .metrics import curl_2d, rotational_fraction

plt.rcParams.update({
    "figure.dpi": 96, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.size": 10,
    "animation.embed_limit": 64,
})

FIGDIR = "figures"

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
_COLORS: Dict[str, str] = {}
_CYCLE = itertools.cycle(_PALETTE)


def color_of(key: str) -> str:
    """Stable colour for a run key (assigned on first use)."""
    if key not in _COLORS:
        _COLORS[key] = next(_CYCLE)
    return _COLORS[key]


def set_colors(mapping: Dict[str, str]) -> None:
    """Pin explicit colours, e.g. to keep 'deng' red across a whole notebook."""
    _COLORS.update(mapping)


def save(fig, name: str, subdir: str = FIGDIR) -> str:
    """Save a figure under `figures/` and return its path (for the report)."""
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, name if name.endswith((".png", ".pdf")) else name + ".png")
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def plot_dataset(data: Dataset, ax=None, figsize=(3.6, 3.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.scatter(*data.p.X.T, s=9, alpha=.45, color="#1f77b4", label=f"target $p$ (n={data.p.n})")
    ax.scatter(*data.q0.X.T, s=9, alpha=.45, color="#d62728", label=f"source $q^0$ (n={data.q0.n})")
    ax.scatter(*data.centers.T, marker="x", s=45, color="k", lw=1.4, label="mode centres")
    ax.set_aspect("equal"); ax.set_title(data.name, fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    return ax


def plot_dataset_grid(datasets: Sequence[Dataset], ncol: int = 4,
                      panel_size: float = 2.25):
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(panel_size * ncol, panel_size * nrow), squeeze=False)
    for ax, d in zip(axes.ravel(), datasets):
        plot_dataset(d, ax)
        ax.set_xlabel(d.obstacle, fontsize=7, color="0.35")
    for ax in axes.ravel()[len(datasets):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #
def field_grid(extent: float, n_grid: int = 30):
    g = np.linspace(-extent, extent, n_grid)
    xx, yy = np.meshgrid(g, g)
    return g, xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def plot_field(field, p: Emp, q: Emp, extent: float = 6.0, n_grid: int = 32,
               sub: int = 2, title: str = "", axes=None,
               figsize=(8.8, 3.7)):
    """Quiver of a drift field next to its scalar curl.

    Returns the finite-grid ratio ||curl||^2/(||curl||^2+||div||^2).  It depends
    on the box, resolution and boundary stencil: a near-zero value is compatible
    with conservativity, while a nonzero value is a diagnostic rather than a
    Helmholtz proof.  Batch fields use only their plotting extension here.
    """
    g, xx, yy, pts = field_grid(extent, n_grid)
    V = field.at(pts, p, q)
    Vx, Vy = V[:, 0].reshape(xx.shape), V[:, 1].reshape(xx.shape)
    speed, curl = np.hypot(Vx, Vy), curl_2d(Vx, Vy, g)
    rot = rotational_fraction(Vx, Vy, g)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize)
    a1, a2 = axes
    a1.scatter(*p.X.T, s=7, alpha=.25, color="#1f77b4", label="$p$")
    a1.scatter(*q.X.T, s=7, alpha=.25, color="#d62728", label="$q$")
    a1.quiver(xx[::sub, ::sub], yy[::sub, ::sub], Vx[::sub, ::sub], Vy[::sub, ::sub],
              speed[::sub, ::sub], cmap="viridis", alpha=.9)
    a1.set_aspect("equal"); a1.legend(fontsize=7, loc="upper right")
    a1.set_title(title or getattr(field, "label", ""), fontsize=10)

    vmax = np.abs(curl).max() + 1e-12
    im = a2.pcolormesh(xx, yy, curl, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    a2.scatter(*p.X.T, s=5, alpha=.2, color="k")
    a2.set_aspect("equal")
    a2.set_title(rf"curl  (rot. fraction $={rot:.3f}$)", fontsize=10)
    plt.colorbar(im, ax=a2, fraction=0.046)
    return rot


def plot_field_gallery(fields: Dict[str, object], p: Emp, q: Emp, extent: float,
                       n_grid: int = 32, figsize=None):
    """One (quiver, curl) row per field; returns {name: rotational fraction}."""
    fig, axes = plt.subplots(len(fields), 2,
                             figsize=figsize or (8.4, 2.4 * len(fields)),
                             squeeze=False)
    rot = {}
    for row, (name, fld) in zip(axes, fields.items()):
        rot[name] = plot_field(fld, p, q, extent, n_grid, title=name, axes=row)
    fig.tight_layout()
    return fig, rot


# --------------------------------------------------------------------------- #
# Run curves
# --------------------------------------------------------------------------- #
_LABELS = {
    "W2": r"$W_2(q^k,p)$", "F": r"$\frac{1}{2}\mathrm{MMD}^2_{k^\#}(q^k,p)$",
    "Vnorm": r"$\|V\|_{L^2(q^k)}$", "H": r"$\mathrm{KL}(q_\sigma\|p_\sigma)$",
    "n_covered": "modes covered", "entropy": "coverage entropy",
    "precision": "precision", "mass_err": r"$\|m(q)-w_p\|_1$",
    "MMD2": r"$\mathrm{MMD}^2_k$", "beta_F": r"$\beta_{\mathcal{F}}$",
    "beta_star": r"$\beta_\star$", "rho_NE": r"$\rho_{NE}$",
    "rho_diss": r"$\rho_{\rm diss}$", "alpha_hat": r"$\hat\alpha$",
    "ot_alignment": r"$-\mathfrak{D}/(W_2\,\Vert V\Vert)$",
    "L_hat": r"$\hat L$", "rho_bound": r"$\rho_\tau$ bound",
    "tau_opt": r"$\tau_{\rm opt}=\hat\alpha/\hat L^2$",
}
_LOG = {"W2", "F", "Vnorm", "H", "MMD2"}


def plot_runs(runs: Dict[str, object], keys: Sequence[str] = ("W2", "F", "n_covered", "entropy"),
              ncol: int = 2, figsize=None, hlines: Optional[Dict[str, float]] = None):
    """Grid of probe curves, one panel per key, one line per run."""
    keys = [k for k in keys if any(k in r.probes for r in runs.values())]
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (4.5 * ncol, 3.1 * nrow),
                             squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        for name, r in runs.items():
            if key in r.probes:
                ax.plot(r.probe_k, r.probes[key], color=color_of(name), lw=1.7, label=r.label)
        if key in _LOG:
            vals = np.concatenate([r.probes[key] for r in runs.values() if key in r.probes])
            if np.nanmin(vals) > 0:
                ax.set_yscale("log")
        if hlines and key in hlines:
            ax.axhline(hlines[key], color="0.4", ls="--", lw=1)
        ax.set(xlabel="iteration $k$", ylabel=_LABELS.get(key, key))
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_schedules(runs: Dict[str, object]):
    """The schedule s_k actually used by each run (sanity check on annealing)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for name, r in runs.items():
        if len(r.schedule) and np.isfinite(r.schedule).any():
            ax.plot(r.schedule, color=color_of(name), lw=1.6, label=r.label)
    ax.set(xlabel="iteration $k$", ylabel=r"schedule $s_k$", yscale="log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_final_bars(runs: Dict[str, object], keys=("W2", "H"), higher_better=()):
    """Final-value bar chart; the report's one-glance comparison."""
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.2), squeeze=False)
    names = list(runs)
    for ax, key in zip(axes.ravel(), keys):
        vals = [runs[n].final(key) for n in names]
        ax.bar(range(len(names)), vals, color=[color_of(n) for n in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([runs[n].label for n in names], rotation=30, ha="right", fontsize=7)
        arrow = "higher better" if key in higher_better else "lower better"
        ax.set(ylabel=_LABELS.get(key, key), title=f"final {_LABELS.get(key, key)} ({arrow})")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Snapshots: the report-friendly alternative to an animation
# --------------------------------------------------------------------------- #
def plot_snapshots(runs: Dict[str, object], p: Emp, ks: Sequence[int],
                   extent: Optional[float] = None, centers=None, r_cover: float = 1.4,
                   color_by_angle: bool = True, panel_size=(1.8, 1.7)):
    """Rows = runs, columns = iterations. Particles coloured by initial angle,
    so one can *see* whether the map stays a map or folds."""
    names = list(runs)
    fig, axes = plt.subplots(len(names), len(ks),
                             figsize=(panel_size[0] * len(ks), panel_size[1] * len(names)),
                             squeeze=False)
    X0 = runs[names[0]].X[0]
    if color_by_angle:
        ang = np.arctan2(X0[:, 1] - X0[:, 1].mean(), X0[:, 0] - X0[:, 0].mean())
        ckw = dict(c=(ang + np.pi) / (2 * np.pi), cmap="hsv", vmin=0, vmax=1)
    else:
        ckw = dict(color="#d62728")
    if extent is None:
        allpts = np.vstack([p.X] + [r.X[-1] for r in runs.values()])
        extent = float(np.abs(allpts).max() * 1.1)

    for i, name in enumerate(names):
        r = runs[name]
        for j, k in enumerate(ks):
            ax = axes[i, j]
            frame = min(k // r.store_every, len(r.X) - 1)
            ax.scatter(*p.X.T, s=5, alpha=.18, color="0.4")
            if centers is not None:
                for c in centers:
                    ax.add_patch(plt.Circle(c, r_cover, fill=False, ls=":", color="gray", lw=.8))
            ax.scatter(*r.X[frame].T, s=7, alpha=.75, **ckw)
            ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(f"$k={k}$", fontsize=9)
            if j == 0:
                ax.set_ylabel(r.label, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Dedicated diagnostic panels
# --------------------------------------------------------------------------- #
def plot_certificates(runs: Dict[str, object], tau: float):
    """Target-distance ratio, dissipativity and star-cocoercivity.

    Reading: rho_NE <= 1 means the step did not move away from p; rho_diss < 0
    means the drift is aligned with the OT displacement; beta_star >= tau/2 is
    the Krasnoselskii-Mann condition that makes the iteration Fejer-monotone.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for name, r in runs.items():
        c = color_of(name)
        for ax, key in zip(axes, ("rho_NE", "rho_diss", "beta_star")):
            if key in r.probes:
                ax.plot(r.probe_k, r.probes[key], color=c, lw=1.6, label=r.label)
    axes[0].axhline(1, color="k", ls=":", lw=1)
    axes[0].set(xlabel="$k$", ylabel=_LABELS["rho_NE"],
                title=r"target-distance / Fejér ratio ($\leq 1$)")
    axes[1].axhline(0, color="k", ls=":", lw=1)
    axes[1].set(xlabel="$k$", ylabel=_LABELS["rho_diss"], title=r"dissipativity ($<0$)")
    axes[2].axhline(tau / 2, color="#d62728", ls="--", lw=1.2, label=r"KM: $\tau/2$")
    axes[2].axhline(0, color="k", ls=":", lw=1)
    axes[2].set(xlabel="$k$", ylabel=_LABELS["beta_star"],
                title=r"star-cocoercivity ($\geq\tau/2$)")
    axes[0].legend(fontsize=7); axes[2].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_f_dissipativity(runs: Dict[str, object], tau: float):
    """The quad / "mechant" split of -D^{V,F} and the realized F-cocoercivity.

    The dashed curve is the normalization term.  The horizontal tau/2 line is
    retained only as a visual scale shared with beta_star; crossing it is not a
    finite-step certificate without a separate smoothness bound for F.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for name, r in runs.items():
        if "quad" not in r.probes:
            continue
        c = color_of(name)
        a.plot(r.probe_k, r.probes["quad"] + r.probes["mechant"], color=c, lw=2, label=r.label)
        a.plot(r.probe_k, r.probes["quad"], color=c, ls=":", lw=1.3)
        a.plot(r.probe_k, r.probes["mechant"], color=c, ls="--", lw=1.3)
        b.plot(r.probe_k, r.probes["beta_F"], color=c, lw=1.7, label=r.label)
    a.axhline(0, color="k", lw=.8)
    a.set(xlabel="$k$", title=r"$-\mathcal{D}^{V,\mathcal{F}}$ (—), quad ($\cdots$), méchant (- -)")
    b.axhline(0, color="k", lw=.8)
    b.axhline(tau / 2, color="#d62728", ls="--", lw=1.2,
              label=r"reference only: $\tau/2$")
    b.set(xlabel="$k$", ylabel=r"$\beta_{\mathcal{F}}$",
          title=r"first-order $\mathcal{F}$ alignment ratio")
    a.legend(fontsize=7); b.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_beta_vs_sigma(sigmas, curves: Dict[str, np.ndarray], tau: Optional[float] = None,
                       title: str = ""):
    """beta_star(sigma) at a frozen state: where does cocoercivity break?"""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for label, beta in curves.items():
        ax.semilogx(sigmas, beta, color=color_of(label), lw=1.7, marker=".", ms=4, label=label)
    ax.axhline(0, color="k", lw=.8)
    if tau is not None:
        ax.axhline(tau / 2, color="#d62728", ls="--", lw=1.2, label=r"KM threshold $\tau/2$")
    ax.set(xlabel=r"bandwidth $\sigma$", ylabel=r"$\beta_\star=-\mathcal{D}/\|V\|^2_{L^2(q)}$",
           title=title or "star-cocoercivity vs bandwidth")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_two_dirac_counterexample(R: float = 3.0, sigma: float = 1.0,
                                  kernel_name: str = "Gaussian"):
    """The 1-D counterexample: kernel locality beats the OT direction.

    Left: the field, with the mean-shift and OT directions at the two atoms of
    q.  Right: D(R/sigma) crossing into the positive region, against the
    predicted threshold sqrt(log 15) and the lower bound R^2/8.
    """
    from .drifts import mean_shift
    from .kernels import get_kernel
    from .metrics import two_dirac_dissipativity

    p_at, q_at = [-R, R], [R / 2.0, 3.0 * R / 4.0]
    kern = get_kernel(kernel_name, sigma)
    p, q = Emp(np.array(p_at)[:, None]), Emp(np.array(q_at)[:, None])
    fld = mean_shift(kern)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    xs = np.linspace(-1.5 * R, 1.5 * R, 600)[:, None]
    a1.plot(xs[:, 0], fld.at(xs, p, q)[:, 0], color="#2ca02c", lw=1.8,
            label=r"$V^{MS}[p_R,q_R](x)$")
    a1.axhline(0, color="0.6", lw=.8)
    a1.plot(p_at, [0, 0], "o", ms=9, color="#1f77b4", label=r"$\mathrm{supp}(p_R)=\{\pm R\}$")
    a1.plot(q_at, [0, 0], "s", ms=8, color="#d62728", label=r"$\mathrm{supp}(q_R)$")
    Vq = fld(p, q)[:, 0]
    for i, (xq, tgt) in enumerate(zip(q_at, [-R, R])):     # monotone coupling in 1-D
        y0 = 0.55 + 0.55 * i
        kw = dict(width=.05, head_width=.16, head_length=.18, length_includes_head=True)
        a1.arrow(xq, y0, .9 * np.sign(Vq[i]), 0, color="#2ca02c", **kw)
        a1.arrow(xq, -y0, .9 * np.sign(tgt - xq), 0, color="#9467bd", **kw)
    a1.plot([], [], color="#2ca02c", lw=3, label="mean-shift direction")
    a1.plot([], [], color="#9467bd", lw=3, label=r"OT direction $T(x)-x$")
    a1.set(xlabel="$x$", ylabel=r"$V^{MS}(x)$", ylim=(-3, 3.2),
           title=rf"(a) field vs coupling  ($R={R:g}$, $\sigma={sigma:g}$)")
    a1.legend(fontsize=7.5, loc="upper left")

    Rs = np.linspace(0.05, 4.0 * sigma, 220)
    Ds = np.array([two_dirac_dissipativity(r, sigma, kernel_name)["D"] for r in Rs])
    thr = sigma * np.sqrt(np.log(15.0))
    a2.plot(Rs / sigma, Ds, color="k", lw=1.8, label=r"$\mathcal{D}_{p_R}^{V^{MS}}(q_R)$")
    a2.fill_between(Rs / sigma, Ds, 0, where=Ds > 0, color="#d62728", alpha=.18,
                    label=r"dissipativity violated ($\mathcal{D}>0$)")
    a2.plot(Rs / sigma, Rs ** 2 / 8, ls="--", color="0.45", lw=1.2,
            label=r"bound $R^2/8$ (past threshold)")
    a2.axvline(thr / sigma, color="#1f77b4", ls=":", lw=1.5)
    a2.axhline(0, color="0.6", lw=.8)
    a2.scatter([R / sigma], [two_dirac_dissipativity(R, sigma, kernel_name)["D"]],
               color="#d62728", zorder=5, label="configuration of (a)")
    a2.set(xlabel=r"$R/\sigma$", ylabel=r"$\mathcal{D}_{p_R}^{V^{MS}}(q_R)$",
           title=r"(b) kernel locality breaks dissipativity")
    a2.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------- #
def animate_particles(runs: Dict[str, object], p: Emp, centers=None, every: int = 5,
                      interval: int = 60, r_cover: float = 1.4, extent=None,
                      every_iter: Optional[int] = None, panel_size: float = 3.4):
    """Side-by-side particle animation.

    ``every`` is a stride in stored frames for backwards compatibility.
    ``every_iter`` is clearer for notebooks and is converted using each run's
    ``store_every``.  Use :func:`animation_html` for a responsive display.
    """
    names = list(runs)
    n_frames = min(len(runs[n].X) for n in names)
    if every_iter is not None:
        every = max(1, int(round(every_iter / runs[names[0]].store_every)))
    frames = list(range(0, n_frames, every))
    if extent is None:
        allpts = np.vstack([p.X] + [r.X[-1] for r in runs.values()])
        extent = float(np.abs(allpts).max() * 1.1)
    ncol = min(len(names), 3)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(panel_size * ncol, panel_size * nrow), squeeze=False)
    X0 = runs[names[0]].X[0]
    ang = np.arctan2(X0[:, 1] - X0[:, 1].mean(), X0[:, 0] - X0[:, 0].mean())
    ckw = dict(c=(ang + np.pi) / (2 * np.pi), cmap="hsv", vmin=0, vmax=1)

    scat = {}
    for ax, name in zip(axes.ravel(), names):
        ax.scatter(*p.X.T, s=7, alpha=.18, color="0.4")
        if centers is not None:
            for c in centers:
                ax.add_patch(plt.Circle(c, r_cover, fill=False, ls=":", color="gray", lw=.8))
        scat[name] = ax.scatter(*X0.T, s=10, alpha=.75, **ckw)
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_aspect("equal"); ax.set_title(runs[name].label, fontsize=9)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    sup = fig.suptitle("k = 0")

    def update(i):
        for name in names:
            scat[name].set_offsets(runs[name].q_at_frame(i).X)
        sup.set_text(f"k = {runs[names[0]].iteration_of_frame(i)}")
        return list(scat.values()) + [sup]

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)
    return anim


def animate_field(run_obj, p: Emp, field_of_k: Callable, extent: float = 8.0,
                  n_grid: int = 22, every: int = 10, interval: int = 80,
                  title: str = "", every_iter: Optional[int] = None,
                  figsize=(4.8, 4.8)):
    """Particles plus the (possibly time-varying) field as unit-length arrows.

    `field_of_k(k) -> DriftField` takes the *iteration* k, so a bandwidth
    schedule becomes visible: the arrows change reach as sigma_k cools.
    """
    g, xx, yy, pts = field_grid(extent, n_grid)
    if every_iter is not None:
        every = max(1, int(round(every_iter / run_obj.store_every)))
    frames = list(range(0, len(run_obj.X), every))
    field_at_frame = lambda i: field_of_k(run_obj.iteration_of_frame(i))
    speeds = [np.linalg.norm(field_at_frame(i).at(pts, p, run_obj.q_at_frame(i)), axis=1)
              for i in frames[:: max(1, len(frames) // 5)]]
    clim = (0.0, float(np.percentile(np.concatenate(speeds), 95)) + 1e-9)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(*p.X.T, s=10, alpha=.2, color="#1f77b4", label="$p$")
    sc = ax.scatter(*run_obj.X[0].T, s=11, alpha=.85, color="#d62728", label="$q^k$")
    V0 = field_at_frame(0).at(pts, p, run_obj.q_at_frame(0))
    s0 = np.linalg.norm(V0, axis=1) + 1e-12
    quiv = ax.quiver(pts[:, 0], pts[:, 1], V0[:, 0] / s0, V0[:, 1] / s0, s0, cmap="viridis",
                     alpha=.8, pivot="mid", scale=n_grid * 1.1, width=.003, clim=clim)
    fig.colorbar(quiv, ax=ax, fraction=.046).set_label(r"$\|V\|$")
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent); ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ttl = ax.set_title(f"{title}\nk = 0", fontsize=10)

    def update(i):
        k = run_obj.iteration_of_frame(i)
        sc.set_offsets(run_obj.X[i])
        V = field_at_frame(i).at(pts, p, run_obj.q_at_frame(i))
        s = np.linalg.norm(V, axis=1) + 1e-12
        quiv.set_UVC(V[:, 0] / s, V[:, 1] / s, s)
        extra = ""
        if len(run_obj.schedule):
            sk = run_obj.schedule[min(k, len(run_obj.schedule) - 1)]
            extra = f",  $s_k$={sk:.2f}" if np.isfinite(sk) else ""
        ttl.set_text(f"{title}\nk = {k}{extra}")
        return sc, quiv, ttl

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)
    return anim


def animation_html(anim, max_width: int = 520):
    """Responsive notebook wrapper for a Matplotlib JS animation."""
    from IPython.display import HTML

    html = anim.to_jshtml()
    return HTML(
        f'<div class="msflow-animation" style="max-width:{int(max_width)}px;width:100%;">'
        f'{html}</div><style>.msflow-animation img,.msflow-animation canvas,'
        '.msflow-animation video{max-width:100%!important;height:auto!important;}'
        '.msflow-animation{overflow-x:auto;}</style>'
    )
