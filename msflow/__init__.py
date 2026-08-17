"""
Modules
-------
    measures    Emp, discrete W_2 / selected OT plans, the seeded dataset zoo
    kernels     radial kernels and their sharp companions k#
    drifts      population row field, Deng Algorithm 2, Proxy, MMD, OT, ...
    sinkhorn    K-step Sinkhorn drifting: K = 0 is the population row field,
                K = inf is exact balancing
    gaussian    the exactly solvable Gaussian sector (moment ODE, Gamma_K, S_eps)
    dynamics    the iteration engine, schedules, probes, Run/Runs containers
    metrics     dissipativity / cocoercivity / energy / coverage diagnostics
    parametric  transport-projection with a parameterized map (Euclid vs natural)
    viz         figures and animations
    report      report-ready tables (CSV + LaTeX)
    checks      the math audit: identities from the notes, verified numerically

Quick start
-----------
    from msflow import *
    data = load_dataset("ring", n=320)
    field = make_drift("mean_shift", "Laplacian", sigma=1.0)
    r = run(data.q0, data.p, field, n_iter=600, tau=0.04,
            probes=standard_probes(data, field_kernel(field)))
    plot_runs({"deng": r})
"""
from . import gaussian, metrics, parametric, report, sinkhorn, viz

from .measures import (
    Emp, Dataset, W2, OT_plan, barycentric_map, load_dataset, DATASETS,
)
from .kernels import (
    Kernel, KERNELS, get_kernel, kde, sharp_kde, mean_shift_numerator,
    softmax_barycentre, smoothed_score, plot_kernel_profiles,
)
from .drifts import (
    DriftField, make_drift,
    mean_shift, deng_population, deng_official, deng_official_batch,
    deng_official_batch_details, DengOfficialBatchResult,
    sinkhorn_proxy, sinkhorn_proxy_details, SinkhornProxyResult,
    mean_shift_unnormalized, mmd, sharp_core, ot, linear_potential,
    pgr_family, power_family, common_normalization, density_floor, NORMALIZERS,
    alpha_mean_shift, alpha_shift, flow_matching, score_regularized,
    gaussian_score, norm_matched, decompose,
)
from .sinkhorn import (
    sinkhorn_drift, sinkhorn_barycentre, sinkhorn_divergence, epsilon_of_sigma,
)
from .dynamics import (
    Run, Runs, run, standard_probes, certificate_probes, replay, save_run, load_run,
    constant, exp_anneal, linear_anneal, cosine_anneal, poly_anneal,
    plateau_then_exp, fm_time, SCHEDULES,
)
from .metrics import (
    MMD2, sharp_energy, dissipativity, drift_norm, step_diagnostics,
    beta_vs_sigma, two_dirac_dissipativity, f_dissipativity, gauge_fraction,
    rotational_fraction, mode_metrics, mass_split_error, smoothed_kl,
    fit_exponential_rate, alpha_operator, gaussian_growth_constant,
)
from .viz import (
    plot_dataset, plot_dataset_grid, plot_field, plot_field_gallery, plot_runs,
    plot_schedules, plot_final_bars, plot_snapshots, plot_certificates,
    plot_f_dissipativity, plot_beta_vs_sigma, plot_two_dirac_counterexample,
    animate_particles, animate_field, animation_html, color_of, set_colors, save,
)
from .report import runs_table, save_table, style, checks_table, pct_above, pct_below

__version__ = "1.0"


def field_kernel(field: DriftField):
    name, sigma = field.meta.get("kernel"), field.meta.get("sigma")
    return get_kernel(name, sigma) if name and sigma else None


__all__ = [n for n in dir() if not n.startswith("_")]
