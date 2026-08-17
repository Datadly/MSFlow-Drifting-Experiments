# MSFlow: finite-particle drift experiments

This repository studies particle updates of the form

```math
q^{k+1} = (\mathrm{Id} + \tau V_{p,q^k})_\# q^k
```

on small empirical measures, mainly in two dimensions. It compares several
kernel, batch-normalized, optimal-transport, and Sinkhorn-based drift fields.
The notebooks contain the experiments; the `msflow` package contains the
reusable implementations.

## Setup

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands and launch notebooks from the repository root so that local
imports and output paths resolve correctly. In VS Code, select the Python
interpreter inside `.venv`. For a browser-based interface, install and start
JupyterLab with:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

## Verification

Before running experiments, check the installation:

```bash
python -m pytest tests -q
python -m msflow.checks
```


## Experiments

| Notebook | Experiment |
|---|---|
| `notebooks/01_lab.ipynb` | Dataset, kernel, drift, and bandwidth overview |
| `notebooks/02_field_geometry.ipynb` | Normalization, geometry, curl, projection, and P/G/R diagnostics |
| `notebooks/03_convergence_certificates.ipynb` | Convergence diagnostics, counterexamples, schedules, and diffusion regularization |
| `notebooks/04_gaussian_sinkhorn.ipynb` | Closed-form Gaussian calculations and the Sinkhorn-$K$ family |
| `notebooks/05_parametrization.ipynb` | Euclidean and natural transport projection for parameterized maps |


## Scientific scope

The main drift families are not interchangeable:

- `mean_shift` (also exposed as `deng_population`) is the row-normalized
  population mean-shift field.
- `deng_official` is a literal NumPy implementation of the transport-only
  batch field in Algorithm 2 of Deng et al.
- `sinkhorn_proxy` implements the separately normalized positive and negative
  blocks of the Sinkhorn Proxy algorithm, with $\tau = 2\sigma^2$ for the
  Gaussian kernel convention used here.
- `sinkhorn_drift(K)` interpolates between the row-normalized population field
  at $K=0$ and balanced Sinkhorn transport as $K$ increases.

## Repository layout

- `msflow/`: measures, kernels, drift fields, dynamics, metrics, Gaussian
  formulas, parameterized projections, plots, and report helpers.
- `notebooks/`: the five executable experiments.
- `tests/`: regression and mathematical consistency tests.
- `figures/` and `tables/`: generated experiment artifacts.
- `requirements.txt`: direct runtime and notebook dependencies.

## Primary references

- [Deng et al., Algorithm 2, arXiv:2602.04770v2](https://arxiv.org/html/2602.04770v2)
- [Gretton et al., Sinkhorn Proxy, arXiv:2605.05118v2](https://arxiv.org/html/2605.05118v2)
- [Upstream drifting implementation](https://github.com/lambertae/drifting)
