# RESS Earth-Moon logistics reproducibility package

This public repository accompanies the V15 manuscript *Reliability-informed
robust optimisation of Earth-Moon infrastructure logistics under launch and
space-elevator disruptions*. It contains the canonical deterministic and
robust MILPs, the corrected Morris screening, and the reproducible LHS-MILP
partial rank correlation coefficient (PRCC) analyses used in the manuscript.

## Installation and full validation

Python 3.12 is the verified environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```

`run_all.py` is fail-closed: any failed numerical or artifact check produces a
nonzero exit code. A successful run ends with:

```text
ALL V15 CHECKS PASSED
```

The full workflow regenerates the deterministic frontiers, the robust
frontier and feasibility scan, Figures 8-11, both Morris datasets, and both
PRCC analyses.

## PRCC reproduction

Run the manuscript Figure 11 analysis at the deterministic knee horizon:

```bash
python src/run_prcc.py
```

This writes:

- `data/prcc_lhs_milp_samples.csv`
- `data/prcc_results.csv`
- `figures/Figure11_PRCC_corrected.png`
- `logs/prcc_run_log.txt`

Run the supplementary T=156 validation without overwriting the T=137 files:

```bash
python src/run_prcc.py --horizon 156
```

This writes:

- `data/prcc_lhs_milp_samples_T156.csv`
- `data/prcc_results_T156.csv`
- `figures/Figure11_PRCC_corrected_T156.png`
- `logs/prcc_run_log_T156.txt`

Both runs use a seven-dimensional Latin hypercube with 256 samples, seed 2026,
and +/-20% ranges around the baseline parameters. Every sample is solved by
the canonical `src/task1_q1_milp.py`; there is no second MILP implementation.
The common `t_mid` input moves the propulsion, elevator-capacity, and
launch-capacity maturation midpoints together.

PRCC is computed by rank-transforming the seven inputs and objective,
residualizing each ranked input and the ranked output against the remaining
six ranked inputs with an intercept, and correlating the two residual vectors.
Two-sided p-values use the partial-correlation t statistic. At T=137, 251 of
256 samples are feasible and `df = 251 - 7 - 1 = 243`. At T=156 all 256
samples are feasible and `df = 256 - 7 - 1 = 248`.

Key deposited coefficients are:

| Horizon | `c_R_base` | `gamma` | `t_mid` |
|---|---:|---:|---:|
| T=137 | +0.9375 | -0.9031 | +0.8225 |
| T=156 | +0.9790 | -0.9686 | +0.8314 |

The sample CSVs retain solver status, solver code, solver message, objective,
scaled objective, infeasible/excluded flags, and an exclusion reason so every
row is auditable.

## Figure 10 Morris screening

`src/fig10_morris_screening.py` is the authoritative Figure 10 generator:

```bash
python src/fig10_morris_screening.py
```

It evaluates deterministic MILP cost at fixed T=137 over the same seven
parameters and +/-20% design box, using seed 2026, 40 trajectories, six grid
levels, and the realized step after boundary reflection. Infeasible endpoint
pairs are retained in `data/morris_elementary_effects.csv` with a documented
reason and excluded from the summary. The supported interpretation is that
nominal rocket cost and cost elasticity are the dominant, approximately
monotonic pair; the technology midpoint is third.

## Model and artifact provenance

| Manuscript result | Generating code | Output/evidence |
|---|---|---|
| Deterministic NH=1/NH=3 frontiers | `src/task1_q1_milp.py` | `data/pareto_nh1.csv`, `data/pareto_nh3.csv` |
| Reliability-adjusted robust frontier | `src/task2b_robust_milp.py` | `outputs/pareto_robust.csv` |
| Robust feasibility boundary | `src/task2b_robust_milp.py --scan` | `logs/robust_feasibility_scan.csv` |
| Figures 8-9 | `src/task2b_robust_milp.py`, `src/fig9_frontiers.py` | corresponding PNGs |
| Figure 10 | `src/fig10_morris_screening.py` | two Morris CSVs and corrected PNG |
| Figure 11 | `src/run_prcc.py` | T=137 samples/results and corrected PNG |
| T=156 supplementary PRCC | `src/run_prcc.py --horizon 156` | suffixed samples/results and PNG |

Headline regression values checked by `run_all.py` include the deterministic
137-year knee at $210.71T with 48.8% elevator share and the robust 156-year
knee at $236.65T with 50.1% elevator share and a 12.3% premium.

## Canonical versus legacy sensitivity code

The old five-parameter functions in `task1_q1_milp.py` are retained only for
backward API compatibility. They use simplified diagnostics and are explicitly
marked `LEGACY ONLY`; they are not called by `run_all.py`, do not generate the
publication figures, and must not be used to reproduce manuscript claims. The
obsolete `fig5a_morris_screening.png` and `fig5b_prcc.png` artifacts have been
removed.

## Rocket costing assumption

Rocket transport is costed per tonne of delivered payload; partially filled
launches incur no unused-capacity penalty. The integer launch variable enforces
payload capacity and the annual launch-slot ceiling, but has zero direct cost
coefficient. This is the intended manuscript assumption, not a per-attempt
cost model.

## Repository hygiene

- All scripts use repository-relative paths.
- Sampling seeds and parameter ranges are explicit.
- Generated logs contain environment versions and relative output paths only.
- No upload tokens, personal machine paths, or obsolete verification logs are
  part of the publication package.
- Regression tests run with `python -m unittest discover -s tests -v`.
