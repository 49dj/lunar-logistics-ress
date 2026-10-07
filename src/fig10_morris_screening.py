#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig10_morris_screening.py — Manuscript Figure 10: Morris elementary-effects
screening of knee-point cost over the seven economic-technological parameters.

This is the authoritative generator for Figure 10 and its underlying data. The
earlier repository shipped only a simplified five-parameter screening embedded in
task1_q1_milp.py (morris_screening); the published Figure 10 uses the full
seven-parameter set below, matching the local-elasticity tornado (Figure 12) and
the "seven economic-technological parameters" described in Section 7.2.

Method: standard Morris (1991) trajectory design. Each of the seven inputs is
scaled to [0, 1] over a ±20% box around its baseline value, discretised to
p = 6 levels with step Delta = p/(2(p-1)) = 0.6. r trajectories are sampled; the
elementary effect of input i on a trajectory is

    EE_i = [ J*(x with x_i stepped) - J*(x) ] / Delta        (normalised units)

where J*(x) is the deterministic knee-horizon cost (MILP solved at T = 137, the
Section-4.4 knee). The screening statistics are

    mu*_i = mean_j | EE_i^(j) |     (overall influence)
    sigma_i = std_j  EE_i^(j)       (nonlinearity / interaction)

Outputs (paths relative to the repository root):
  data/morris_screening.csv                 summary statistics and ranges
  data/morris_elementary_effects.csv        one auditable row per effect
  figures/Figure10_Morris_screening_corrected.png

Run (from the repository root):
  python src/fig10_morris_screening.py

Deterministic: fixed RNG seed (MORRIS_SEED). Runtime ~10-30 s (r x 8 MILP solves).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_q1_milp as t1

ROOT = Path(__file__).resolve().parent.parent
OUT_DATA = ROOT / "data"
OUT_FIG = ROOT / "figures"
for _d in (OUT_DATA, OUT_FIG):
    _d.mkdir(parents=True, exist_ok=True)

MORRIS_SEED = 2026
T_KNEE = 137          # Section-4.4 deterministic knee horizon
N_TRAJ = 40           # Morris trajectories
P_LEVELS = 6          # grid levels
DELTA = P_LEVELS / (2 * (P_LEVELS - 1))   # = 0.6
REL = 0.20            # +/-20% box around each baseline value

# Seven economic-technological parameters (baseline, ±20% box, display label).
# 't_mid' shifts all three technology-maturation sigmoid midpoints together
# (specific impulse, elevator capacity, launch-slot ceiling).
PARAMS = [
    ("gamma",   1.2,     r"Cost elasticity $\gamma$"),
    ("c_R_base", 6.5e6,  r"Nominal rocket cost $c_R^{\mathrm{base}}$"),
    ("P_R",     138.0,   r"Rocket payload $P_R$"),
    ("N_max",   5000.0,  r"Max launches $N_{\max}$"),
    ("k_growth", 0.12,   r"Growth rate $k$"),
    ("t_mid",   50.0,    r"Technology midpoint $t_{\mathrm{mid}}$"),
    ("C_fixed", 100e9,   r"Elevator fixed cost $C_{\mathrm{fixed}}$"),
]
KEYS = [p[0] for p in PARAMS]
BASE = np.array([p[1] for p in PARAMS], float)
LOW = BASE * (1 - REL)
HIGH = BASE * (1 + REL)


def _apply(unit_vec):
    """Map a point in the [0,1]^7 design cube to a params object and solve."""
    real = LOW + unit_vec * (HIGH - LOW)
    p = t1.clone_params(t1.P)
    p.GAMMA = real[0]
    p.C_R_BASE = real[1]
    p.P_R = real[2]
    p.N_LAUNCH_MAX = real[3]
    p.GROWTH_K = real[4]
    p.T_MID_E = p.T_MID_R = p.T_MID_ISP = real[5]
    p.FIXED_COST = real[6]
    r = t1.solve_milp(T_KNEE, p)
    return None if r is None else r["total_cost"]


def _trajectory(rng, k):
    """One Morris trajectory: (k+1) points, each differing from the previous in
    exactly one coordinate by +/-Delta. Returns the (k+1, k) design matrix."""
    levels = np.linspace(0.0, 1.0 - DELTA, P_LEVELS - int(round(DELTA * (P_LEVELS - 1))))
    base = rng.choice(levels, size=k)
    order = rng.permutation(k)
    signs = rng.choice([-1.0, 1.0], size=k)
    pts = [base.copy()]
    cur = base.copy()
    for idx in order:
        step = signs[idx] * DELTA
        if cur[idx] + step > 1.0 or cur[idx] + step < 0.0:
            step = -step
        cur = cur.copy()
        cur[idx] += step
        pts.append(cur.copy())
    return np.array(pts), order, signs


def run_morris(seed=MORRIS_SEED, n_traj=N_TRAJ, verbose=True):
    rng = np.random.default_rng(seed)
    k = len(KEYS)
    ee = {key: [] for key in KEYS}
    raw_rows = []
    used = skipped = 0
    for trajectory_id in range(n_traj):
        design, order, signs = _trajectory(rng, k)
        ys = [_apply(pt) for pt in design]
        for step_i, idx in enumerate(order):
            y0, y1 = ys[step_i], ys[step_i + 1]
            realized_step = design[step_i + 1][idx] - design[step_i][idx]
            feasible_y0 = y0 is not None and np.isfinite(y0)
            feasible_y1 = y1 is not None and np.isfinite(y1)
            skipped_reason = ""
            eff = np.nan
            if y0 is None or y1 is None:
                skipped += 1
                skipped_reason = "infeasible_y0" if y0 is None else "infeasible_y1"
                if y0 is None and y1 is None:
                    skipped_reason = "infeasible_y0_and_y1"
            elif not np.isfinite(realized_step) or realized_step == 0:
                skipped += 1
                skipped_reason = "zero_or_nonfinite_realized_step"
            else:
                # FIX (2026-07): divide by the REALIZED step. The trajectory may
                # reflect an attempted step at the cube boundary, so the originally
                # sampled sign is not a reliable denominator.
                eff = (y1 - y0) / realized_step
                ee[KEYS[idx]].append(eff)
                used += 1

            real0 = LOW[idx] + design[step_i][idx] * (HIGH[idx] - LOW[idx])
            real1 = LOW[idx] + design[step_i + 1][idx] * (HIGH[idx] - LOW[idx])
            raw_rows.append({
                "trajectory_id": trajectory_id,
                "parameter": KEYS[idx],
                "x0": float(design[step_i][idx]),
                "x1": float(design[step_i + 1][idx]),
                "actual_x0": float(real0),
                "actual_x1": float(real1),
                "y0": np.nan if y0 is None else float(y0),
                "y1": np.nan if y1 is None else float(y1),
                "realized_step": float(realized_step),
                "elementary_effect": eff,
                "feasible_y0": bool(feasible_y0),
                "feasible_y1": bool(feasible_y1),
                "skipped_reason": skipped_reason,
            })
    rows = []
    for key, _, label in PARAMS:
        arr = np.array(ee[key], float)
        param_i = KEYS.index(key)
        mu_star = float(np.abs(arr).mean()) if arr.size else np.nan
        sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        rows.append(dict(param=key, label=label,
                         baseline=float(BASE[param_i]),
                         lower=float(LOW[param_i]),
                         upper=float(HIGH[param_i]),
                         mu=float(arr.mean()) if arr.size else np.nan,
                         mu_star=mu_star,
                         sigma=sigma,
                         sigma_over_mu_star=(sigma / mu_star) if mu_star else np.nan,
                         n_eff=int(arr.size)))
    df = pd.DataFrame(rows).sort_values("mu_star", ascending=False).reset_index(drop=True)
    raw_df = pd.DataFrame(raw_rows)
    if verbose:
        print(f"Morris screening: {n_traj} trajectories, "
              f"{used} elementary effects ({skipped} skipped for infeasibility)")
        print(df[["param", "mu_star", "sigma", "n_eff"]].to_string(index=False))
        top = df.iloc[0]
        second = df.iloc[1]
        print(f"\n  dominant parameter: {top['param']}  "
              f"mu*={top['mu_star']:.3e}  sigma={top['sigma']:.3e}  "
              f"sigma/mu*={top['sigma']/top['mu_star']:.3f}")
        print(f"  mu* ratio (dominant / 2nd): {top['mu_star']/second['mu_star']:.1f}x")
        region = "monotonic / near-linear" if top['sigma'] < 0.5 * top['mu_star'] \
            else "nonlinear / interacting"
        print(f"  dominant parameter sits in the {region} region "
              f"(sigma {'<' if top['sigma'] < 0.5*top['mu_star'] else '>='} 0.5 mu*)")
    return df, raw_df


def make_figure(df, out_png=None):
    """Data-faithful Morris plot: every parameter at its computed (mu*, sigma),
    the two standard diagnostic lines (sigma = mu*, sigma = 0.5 mu*), and
    leader-lined labels. The two rocket-cost parameters (c_R^base, gamma) are
    highlighted. No point positions are hand-set; all come from the CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(t1.STYLE)
    CB = t1.CB
    out_png = Path(out_png) if out_png else OUT_FIG / "Figure10_Morris_screening_corrected.png"

    scale = 1e13
    df = df.reset_index(drop=True)
    mu_s = df["mu_star"].values / scale
    sg_s = df["sigma"].values / scale
    labels = df["label"].values
    keys = df["param"].values

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    xmax = mu_s.max() * 1.30
    ymax = sg_s.max() * 1.18
    xs = np.linspace(0, xmax, 50)
    ax.plot(xs, xs, ls="--", color="0.6", lw=1.2, zorder=1)
    ax.plot(xs, 0.5 * xs, ls="-.", color="0.65", lw=1.1, zorder=1)
    ax.text(xmax * 0.055, ymax * 0.90,
            r"$\sigma=\mu^*$ (non-linear / interacting)", color="0.5",
            fontsize=9.5, style="italic")
    ax.text(0.73, 0.90, r"$\sigma=0.5\,\mu^*$" + "\n(monotonic)",
            transform=ax.transAxes, color="0.55", fontsize=9.5, style="italic")

    highlight = {"gamma": CB["red"], "c_R_base": CB["orange"]}
    for i in range(len(df)):
        col = highlight.get(keys[i], CB["blue"])
        big = keys[i] in highlight
        ax.scatter([mu_s[i]], [sg_s[i]], s=150 if big else 75, color=col,
                   edgecolor="white", linewidth=1.0, zorder=5 if big else 4)

    # leader-lined labels stacked down the right margin, ordered by mu*
    order = np.argsort(-mu_s)
    y_slots = np.linspace(ymax * 0.80, ymax * 0.22, len(df))
    x_label = xmax * 0.58
    for slot, i in zip(y_slots, order):
        col = highlight.get(keys[i], "0.15")
        ax.annotate(labels[i], xy=(mu_s[i], sg_s[i]),
                    xytext=(x_label, slot), fontsize=10, va="center", ha="left",
                    color=col, fontweight="bold" if keys[i] in highlight else "normal",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"Mean absolute effect  $\mu^*$  ($\times 10^{13}$)")
    ax.set_ylabel(r"Standard deviation  $\sigma$  ($\times 10^{13}$)")
    ax.set_title(r"Morris screening: parameter importance at fixed $T=137$"
                 "\n" r"(seven-parameter, $\pm20\%$ design box)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"  figure -> {out_png}")
    return out_png


if __name__ == "__main__":
    df, raw_df = run_morris()
    df.to_csv(OUT_DATA / "morris_screening.csv", index=False)
    print(f"  data -> {OUT_DATA / 'morris_screening.csv'}")
    raw_df.to_csv(OUT_DATA / "morris_elementary_effects.csv", index=False)
    print(f"  raw effects -> {OUT_DATA / 'morris_elementary_effects.csv'}")
    make_figure(df)
