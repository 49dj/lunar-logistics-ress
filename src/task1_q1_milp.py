#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Deterministic MILP baseline (manuscript Section 4)
Reliability-informed robust optimisation of Earth–Moon infrastructure logistics
=============================================================================
Model:  Temporal-Logistical MILP with ε-Constraint Method
Primary result (NH=1, Single Harbor):
  T_min = 119 yr,  T_max = 233 yr
  Knee  = 137 yr   Cost = $210.71 T   Elevator = 48.8%
NH=3 comparison:
  Knee  =  93 yr   Cost = $171.9 T    Elevator = 75.5%

Run (from the repository root):
    pip install -r requirements.txt
    python src/task1_q1_milp.py
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
import warnings
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent  # repository root
OUT_FIG  = ROOT / "figures"
OUT_DATA = ROOT / "data"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

STYLE = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    'mathtext.fontset': 'stix',
}
plt.rcParams.update(STYLE)

CB = dict(blue='#2E86AB', red='#E94F37', orange='#F39C12',
          green='#27AE60', purple='#8E44AD', gray='#95A5A6',
          teal='#2A9D8F', coral='#E76F51', navy='#264653')


def _save(fig, name):
    p = OUT_FIG / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"  ✓ {p.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class P:
    """Parameters for SINGLE HARBOR (NH=1) — manuscript's primary baseline."""
    M_TARGET   = 1e8          # total mass [tons]
    N_H        = 1            # number of harbours (baseline)
    G0         = 9.81         # [m/s²]
    DELTA_V    = 4500         # apex→Moon Δv [m/s]

    # Space Elevator (single harbour)
    CAP_E_INIT = 179_000      # [tons/yr] per harbour
    CAP_E_MAX  = 895_000      # [tons/yr] 5× ceiling
    GROWTH_K   = 0.12         # S-curve growth rate [1/yr]
    T_MID_E    = 50           # S-curve midpoint [yr from 2050]
    FIXED_COST = 100e9        # infrastructure [$]
    C_E_OP     = 1.3e6        # elevator op. cost [$/ton]

    # Rocket (GLOBAL — not per harbor; rockets launch from Earth pads independently)
    P_R           = 138       # payload/launch [tons]
    N_LAUNCH_INIT = 2029      # global launches/yr (same for NH=1 and NH=3)
    N_LAUNCH_MAX  = 5000      # global max launches/yr
    T_MID_R       = 50
    C_R_BASE   = 6.5e6        # [$/ton]
    GAMMA      = 1.2          # cost elasticity

    # Isp evolution
    ISP_INIT   = 450          # [s]
    ISP_MAX    = 900          # [s]
    T_MID_ISP  = 50

    BASE_YEAR  = 2050


class P3(P):
    """NH=3 scenario (optimistic) — three harbours."""
    N_H            = 3
    CAP_E_INIT     = 537_000   # 3 × 179k
    CAP_E_MAX      = 2_685_000 # 3 × 895k
    N_LAUNCH_INIT  = 2029
    N_LAUNCH_MAX   = 5000


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TECHNOLOGY MODEL
# ─────────────────────────────────────────────────────────────────────────────

def clone_params(base=None):
    """Independent copy of a parameter CLASS as a plain namespace object.

    copy.copy() on a class returns the SAME class object, so perturbing a
    "copy" silently mutates the global parameters for every later caller in
    the process (this contaminated downstream modules before it was fixed).
    """
    base = base if base is not None else P
    class _Params: pass
    q = _Params()
    for k in dir(base):
        if not k.startswith('_'):
            setattr(q, k, getattr(base, k))
    return q


def sigmoid(t, v0, vmax, k, t_mid):
    return v0 + (vmax - v0) / (1 + np.exp(-k * (t - t_mid)))


def isp(t, params=P):
    return sigmoid(t, params.ISP_INIT, params.ISP_MAX, params.GROWTH_K, params.T_MID_ISP)


def kappa(t, params=P):
    """Tsiolkovsky bootstrapping factor κ(t) = exp(Δv / (g₀·Isp(t)))."""
    return np.exp(params.DELTA_V / (params.G0 * isp(t, params)))


def cap_elevator_effective(t, params=P):
    """Effective SE capacity [tons/yr] = nominal / κ."""
    cap_nom = sigmoid(t, params.CAP_E_INIT, params.CAP_E_MAX,
                      params.GROWTH_K, params.T_MID_E)
    return cap_nom / kappa(t, params)


def n_launch_max(t, params=P):
    return int(sigmoid(t, params.N_LAUNCH_INIT, params.N_LAUNCH_MAX,
                       params.GROWTH_K, params.T_MID_R))


def cost_rocket(t, params=P):
    """Time-dependent rocket cost [$/ton] via reusability elasticity."""
    return params.C_R_BASE * (params.ISP_INIT / isp(t, params)) ** params.GAMMA


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MILP SOLVER
# ─────────────────────────────────────────────────────────────────────────────
def solve_milp(T: int, params=P) -> dict:
    """
    Solve cost-minimizing MILP for fixed horizon T.

    Variables: [a_0,...,a_{T-1}, x_0,...,x_{T-1}, n_0,...,n_{T-1}]
    """
    N = T  # number of time steps
    ts = np.arange(N)

    c_e = params.C_E_OP * np.ones(N)
    c_r = np.array([cost_rocket(t, params) for t in ts])
    # integer variable cost (n_t) — objective only cares about x_t
    c_n = np.zeros(N)

    c_obj = np.concatenate([c_e, c_r, c_n])  # length 3N
    integrality = np.concatenate([np.zeros(2*N), np.ones(N)])  # n_t integer

    # Constraints:
    # 1. Demand: sum(a+x) >= M_TARGET
    # 2. Elevator cap: a_t <= C_eff(t)  → -a_t >= -C_eff(t)
    # 3. Rocket-launch coupling: x_t <= n_t * P_R  → x_t - n_t*P_R <= 0
    # 4. Launch limit: n_t <= N_max(t)

    C_eff = np.array([cap_elevator_effective(t, params) for t in ts])
    N_max = np.array([n_launch_max(t, params) for t in ts])

    # Constraint 1: one equality-style lower bound (as lb ≤ A·z ≤ ub)
    row_demand = np.concatenate([np.ones(N), np.ones(N), np.zeros(N)])

    rows = [row_demand]
    lbs  = [params.M_TARGET]
    ubs  = [np.inf]

    for t in range(N):
        # 2. a_t ≤ C_eff(t)
        row = np.zeros(3*N); row[t] = 1.0
        rows.append(row); lbs.append(-np.inf); ubs.append(C_eff[t])

        # 3. x_t - P_R * n_t ≤ 0
        row = np.zeros(3*N); row[N+t] = 1.0; row[2*N+t] = -params.P_R
        rows.append(row); lbs.append(-np.inf); ubs.append(0.0)

        # 4. n_t ≤ N_max(t)
        row = np.zeros(3*N); row[2*N+t] = 1.0
        rows.append(row); lbs.append(-np.inf); ubs.append(N_max[t])

    A = np.array(rows)
    lbs = np.array(lbs, dtype=float)
    ubs = np.array(ubs, dtype=float)

    bounds = Bounds(lb=np.zeros(3*N), ub=np.full(3*N, np.inf))
    lc = LinearConstraint(A, lbs, ubs)

    res = milp(c_obj, constraints=lc, integrality=integrality, bounds=bounds,
               options={'disp': False, 'time_limit': 30})

    if res.status not in (0, 1) or res.x is None or res.fun is None:
        return None

    z = res.x
    a = z[:N]; x = z[N:2*N]; n = z[2*N:]
    total_cost = params.FIXED_COST + float(res.fun)
    frac_e = a.sum() / (a.sum() + x.sum()) if (a.sum() + x.sum()) > 0 else 0

    return {
        'T': T, 'a': a, 'x': x, 'n': n,
        'total_cost': total_cost,
        'frac_elevator': frac_e,
        'mass_elevator': a.sum(),
        'mass_rocket': x.sum(),
        'solver_status': 'optimal' if res.status == 0 else 'limit_reached',
        'solver_code': int(res.status),
        'solver_message': str(res.message),
    }


def find_T_min(params=P, T_lo=50, T_hi=300) -> int:
    """Binary search for minimum feasible T."""
    while T_lo < T_hi:
        mid = (T_lo + T_hi) // 2
        r = solve_milp(mid, params)
        if r is not None:
            T_hi = mid
        else:
            T_lo = mid + 1
    return T_lo


def pareto_frontier(T_min: int, T_max: int, params=P) -> pd.DataFrame:
    """Run MILP for each T in [T_min, T_max] to build Pareto frontier."""
    rows = []
    for T in range(T_min, T_max + 1):
        r = solve_milp(T, params)
        if r is not None:
            rows.append({
                'T': T,
                'cost_T': r['total_cost'] / 1e12,  # Trillion $
                'frac_elevator': r['frac_elevator'],
            })
    return pd.DataFrame(rows)


def find_knee(df: pd.DataFrame) -> pd.Series:
    """
    Knee point: max perpendicular distance from line (T_min→T_max).
    Vector Normalization Method (manuscript §4.3).
    """
    pts = df[['T', 'cost_T']].values.astype(float)
    # Normalize
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    pn = (pts - mn) / (mx - mn + 1e-12)

    # Line from first to last point
    v = pn[-1] - pn[0]
    v_norm = v / np.linalg.norm(v)

    dists = []
    for p in pn:
        w = p - pn[0]
        d = np.linalg.norm(w - np.dot(w, v_norm) * v_norm)
        dists.append(d)

    idx = np.argmax(dists)
    return df.iloc[idx]


def lp_norm_score(T_norm, C_norm, w, p=8):
    return (w * C_norm**p + (1-w) * T_norm**p) ** (1/p)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def morris_screening(n_trials=200, seed=42):
    """
    LEGACY ONLY: this five-parameter diagnostic is retained for backward
    compatibility. Manuscript Figure 10 is generated by the seven-parameter
    realized-step implementation in ``fig10_morris_screening.py``.

    Simplified Morris μ* for 5 key parameters.
    Returns (param_names, mu_star, sigma).
    """
    rng = np.random.default_rng(seed)
    params_keys = ['gamma', 'C_R_BASE', 'P_R', 'C_E_OP', 'CAP_E_MAX']
    base_vals   = [1.2,     6.5e6,      138,    1.3e6,   895_000]
    delta       = 0.10  # 10% perturbation

    # Reference cost at T=137
    def eval_cost(pv):
        p = clone_params(P)
        p.GAMMA    = pv[0]
        p.C_R_BASE = pv[1]
        p.P_R      = pv[2]
        p.C_E_OP   = pv[3]
        p.CAP_E_MAX= pv[4]
        r = solve_milp(137, p)
        return r['total_cost'] / 1e12 if r else np.nan

    ee = {k: [] for k in params_keys}
    for _ in range(n_trials):
        x = np.array(base_vals) * (1 + rng.uniform(-delta, delta, len(base_vals)))
        f0 = eval_cost(x)
        if np.isnan(f0): continue
        for i, k in enumerate(params_keys):
            xp = x.copy(); xp[i] *= (1 + delta)
            fp = eval_cost(xp)
            if not np.isnan(fp):
                ee[k].append(abs(fp - f0) / (f0 * delta + 1e-12))

    mu_star = np.array([np.mean(ee[k]) if ee[k] else 0 for k in params_keys])
    sigma   = np.array([np.std(ee[k])  if ee[k] else 0 for k in params_keys])
    return params_keys, mu_star, sigma


def prcc_analysis(n_samples=300, seed=0):
    """
    LEGACY ONLY: this closed-form diagnostic is not confirmatory PRCC and is
    not called by the publication workflow. Use ``run_prcc.py`` for
    the rank-residualized LHS-MILP implementation.

    Partial Rank Correlation Coefficients (simplified).
    Returns (labels, prcc_values).
    """
    rng = np.random.default_rng(seed)
    N = n_samples
    labels = [r'$\gamma$', r'$P_R$', r'$C_R^{base}$', r'$C_E^{op}$', r'$C_E^{max}$']
    # Sample parameter space ±20%
    gamma   = rng.uniform(0.96, 1.44, N)    # ±20% of 1.2
    P_R     = rng.uniform(110, 166, N)       # ±20% of 138
    C_R     = rng.uniform(5.2e6, 7.8e6, N)  # ±20% of 6.5M
    C_E     = rng.uniform(1.04e6, 1.56e6, N)
    cap_max = rng.uniform(716e3, 1074e3, N)

    # Approximate cost at knee T=137 using closed-form approximation
    # (full MILP is too slow for 300 samples; use scaling)
    # Cost ≈ FIXED + C_E_OP * M_E + C_R(T) * M_R
    # At knee: frac_e ≈ 0.488
    M_E = 0.488 * 1e8; M_R = 0.512 * 1e8
    Isp_knee = float(isp(37))  # t=37 into 137yr horizon midpoint-ish
    c_R_t = C_R * (450 / Isp_knee) ** gamma
    costs = 100e9 + C_E * M_E + c_R_t * M_R

    X = np.column_stack([gamma, P_R, C_R, C_E, cap_max])
    from scipy.stats import spearmanr, rankdata
    r_costs = rankdata(costs)
    prcc = []
    for i in range(X.shape[1]):
        r_xi = rankdata(X[:, i])
        # Partial: residualize against other columns
        other = np.delete(X, i, axis=1)
        from numpy.linalg import lstsq
        res_x = r_xi - other @ lstsq(other, r_xi, rcond=None)[0]
        res_y = r_costs - other @ lstsq(other, r_costs, rcond=None)[0]
        corr  = np.corrcoef(res_x, res_y)[0, 1]
        prcc.append(corr)

    return labels, np.array(prcc)


def monte_carlo_pareto(n_mc=200, seed=1):
    """
    Generate probabilistic Pareto cloud by perturbing parameters ±20%.
    Returns list of (T_array, cost_array) pairs.
    """
    rng = np.random.default_rng(seed)
    clouds = []
    for _ in range(n_mc):
        p = clone_params(P)
        p.GAMMA    = P.GAMMA    * rng.uniform(0.8, 1.2)
        p.C_R_BASE = P.C_R_BASE * rng.uniform(0.8, 1.2)
        p.C_E_OP   = P.C_E_OP   * rng.uniform(0.8, 1.2)
        p.CAP_E_MAX= P.CAP_E_MAX* rng.uniform(0.8, 1.2)
        row = []
        for T in range(110, 180, 5):
            r = solve_milp(T, p)
            if r:
                row.append((T, r['total_cost']/1e12))
        if row:
            clouds.append(list(zip(*row)))
    return clouds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def fig_pareto(df1: pd.DataFrame, df3: pd.DataFrame, knee1: pd.Series, knee3: pd.Series):
    """Figure 3 & 4: Pareto frontiers for NH=1 and NH=3."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, knee, nh, color in zip(
            axes,
            [df1, df3],
            [knee1, knee3],
            [1, 3],
            [CB['blue'], CB['teal']]):
        ax.plot(df['T'], df['cost_T'], '-', color=color, lw=2.0, label=f'NH={nh} Frontier')
        ax.scatter([knee['T']], [knee['cost_T']], s=120, color=CB['red'],
                   zorder=5, label=f'Knee T*={int(knee["T"])}yr')
        ax.annotate(f"  T*={int(knee['T'])}yr\n  $\\${knee['cost_T']:.2f}$T\n  Elev={knee['frac_elevator']*100:.1f}%",
                    xy=(knee['T'], knee['cost_T']),
                    xytext=(knee['T']+8, knee['cost_T']+8),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
        ax.set_xlabel('Project Duration $T$ [years]')
        ax.set_ylabel('Total Cost [Trillion USD]')
        ax.set_title(f'Pareto Frontier ($N_H={nh}$)')
        ax.legend()

    plt.tight_layout()
    _save(fig, 'fig1_pareto_frontier_nh1_nh3')


def fig_transport_schedule(df1: pd.DataFrame, knee1: pd.Series):
    """Figure showing annual transport schedule at Knee Point."""
    T_knee = int(knee1['T'])
    r = solve_milp(T_knee)
    if r is None: return

    ts = np.arange(T_knee) + P.BASE_YEAR
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Annual mass breakdown
    ax = axes[0]
    ax.fill_between(ts, 0, r['a']/1e3, alpha=0.7, color=CB['blue'], label='Space Elevator')
    ax.fill_between(ts, r['a']/1e3, (r['a']+r['x'])/1e3, alpha=0.7, color=CB['coral'], label='Rockets')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Mass [ktons/yr]')
    ax.set_title(f'Annual Transport Schedule (T*={T_knee}yr, NH=1)')
    ax.legend()

    # Cumulative mass
    ax = axes[1]
    cum = np.cumsum(r['a'] + r['x']) / 1e6  # mega-tons
    ax.plot(ts, cum, color=CB['navy'], lw=2)
    ax.axhline(1e8/1e6, color=CB['red'], ls='--', label='Target 10⁸ tons')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Mass [Mtons]')
    ax.set_title('Cumulative Mass Delivery')
    ax.legend()

    plt.tight_layout()
    _save(fig, 'fig2_transport_schedule_knee')


def fig_elevator_fraction_evolution(df1: pd.DataFrame, df3: pd.DataFrame):
    """Elevator fraction vs planning horizon."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df1['T'], df1['frac_elevator']*100, color=CB['blue'], lw=2, label='NH=1 (Single Harbor)')
    ax.plot(df3['T'], df3['frac_elevator']*100, color=CB['teal'], lw=2, ls='--', label='NH=3 (Full Capacity)')
    ax.axhline(48.8, color=CB['blue'], ls=':', alpha=0.6)
    ax.axhline(75.5, color=CB['teal'], ls=':', alpha=0.6)
    ax.set_xlabel('Project Duration $T$ [years]')
    ax.set_ylabel('Space Elevator Mass Share [%]')
    ax.set_title('Optimal Elevator Fraction vs Timeline')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'fig3_elevator_fraction')


def fig_lp_norm(df1: pd.DataFrame):
    """Figure 4: Lp-norm preference weight scan (p=8)."""
    T_arr = df1['T'].values
    C_arr = df1['cost_T'].values
    T_n = (T_arr - T_arr.min()) / (T_arr.max() - T_arr.min())
    C_n = (C_arr - C_arr.min()) / (C_arr.max() - C_arr.min())

    weights = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.cm.viridis
    for i, w in enumerate(weights):
        scores = [(w*c**8 + (1-w)*t**8)**(1/8) for c, t in zip(C_n, T_n)]
        best_i = np.argmin(scores)
        color = cmap(i / len(weights))
        ax.scatter(T_arr[best_i], C_arr[best_i], s=60, color=color, zorder=3)

    ax.plot(T_arr, C_arr, 'k-', lw=1.5, alpha=0.4, label='Pareto Frontier (NH=1)')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Weight $w$ (Cost emphasis)')
    ax.set_xlabel('Project Duration $T$ [years]')
    ax.set_ylabel('Total Cost [Trillion USD]')
    ax.set_title('$L_p$-Norm Preference Scan ($p=8$) — Solutions Converge Near Knee')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'fig4_lp_norm_preference')


def fig_sensitivity_tornado(knee1: pd.Series):
    """Figure 6: Tornado diagram — local elasticities at Knee Point."""
    T_k = int(knee1['T'])
    r0  = solve_milp(T_k)
    if r0 is None: return
    C0  = r0['total_cost']; T0 = T_k; fe0 = r0['frac_elevator']

    param_labels = [r'$\gamma$', r'$C_R^{base}$', r'$P_R$', r'$C_E^{op}$', r'$C_E^{max}$',
                    r'$I_{sp,max}$', r'$k_{growth}$']
    perturbation = 0.10

    def perturb(key, factor):
        p = clone_params(P)
        attrs = {'gamma': 'GAMMA', 'C_R': 'C_R_BASE', 'P_R': 'P_R',
                 'C_E': 'C_E_OP', 'cap_max': 'CAP_E_MAX',
                 'isp_max': 'ISP_MAX', 'k': 'GROWTH_K'}
        setattr(p, attrs[key], getattr(P, attrs[key]) * factor)
        return solve_milp(T_k, p)

    keys = ['gamma', 'C_R', 'P_R', 'C_E', 'cap_max', 'isp_max', 'k']
    elas_C = []
    for key in keys:
        rp = perturb(key, 1+perturbation)
        rm = perturb(key, 1-perturbation)
        if rp and rm:
            dC = (rp['total_cost'] - rm['total_cost']) / (2*perturbation*C0)
            elas_C.append(dC)
        else:
            elas_C.append(0.0)

    elas_C = np.array(elas_C)
    sort_idx = np.argsort(np.abs(elas_C))[::-1]
    labels_s = [param_labels[i] for i in sort_idx]
    elas_s   = elas_C[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CB['red'] if e > 0 else CB['green'] for e in elas_s]
    ax.barh(range(len(labels_s)), elas_s, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels_s))); ax.set_yticklabels(labels_s)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Normalized Elasticity $S_p = (\\partial y/y)/(\\partial p/p)$')
    ax.set_title(f'Local Parameter Elasticity at Knee ($T^*={T_k}$ yr)\n'
                 f'Red = positive, Green = negative')
    plt.tight_layout()
    _save(fig, 'fig6_tornado_elasticity')


def fig_monte_carlo(clouds):
    """Figure 7: Cone of uncertainty + Monte Carlo knee distribution."""
    if not clouds: return

    T_grid = np.arange(115, 180)
    cost_at_T = {T: [] for T in T_grid}
    for (Ts, Cs) in clouds:
        for T, C in zip(Ts, Cs):
            if T in cost_at_T:
                cost_at_T[T].append(C)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel b: cone of uncertainty
    ax = axes[1]
    medians = []; lo = []; hi = []
    T_plot = [T for T in T_grid if len(cost_at_T[T]) >= 5]
    for T in T_plot:
        v = cost_at_T[T]
        medians.append(np.median(v))
        lo.append(np.percentile(v, 2.5))
        hi.append(np.percentile(v, 97.5))
    ax.fill_between(T_plot, lo, hi, alpha=0.3, color=CB['blue'], label='95% CI')
    ax.plot(T_plot, medians, color=CB['blue'], lw=2, label='Median')
    ax.axvline(137, color=CB['red'], ls='--', lw=1.5, label='Knee T*=137')
    ax.set_xlabel('Project Duration $T$ [years]')
    ax.set_ylabel('Total Cost [Trillion USD]')
    ax.set_title('(b) Cone of Uncertainty')
    ax.legend()

    # Panel a: knee distribution
    ax = axes[0]
    knee_costs = []
    for (Ts, Cs) in clouds:
        arr = list(zip(Ts, Cs))
        if len(arr) < 3: continue
        # find pseudo-knee: max distance heuristic
        Ta = np.array([x[0] for x in arr])
        Ca = np.array([x[1] for x in arr])
        Tn = (Ta - Ta.min())/(Ta.max()-Ta.min()+1e-9)
        Cn = (Ca - Ca.min())/(Ca.max()-Ca.min()+1e-9)
        dists = []
        v = np.array([Tn[-1]-Tn[0], Cn[-1]-Cn[0]])
        vn = v / (np.linalg.norm(v)+1e-9)
        for tn, cn in zip(Tn, Cn):
            w = np.array([tn-Tn[0], cn-Cn[0]])
            d = np.linalg.norm(w - np.dot(w,vn)*vn)
            dists.append(d)
        ki = np.argmax(dists)
        knee_costs.append(Ca[ki])

    if knee_costs:
        ax.hist(knee_costs, bins=20, color=CB['blue'], alpha=0.7, edgecolor='white')
        ax.axvline(210.71, color=CB['red'], ls='--', lw=1.5, label='Manuscript $\\$210.71$T')
        lo95 = np.percentile(knee_costs, 2.5)
        hi95 = np.percentile(knee_costs, 97.5)
        ax.axvspan(lo95, hi95, alpha=0.15, color=CB['orange'],
                   label=f'95% CI [{lo95:.0f}–{hi95:.0f}]T')
        ax.set_xlabel('Knee Point Cost [Trillion USD]')
        ax.set_ylabel('Count')
        ax.set_title('(a) Confidence Region of Optimal Strategy')
        ax.legend()

    plt.tight_layout()
    _save(fig, 'fig7_monte_carlo_cone')


def fig_morris(param_labels, mu_star, sigma):
    """Figure 5a: Morris screening μ*–σ plane."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mu_star, sigma, s=80, color=CB['blue'], zorder=3)
    for name, mx, sx in zip(param_labels, mu_star, sigma):
        ax.annotate(name, (mx, sx), textcoords='offset points',
                    xytext=(5, 3), fontsize=10)
    ax.set_xlabel(r'$\mu^*$ (Mean absolute EE)')
    ax.set_ylabel(r'$\sigma$ (Std of EE)')
    ax.set_title('Legacy five-parameter diagnostic (not manuscript Figure 10)')
    plt.tight_layout()
    _save(fig, 'fig5a_morris_screening')


def fig_prcc(labels, prcc):
    """Figure 5b: PRCC bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [CB['green'] if v > 0 else CB['red'] for v in prcc]
    ax.bar(labels, prcc, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('PRCC')
    ax.set_title('Legacy closed-form diagnostic (not manuscript Figure 11)')
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    _save(fig, 'fig5b_prcc')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_optimization(verbose=True):
    print("=" * 60)
    print("TASK 1: Multi-Objective MILP Optimization")
    print("Manuscript §4.4 — NH=1 single-harbour baseline")
    print("=" * 60)

    # ── NH=1 (primary) ───────────────────────────────────────────────────────
    print("\n[1/4] Computing NH=1 Pareto frontier...")
    T_min1 = find_T_min(P, T_lo=80, T_hi=200)
    print(f"  T_min (NH=1) = {T_min1} yr  [manuscript: 119 yr]")

    # Manuscript uses T_max = 233 yr (elevator-only completion); we use 233 directly
    T_max1 = 233
    print(f"  T_max (NH=1) = {T_max1} yr  [manuscript: 233 yr]")

    df1 = pareto_frontier(T_min1, min(T_max1, 240), P)
    knee1 = find_knee(df1)
    print(f"\n  ► Knee (NH=1): T*={int(knee1['T'])}yr, "
          f"Cost=${knee1['cost_T']:.2f}T, "
          f"Elevator={knee1['frac_elevator']*100:.1f}%")
    print(f"  [manuscript: T*=137yr, Cost=$210.71T, Elevator=48.8%]")

    # ── NH=3 (comparison) ────────────────────────────────────────────────────
    print("\n[2/4] Computing NH=3 Pareto frontier...")
    T_min3 = find_T_min(P3, T_lo=60, T_hi=150)
    T_max3 = 110
    df3 = pareto_frontier(T_min3, T_max3, P3)
    knee3 = find_knee(df3)
    print(f"  ► Knee (NH=3): T*={int(knee3['T'])}yr, "
          f"Cost=${knee3['cost_T']:.2f}T, "
          f"Elevator={knee3['frac_elevator']*100:.1f}%")
    print(f"  [manuscript: T*=93yr, Cost=$171.9T, Elevator=75.5%]")

    # Export
    df1.to_csv(OUT_DATA / 'pareto_nh1.csv', index=False)
    df3.to_csv(OUT_DATA / 'pareto_nh3.csv', index=False)

    # ── Sensitivity ──────────────────────────────────────────────────────────
    print("\n[3/4] Sensitivity analysis...")
    print("  Figure 10 is generated by src/fig10_morris_screening.py")
    print("  Figure 11 is generated by src/run_prcc.py")

    print("\n[4/4] Monte Carlo cone of uncertainty...")
    clouds = monte_carlo_pareto(n_mc=100, seed=42)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_pareto(df1, df3, knee1, knee3)
    fig_transport_schedule(df1, knee1)
    fig_elevator_fraction_evolution(df1, df3)
    fig_lp_norm(df1)
    fig_sensitivity_tornado(knee1)
    fig_monte_carlo(clouds)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Manuscript §4.4 results")
    print("=" * 60)
    print(f"  Scenario        T_min   T_max   T*     Cost($T)   Elev%")
    print(f"  NH=1 (primary)  {T_min1:>5}   {T_max1:>5}   "
          f"{int(knee1['T']):>5}   {knee1['cost_T']:>7.2f}   {knee1['frac_elevator']*100:.1f}%")
    print(f"  NH=3 (alt)      {T_min3:>5}   {T_max3:>5}   "
          f"{int(knee3['T']):>5}   {knee3['cost_T']:>7.2f}   {knee3['frac_elevator']*100:.1f}%")
    print(f"\n  Manuscript reference:  NH=1 → T*=137yr $210.71T 48.8%")
    print(f"                    NH=3 → T*= 93yr $171.9T  75.5%")
    print(f"\n  Figures saved to: {OUT_FIG}")

    return {
        'df1': df1, 'df3': df3, 'knee1': knee1, 'knee3': knee3,
        'T_min1': T_min1, 'T_max1': T_max1,
        'T_min3': T_min3, 'T_max3': T_max3,
    }


if __name__ == '__main__':
    results = run_optimization(verbose=True)
