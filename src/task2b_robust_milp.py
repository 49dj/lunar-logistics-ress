#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task2b_robust_milp.py — Reliability-adjusted MILP re-solve
(manuscript Section 6.1; source of every robust-optimisation number in
Sections 6.1 and 7.1 and Tables 8-10).

Model = Task-1 deterministic MILP (task1_q1_milp.py, imported unchanged) with:
  (i)   elevator capacity derated by CTMC availability A_ss (Eq. 11)
  (ii)  rocket throughput derated by (1 - q_eff) (Eq. 16)
  (iii) risk-adjusted objective (Eq. 21):
          C_fail : (1 + PHI*q_T) loading on rocket operating cost
          C_repl : annualised tether replacement provision U_REPL * T
          C_rare : campaign-level rare-event provision B_RISK * C_DET
Selection: knee of the reliability-adjusted cost-time frontier, same
maximum-perpendicular-distance criterion as the deterministic baseline.

Calibrated inputs (Section 6.1, Table 8):
  A_SS=0.8485  q_T=0.0047  q_W=0.20  ->  q_eff=0.2038
  PHI=2.0   U_REPL=$76.1B/yr (~$1.9T per ~25-yr cycle)
  B_RISK in {0.05, 0.074, 0.10}, baseline 0.074 (C_rare = I_BS + E[L_cat])

Outputs (paths relative to the repository root):
  outputs/pareto_robust.csv               130-point frontier, T = 137-266
  figures/figure8_robust_comparison.png   manuscript Figure 8
  printed knee decomposition              manuscript Tables 8-9

Usage (from the repository root):
  python src/task2b_robust_milp.py          # frontier + knee + figure
  python src/task2b_robust_milp.py --scan   # exhaustive feasibility scan
                                            #   T = 100..137 -> logs/robust_feasibility_scan.csv
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_q1_milp as t1
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT     = Path(__file__).resolve().parent.parent   # repository root
OUT_DATA = ROOT / "outputs"
OUT_FIG  = ROOT / "figures"
OUT_LOG  = ROOT / "logs"
for _d in (OUT_DATA, OUT_FIG, OUT_LOG):
    _d.mkdir(parents=True, exist_ok=True)

P = t1.P
C_DET = 210.707866e12          # deterministic knee cost (data/pareto_nh1.csv, T=137)
T_DET = 137
FRAC_E_DET = 0.487789

A_SS   = 0.8485
Q_T    = 0.0047
Q_W    = 0.20
P_SUCC = (1 - Q_T) * (1 - Q_W)
Q_EFF  = 1 - P_SUCC
LAM_T  = 1 / P_SUCC

PHI    = 2.0
U_REPL = 76.1e9                # $/yr
B_RISK = 0.074                 # baseline; sensitivity {0.05, 0.074, 0.10}


def solve_milp_robust(T, a_ss=A_SS, q_eff=Q_EFF, q_t=Q_T,
                      phi=PHI, u_repl=U_REPL, b_risk=B_RISK):
    """Task-1 MILP with derated capacities and risk-adjusted objective."""
    N = T
    ts = np.arange(N)
    c_e = P.C_E_OP * np.ones(N)
    c_r = np.array([t1.cost_rocket(t) for t in ts]) * (1 + phi * q_t)
    c_obj = np.concatenate([c_e, c_r, np.zeros(N)])
    integrality = np.concatenate([np.zeros(2 * N), np.ones(N)])

    C_eff = a_ss * np.array([t1.cap_elevator_effective(t) for t in ts])
    N_max = np.array([t1.n_launch_max(t) for t in ts])

    rows = [np.concatenate([np.ones(N), np.ones(N), np.zeros(N)])]
    lbs, ubs = [P.M_TARGET], [np.inf]
    for t in range(N):
        r = np.zeros(3 * N); r[t] = 1.0
        rows.append(r); lbs.append(-np.inf); ubs.append(C_eff[t])
        r = np.zeros(3 * N); r[N + t] = 1.0; r[2 * N + t] = -P.P_R * (1 - q_eff)
        rows.append(r); lbs.append(-np.inf); ubs.append(0.0)
        r = np.zeros(3 * N); r[2 * N + t] = 1.0
        rows.append(r); lbs.append(-np.inf); ubs.append(N_max[t])

    lc = LinearConstraint(np.array(rows), np.array(lbs, float), np.array(ubs, float))
    res = milp(c_obj, constraints=lc, integrality=integrality,
               bounds=Bounds(lb=np.zeros(3 * N), ub=np.full(3 * N, np.inf)),
               options={'disp': False, 'time_limit': 60})
    if res.status not in (0, 1) or res.x is None:
        return None
    z = res.x
    a, x, n = z[:N], z[N:2 * N], z[2 * N:]
    ops_e = float((c_e * a).sum())
    ops_r_loaded = float((c_r * x).sum())
    ops_r = ops_r_loaded / (1 + phi * q_t)
    c_fail = ops_r_loaded - ops_r
    c_repl = u_repl * T
    c_rare = b_risk * C_DET
    total = P.FIXED_COST + ops_e + ops_r_loaded + c_repl + c_rare
    me, mr = float(a.sum()), float(x.sum())
    return dict(T=T, total_cost=total, frac_elevator=me / (me + mr),
                mass_elevator=me, mass_rocket=mr,
                ops_elevator=ops_e, ops_rocket_base=ops_r,
                c_fail=c_fail, c_repl=c_repl, c_rare=c_rare,
                n_attempts=float(n.sum()))


def find_T_min_robust(T_lo=100, T_hi=280):
    """Bisection for the minimum feasible robust horizon (assumes feasibility
    is monotonic in T; verified independently by feasibility_scan below)."""
    while T_lo < T_hi:
        mid = (T_lo + T_hi) // 2
        if solve_milp_robust(mid) is not None:
            T_hi = mid
        else:
            T_lo = mid + 1
    return T_lo


def feasibility_scan(T_lo=100, T_hi=137, out_csv=None):
    """Exhaustive infeasibility scan supporting the manuscript's zero-schedule-
    margin claim (Section 6.1): solve the reliability-adjusted MILP at EVERY
    integer horizon in [T_lo, T_hi] and record feasible/infeasible.

    Expected result: T = 100..136 infeasible; T = 137 feasible.
    Writes logs/robust_feasibility_scan.csv.
    """
    out_csv = Path(out_csv) if out_csv else OUT_LOG / "robust_feasibility_scan.csv"
    rows = []
    for T in range(T_lo, T_hi + 1):
        r = solve_milp_robust(T)
        rows.append({'T': T, 'feasible': bool(r),
                     'total_cost_T': round(r['total_cost'] / 1e12, 6) if r else ''})
        print(f"  T={T}: {'feasible' if r else 'infeasible'}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    n_inf = sum(1 for r in rows if not r['feasible'])
    print(f"  scan -> {out_csv}  ({n_inf} infeasible, "
          f"first feasible T = {next(r['T'] for r in rows if r['feasible'])})")
    return rows


def elevator_only_Tmax(cap=300):
    cum = 0.0
    for T in range(1, cap + 1):
        cum += A_SS * t1.cap_elevator_effective(T - 1)
        if cum >= P.M_TARGET:
            return T
    return cap


def make_figure(rk, out_png):
    """Manuscript Figure 8: (a) key metrics, (b) robust cost decomposition.

    Small pie wedges are labelled outside the pie with staggered leader
    lines so that no labels collide (publication requirement)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(t1.STYLE)
    CB = t1.CB
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    keys = ['Duration [yr]', 'Elevator [%]', 'Cost [$T / 10]']
    ideal = [T_DET, FRAC_E_DET * 100, C_DET / 1e13]
    robust = [rk['T'], rk['frac_elevator'] * 100, rk['total_cost'] / 1e13]
    w = 0.3
    ax.bar(np.arange(3) - w / 2, ideal, w, color=CB['green'], alpha=0.85,
           label='Deterministic (137 yr)')
    ax.bar(np.arange(3) + w / 2, robust, w, color=CB['red'], alpha=0.85,
           label=f"Reliability-adjusted ({rk['T']} yr)")
    for i, (v1, v2) in enumerate(zip(ideal, robust)):
        ax.text(i - w / 2, v1 + 2, f'{v1:.1f}', ha='center', fontsize=9)
        ax.text(i + w / 2, v2 + 2, f'{v2:.1f}', ha='center', fontsize=9)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(keys); ax.legend()
    ax.set_title('(a) Key Metrics: Deterministic vs Robust\n'
                 '(reliability-adjusted MILP re-solve)')

    ax = axes[1]
    ops = (P.FIXED_COST + rk['ops_elevator'] + rk['ops_rocket_base']) / 1e12
    total = rk['total_cost'] / 1e12
    sizes = [ops, rk['c_rare'] / 1e12, rk['c_repl'] / 1e12, rk['c_fail'] / 1e12]
    names = ['Operations\n(re-optimised)', 'Rare-event provision $C_{rare}$',
             'Replacement $C_{repl}$', 'Failure losses $C_{fail}$']
    colors = [CB['green'], CB['orange'], CB['purple'], CB['red']]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(edgecolor='white'))
    # Large wedge: label inside. Small wedges: staggered external labels with
    # leader lines. Slots follow the wedges' vertical order (rare lowest,
    # fail highest), so the leader lines cannot cross.
    ax.text(-0.35, -0.25,
            f"{names[0]}\n\\${sizes[0]:.1f}T ({sizes[0]/total*100:.1f}%)",
            ha='center', va='center', fontsize=10)
    slots = [0.45, 0.85, 1.25]        # y-positions: rare, repl, fail (bottom-up)
    for w_, name, val, y_slot in zip(wedges[1:], names[1:], sizes[1:], slots):
        ang = np.deg2rad(0.5 * (w_.theta1 + w_.theta2))
        x, y = np.cos(ang), np.sin(ang)
        ax.annotate(f"{name}: \\${val:.2f}T ({val/total*100:.1f}%)",
                    xy=(0.99 * x, 0.99 * y), xytext=(1.30, y_slot),
                    ha='left', va='center', fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='0.45', lw=0.9))
    ax.set_xlim(-1.25, 3.1)
    ax.set_title(f"(b) Robust Cost Decomposition\n"
                 f"Total=\\${total:.2f}T "
                 f"(+{(rk['total_cost']-C_DET)/C_DET*100:.1f}%); "
                 f"ops \\${ops:.1f}T < ideal \\${C_DET/1e12:.1f}T")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"  figure -> {out_png}")


def run_robust_analysis(write_outputs=True, verbose=True):
    """Full Section-6.1 analysis. Returns a dict of every manuscript-reported
    quantity (used by run_all.py for verification)."""
    t0 = time.time()
    if verbose:
        print("TASK 2b: reliability-adjusted robust MILP re-solve (manuscript §6.1)")
        print(f"  A_ss={A_SS} q_T={Q_T} q_W={Q_W} q_eff={Q_EFF:.4f} lam_T={LAM_T:.4f}")
        print(f"  PHI={PHI} U_REPL=${U_REPL/1e9:.1f}B/yr B_RISK={B_RISK} "
              f"(lump {B_RISK*C_DET/1e12:.2f}T)")

    T_min_r = find_T_min_robust()
    T_max_r = elevator_only_Tmax()
    if verbose:
        print(f"  feasible horizons: [{T_min_r}, {T_max_r}]  "
              f"(deterministic knee {T_DET} = robust minimum: zero schedule margin)")

    rows, sols = [], {}
    for T in range(T_min_r, T_max_r + 1):
        r = solve_milp_robust(T)
        if r:
            sols[T] = r
            rows.append({'T': T, 'cost_T': r['total_cost'] / 1e12,
                         'frac_elevator': r['frac_elevator']})
    df = pd.DataFrame(rows)
    if write_outputs:
        df.to_csv(OUT_DATA / 'pareto_robust.csv', index=False)
    knee = t1.find_knee(df); Tk = int(knee['T']); rk = sols[Tk]

    prem = rk['total_cost'] - C_DET
    ops = P.FIXED_COST + rk['ops_elevator'] + rk['ops_rocket_base']
    if verbose:
        print(f"\nROBUST KNEE: T*={Tk} yr  total=${rk['total_cost']/1e12:.2f}T "
              f"(+{prem/C_DET*100:.1f}%)  elevator share={rk['frac_elevator']*100:.1f}%")
        print(f"  ops(fixed+elev+rocket)=${ops/1e12:.2f}T (vs det ${C_DET/1e12:.2f}T, "
              f"delta {(ops-C_DET)/1e12:+.2f}T)")
        print(f"  C_fail=${rk['c_fail']/1e12:.2f}T  C_repl=${rk['c_repl']/1e12:.2f}T  "
              f"C_rare=${rk['c_rare']/1e12:.2f}T")
        print(f"  elevator cost share={rk['ops_elevator']/rk['total_cost']*100:.1f}% "
              f"(det {0.487789e8*P.C_E_OP/C_DET*100:.1f}%)")
        print(f"  naive lam_T x 137 = {LAM_T*137:.0f} yr -> re-optimisation recovers "
              f"{LAM_T*137 - Tk:.0f} yr")

    sens_b, sens_q = {}, {}
    if verbose:
        print("\nSensitivity:")
    for b in (0.05, 0.074, 0.10):
        tot = rk['total_cost'] + (b - B_RISK) * C_DET
        df2 = df.copy(); df2['cost_T'] += (b - B_RISK) * C_DET / 1e12
        k2 = int(t1.find_knee(df2)['T'])
        sens_b[b] = dict(T=k2, total=tot, share=rk['frac_elevator'])
        if verbose:
            print(f"  b_risk={b:<5}: T*={k2} yr  total=${tot/1e12:.2f}T "
                  f"(+{(tot-C_DET)/C_DET*100:.1f}%)  share={rk['frac_elevator']*100:.1f}%")
    for qt, lbl in ((0.0036, 'q_T mean'), (0.0088, 'q_T 95% UB')):
        qe = 1 - (1 - qt) * (1 - Q_W)
        rr = solve_milp_robust(Tk, q_eff=qe, q_t=qt)
        sens_q[qt] = dict(total=rr['total_cost'], share=rr['frac_elevator'])
        if verbose:
            print(f"  {lbl:<12}: total=${rr['total_cost']/1e12:.2f}T  "
                  f"share={rr['frac_elevator']*100:.1f}%")

    if write_outputs:
        make_figure(rk, str(OUT_FIG / 'figure8_robust_comparison.png'))
    if verbose:
        print(f"\ndone in {time.time()-t0:.1f}s")

    return dict(T_min=T_min_r, T_max=T_max_r, knee_T=Tk,
                total_cost=rk['total_cost'], frac_elevator=rk['frac_elevator'],
                premium_pct=prem / C_DET * 100,
                ops_total=ops, ops_delta=ops - C_DET,
                c_fail=rk['c_fail'], c_repl=rk['c_repl'], c_rare=rk['c_rare'],
                elevator_cost_share=rk['ops_elevator'] / rk['total_cost'],
                sens_b=sens_b, sens_q=sens_q, df=df)


if __name__ == '__main__':
    if '--scan' in sys.argv:
        print("TASK 2b: exhaustive feasibility scan (manuscript §6.1 evidence)")
        feasibility_scan()
    else:
        run_robust_analysis()
