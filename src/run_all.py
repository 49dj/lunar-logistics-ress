#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
run_all.py  —  Master verification script
Manuscript: "Reliability-informed robust optimisation of Earth-Moon
infrastructure logistics under launch and space-elevator disruptions"
=============================================================================
Runs every analysis script and checks the key numerical outputs against the
values reported in the manuscript:

  Task 1   deterministic MILP baseline .......... manuscript §4.4, Figure 2
  Task 2b  reliability-adjusted robust re-solve .. manuscript §6.1/§7.1, Tables 8-10
           + exhaustive feasibility scan ......... §6.1 zero-schedule-margin evidence
           + byte-identity of pareto_robust.csv .. reproducibility gate
  Figure 9 frontier-comparison figure ............ manuscript Figure 9
  Task 3   water-resupply extension .............. manuscript §8.4
  Task 4   environmental extension ............... manuscript §8.4

Usage (from the repository root):
    python src/run_all.py

Runtime: ~2-5 minutes on a laptop (Task 1's Morris/Monte-Carlo sweeps
dominate; Task 2b itself takes ~1 second).
=============================================================================
"""

import hashlib
import math
import sys
import time
from pathlib import Path

ROOT_SRC = Path(__file__).resolve().parent
ROOT     = ROOT_SRC.parent
sys.path.insert(0, str(ROOT_SRC))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(name, got, expected, tol_pct=2.0, fmt=".2f"):
    err = abs(got - expected) / (abs(expected) + 1e-12) * 100
    ok  = err <= tol_pct
    tag = PASS if ok else FAIL
    print(f"    {tag}  {name}: got={got:{fmt}}  expected={expected:{fmt}}  err={err:.2f}%")
    return ok


def check_exact(name, got, expected):
    ok = (got == expected)
    tag = PASS if ok else FAIL
    print(f"    {tag}  {name}: got={got}  expected={expected}")
    return ok


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes_match(reference_sha, generated_sha):
    """Return True only when a checked-in reference exists and matches."""
    return reference_sha is not None and reference_sha == generated_sha


def _scan_frame_is_valid(df):
    """Validate the deposited T=100..137 feasibility evidence by content."""
    if list(df.columns) != ['T', 'feasible', 'total_cost_T'] or len(df) != 38:
        return False
    try:
        horizons = [int(value) for value in df['T']]
        feasible = []
        for value in df['feasible']:
            token = str(value).strip().lower()
            if token not in {'true', 'false'}:
                return False
            feasible.append(token == 'true')
        cost_at_137 = float(df.loc[df['T'] == 137, 'total_cost_T'].iloc[0])
    except (IndexError, TypeError, ValueError):
        return False
    return (
        horizons == list(range(100, 138))
        and feasible == [False] * 37 + [True]
        and math.isclose(cost_at_137, 289.918795, rel_tol=0.0, abs_tol=5e-7)
    )


def run_task1():
    print("\n" + "=" * 60)
    print("TASK 1: Deterministic MILP baseline (manuscript §4.4)")
    print("=" * 60)
    t0 = time.time()

    import task1_q1_milp as t1
    res = t1.run_optimization(verbose=False)
    knee1 = res['knee1']
    knee3 = res['knee3']

    print(f"\n  Timing: {time.time()-t0:.1f}s")
    print("  Manuscript §4.4 verification:")

    passed = []
    passed.append(check("Knee T* (NH=1)",    knee1['T'],              137,    tol_pct=5.0, fmt=".0f"))
    passed.append(check("Knee Cost (NH=1)",  knee1['cost_T'],         210.71, tol_pct=8.0))
    passed.append(check("Elev Frac (NH=1)",  knee1['frac_elevator']*100, 48.8, tol_pct=8.0))
    passed.append(check("Knee T* (NH=3)",    knee3['T'],              93,     tol_pct=5.0, fmt=".0f"))
    passed.append(check("Knee Cost (NH=3)",  knee3['cost_T'],         171.9,  tol_pct=8.0))
    passed.append(check("Elev Frac (NH=3)",  knee3['frac_elevator']*100, 75.5, tol_pct=8.0))

    n_pass = sum(passed); n_total = len(passed)
    print(f"\n  Result: {n_pass}/{n_total} checks passed")
    return n_pass, n_total


def run_task2b():
    print("\n" + "=" * 60)
    print("TASK 2b: Reliability-adjusted robust re-solve (manuscript §6.1/§7.1)")
    print("=" * 60)
    t0 = time.time()

    import task2b_robust_milp as t2b

    csv_path = ROOT / "outputs" / "pareto_robust.csv"
    sha_before = _sha256(csv_path) if csv_path.exists() else None

    res = t2b.run_robust_analysis(write_outputs=True, verbose=False)

    print(f"\n  Timing: {time.time()-t0:.1f}s")
    print("  Manuscript Tables 8-10 / §6.1 / §7.1 verification:")

    passed = []
    passed.append(check_exact("Min feasible robust horizon", res['T_min'], 137))
    passed.append(check_exact("Robust knee T*",              res['knee_T'], 156))
    passed.append(check("Total cost ($T)",       res['total_cost']/1e12,     236.65, tol_pct=0.1))
    passed.append(check("Elevator share (%)",    res['frac_elevator']*100,   50.1,   tol_pct=0.5))
    passed.append(check("Cost premium (%)",      res['premium_pct'],         12.3,   tol_pct=1.0))
    passed.append(check("C_fail ($T)",           res['c_fail']/1e12,         1.34,   tol_pct=1.0))
    passed.append(check("C_repl ($T)",           res['c_repl']/1e12,         11.87,  tol_pct=1.0))
    passed.append(check("C_rare ($T)",           res['c_rare']/1e12,         15.59,  tol_pct=1.0))
    passed.append(check("Ops delta vs det ($T)", res['ops_delta']/1e12,      -2.86,  tol_pct=5.0))

    print("  b_risk sensitivity (Table 9) — knee invariant, cost level shifts:")
    exp_b = {0.05: 231.59, 0.074: 236.65, 0.10: 242.13}
    for b, exp in exp_b.items():
        passed.append(check_exact(f"T* at b_risk={b}", res['sens_b'][b]['T'], 156))
        passed.append(check(f"Total at b_risk={b} ($T)", res['sens_b'][b]['total']/1e12, exp, tol_pct=0.1))

    print("  Reproducibility gate:")
    sha_after = _sha256(csv_path)
    ok = _canonical_bytes_match(sha_before, sha_after)
    if sha_before is None:
        print(f"    {FAIL}  pareto_robust.csv had no checked-in copy to compare against")
    else:
        print(f"    {PASS if ok else FAIL}  pareto_robust.csv regenerated byte-identically "
              f"(sha256 {sha_after[:12]}...)")
    passed.append(ok)

    print("  Exhaustive feasibility scan (§6.1 zero-schedule-margin evidence):")
    rows = []
    scan_records = []
    for T in range(100, 138):
        r = t2b.solve_milp_robust(T)
        rows.append((T, bool(r)))
        scan_records.append({
            'T': T,
            'feasible': bool(r),
            'total_cost_T': round(r['total_cost'] / 1e12, 6) if r else '',
        })
    n_inf = sum(1 for _, f in rows if not f)
    first_feasible = next(T for T, f in rows if f)
    passed.append(check_exact("Infeasible horizons in [100,136]", n_inf, 37))
    passed.append(check_exact("First feasible horizon", first_feasible, 137))
    import pandas as pd
    scan_csv = ROOT / "logs" / "robust_feasibility_scan.csv"
    pd.DataFrame(scan_records).to_csv(scan_csv, index=False)
    scan_ok = _scan_frame_is_valid(pd.read_csv(scan_csv))
    print(f"    {PASS if scan_ok else FAIL}  deposited scan content: {scan_csv}")
    passed.append(scan_ok)

    n_pass = sum(passed); n_total = len(passed)
    print(f"\n  Result: {n_pass}/{n_total} checks passed")
    return n_pass, n_total


def run_fig9():
    print("\n" + "=" * 60)
    print("FIGURE 9: Frontier comparison (manuscript Figure 9)")
    print("=" * 60)
    import fig9_frontiers
    out = fig9_frontiers.main()
    ok = Path(out).exists()
    print(f"    {PASS if ok else FAIL}  figure9_frontiers.png generated")
    return (1 if ok else 0), 1


def run_task3():
    print("\n" + "=" * 60)
    print("TASK 3: Water-resupply extension (manuscript §8.4)")
    print("=" * 60)
    t0 = time.time()

    import task3_q3_water as t3
    res = t3.run_water_analysis()

    print(f"\n  Timing: {time.time()-t0:.1f}s")
    print("  Manuscript §8.4 verification:")

    passed = []
    passed.append(check("W_gross (t/yr)",   res['W_gross'],    4_519_000, tol_pct=0.1, fmt=".0f"))
    passed.append(check("W_earth (t/yr)",   res['W_earth'],      114_164, tol_pct=0.1, fmt=".0f"))
    passed.append(check("B buffer (t)",     res['B'],             22_833, tol_pct=0.5, fmt=".0f"))
    passed.append(check("W_Y1 (t)",         res['W_Y1'],         136_997, tol_pct=0.1, fmt=".0f"))
    passed.append(check("S1/S2 ratio",      res['cost_ratio'],        6.0, tol_pct=2.0))

    df_s = res['df_single']
    s1a_cost = float(df_s[df_s['config']=='S1A']['delta_C_T'])
    s2a_cost = float(df_s[df_s['config']=='S2A']['delta_C_T'])
    passed.append(check("S1A ΔC ($T)",      s1a_cost,   0.178, tol_pct=1.0))
    passed.append(check("S2A ΔC ($T)",      s2a_cost,   0.030, tol_pct=1.0))

    n_pass = sum(passed); n_total = len(passed)
    print(f"\n  Result: {n_pass}/{n_total} checks passed")
    return n_pass, n_total


def run_task4():
    print("\n" + "=" * 60)
    print("TASK 4: Environmental extension (manuscript §8.4)")
    print("=" * 60)
    t0 = time.time()

    import task4_q4_environment as t4
    res = t4.run_environment_analysis()

    print(f"\n  Timing: {time.time()-t0:.1f}s")
    print("  Manuscript §8.4 verification:")

    topsis_s = res['scores']['TOPSIS']
    prom_s   = res['scores']['PROMETHEE']

    passed = []
    # x_fracs[0] = 0.0 -> elevator-only; x_fracs[100] = 1.0 -> rocket-only
    elev_idx = 0
    rock_idx = 100

    passed.append(check("TOPSIS elev C*",       topsis_s[elev_idx],  1.000,   tol_pct=0.5))
    passed.append(check("TOPSIS rocket C*",     topsis_s[rock_idx],  0.000,   tol_pct=50.0))
    passed.append(check("PROMETHEE elev phi",   prom_s[elev_idx],    0.5217,  tol_pct=10.0))
    passed.append(check("PROMETHEE rocket phi", prom_s[rock_idx],   -0.4883,  tol_pct=10.0))
    passed.append(check("Tri-obj T_balanced",   res['T_tri'],        220,     tol_pct=20.0, fmt=".0f"))

    W_paper = t4.W_ENTROPY
    W_expected = [0.3847, 0.0841, 0.1419, 0.0430, 0.0607, 0.0805, 0.2051]
    for i, (got, exp) in enumerate(zip(W_paper, W_expected)):
        passed.append(check(f"w_entropy[C{i+1}]", got, exp, tol_pct=0.1))

    n_pass = sum(passed); n_total = len(passed)
    print(f"\n  Result: {n_pass}/{n_total} checks passed")
    return n_pass, n_total


# -------------------------------------------------------------------------
def main(task_specs=None):
    """Run the verification suite and return a process-style exit code.

    A task exception counts as one failed verification item.  This prevents a
    crashed task from contributing ``0/0`` and accidentally satisfying the
    aggregate ``total_pass == total_checks`` success condition.
    """
    import os
    os.chdir(str(ROOT))

    if task_specs is None:
        task_specs = [(run_task1,  'Task1  (§4.4)'),
                      (run_task2b, 'Task2b (§6.1/§7.1)'),
                      (run_fig9,   'Figure9'),
                      (run_task3,  'Task3  (§8.4)'),
                      (run_task4,  'Task4  (§8.4)')]

    print("=" * 60)
    print("MASTER VERIFICATION — manuscript numerical claims")
    print("=" * 60)

    total_pass = 0; total_checks = 0
    results = {}
    had_task_error = False

    for task_fn, name in task_specs:
        try:
            p, n = task_fn()
            if n <= 0 or p < 0 or p > n:
                raise ValueError(
                    f"invalid verification counts returned: passed={p}, total={n}"
                )
            total_pass += p; total_checks += n
            results[name] = (p, n, 'ok')
        except Exception as e:
            import traceback
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            had_task_error = True
            total_checks += 1
            results[name] = (0, 1, str(e))

    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    for name, (p, n, status) in results.items():
        tag = PASS if (status == 'ok' and p == n) else (WARN if status == 'ok' else FAIL)
        print(f"  {tag}  {name}: {p}/{n} checks  [{status}]")
    print(f"\n  Overall: {total_pass}/{total_checks} checks passed")
    if not had_task_error and total_pass == total_checks:
        print("  ALL CHECKS PASSED — code reproduces the manuscript's reported values")
        return 0
    else:
        n_fail = total_checks - total_pass
        print(f"  {n_fail} checks outside tolerance — review above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
