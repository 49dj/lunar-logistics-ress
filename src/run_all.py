#!/usr/bin/env python3
"""Fail-closed V15 reproducibility harness.

Run from any working directory with ``python src/run_all.py``. The harness
checks the deterministic and robust MILPs, the robust feasibility boundary,
Figure 8, corrected Figure 10 Morris screening, and the T=137 and T=156
LHS-MILP-PRCC analyses. CSV comparisons are parsed and numeric, so LF/CRLF
differences do not cause false failures.
"""

import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT_SRC = Path(__file__).resolve().parent
ROOT = ROOT_SRC.parent
sys.path.insert(0, str(ROOT_SRC))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _safe_console():
    """Prevent non-ASCII diagnostics in legacy modules from crashing GBK."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def check(name, condition, detail=""):
    ok = bool(condition)
    suffix = f" - {detail}" if detail else ""
    print(f"    {PASS if ok else FAIL} {name}{suffix}")
    return ok


def close(name, got, expected, abs_tol=0.0, rel_tol=0.0):
    ok = math.isclose(float(got), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)
    return check(name, ok, f"got={got}, expected={expected}")


def _frames_numerically_equal(before, after, rtol=1e-10, atol=1e-9):
    """Line-ending-independent comparison of parsed CSV content."""
    if before is None or list(before.columns) != list(after.columns) or before.shape != after.shape:
        return False
    for column in before.columns:
        left = before[column]
        right = after[column]
        try:
            left_num = pd.to_numeric(left)
            right_num = pd.to_numeric(right)
            if not np.allclose(left_num, right_num, rtol=rtol, atol=atol, equal_nan=True):
                return False
        except (TypeError, ValueError):
            if not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
                return False
    return True


def _read_csv_if_present(path):
    return pd.read_csv(path) if path.exists() else None


def _scan_frame_is_valid(df):
    required = ["T", "feasible", "total_cost_T"]
    if list(df.columns) != required or len(df) != 38:
        return False
    horizons = pd.to_numeric(df["T"], errors="coerce").tolist()
    flags = df["feasible"].astype(str).str.lower().tolist()
    if horizons != list(range(100, 138)) or flags != ["false"] * 37 + ["true"]:
        return False
    cost = float(df.loc[pd.to_numeric(df["T"]) == 137, "total_cost_T"].iloc[0])
    return math.isclose(cost, 289.918795, abs_tol=5e-7, rel_tol=0.0)


def run_deterministic():
    import task1_q1_milp as t1

    print("\nDETERMINISTIC MILP")
    before1 = _read_csv_if_present(ROOT / "data" / "pareto_nh1.csv")
    before3 = _read_csv_if_present(ROOT / "data" / "pareto_nh3.csv")
    result = t1.run_optimization(verbose=False)
    checks = [
        close("NH=1 feasible minimum", result["T_min1"], 119),
        close("NH=1 knee horizon", result["knee1"]["T"], 137),
        close("NH=1 knee cost (TUSD)", result["knee1"]["cost_T"], 210.71, abs_tol=0.02),
        close("NH=1 elevator share", result["knee1"]["frac_elevator"], 0.488, abs_tol=0.002),
        close("NH=3 knee horizon", result["knee3"]["T"], 93),
        close("NH=3 knee cost (TUSD)", result["knee3"]["cost_T"], 171.9, abs_tol=0.03),
        close("NH=3 elevator share", result["knee3"]["frac_elevator"], 0.755, abs_tol=0.002),
    ]
    after1 = pd.read_csv(ROOT / "data" / "pareto_nh1.csv")
    after3 = pd.read_csv(ROOT / "data" / "pareto_nh3.csv")
    checks.append(check("pareto_nh1.csv numeric regeneration", _frames_numerically_equal(before1, after1)))
    checks.append(check("pareto_nh3.csv numeric regeneration", _frames_numerically_equal(before3, after3)))
    return sum(checks), len(checks)


def run_robust():
    import task2b_robust_milp as robust

    print("\nROBUST MILP, FEASIBILITY SCAN, AND FIGURE 8")
    csv_path = ROOT / "outputs" / "pareto_robust.csv"
    before = _read_csv_if_present(csv_path)
    result = robust.run_robust_analysis(write_outputs=True, verbose=False)
    checks = [
        close("first robust feasible horizon", result["T_min"], 137),
        close("robust knee horizon", result["knee_T"], 156),
        close("robust total cost (TUSD)", result["total_cost"] / 1e12, 236.65, abs_tol=0.02),
        close("robust elevator share", result["frac_elevator"], 0.501, abs_tol=0.002),
        close("robust premium percent", result["premium_pct"], 12.3, abs_tol=0.08),
        close("failure cost (TUSD)", result["c_fail"] / 1e12, 1.34, abs_tol=0.02),
        close("replacement cost (TUSD)", result["c_repl"] / 1e12, 11.87, abs_tol=0.02),
        close("rare-event buffer (TUSD)", result["c_rare"] / 1e12, 15.59, abs_tol=0.02),
    ]
    after = pd.read_csv(csv_path)
    checks.append(check("pareto_robust.csv numeric regeneration", _frames_numerically_equal(before, after)))
    scan_path = ROOT / "logs" / "robust_feasibility_scan.csv"
    robust.feasibility_scan(T_lo=100, T_hi=137, out_csv=scan_path)
    scan = pd.read_csv(scan_path)
    checks.append(check("T=100..136 infeasible and T=137 first feasible", _scan_frame_is_valid(scan)))
    checks.append(check("Figure 8 generated", (ROOT / "figures" / "figure8_robust_comparison.png").exists()))
    return sum(checks), len(checks)


def run_morris():
    import fig10_morris_screening as morris

    print("\nFIGURE 10: CORRECTED SEVEN-PARAMETER MORRIS")
    summary, raw = morris.run_morris(verbose=False)
    summary.to_csv(ROOT / "data" / "morris_screening.csv", index=False, lineterminator="\n")
    raw.to_csv(ROOT / "data" / "morris_elementary_effects.csv", index=False, lineterminator="\n")
    morris.make_figure(summary)
    by_param = summary.set_index("param")
    required_raw = {
        "trajectory_id", "parameter", "x0", "x1", "y0", "y1", "realized_step",
        "elementary_effect", "feasible_y0", "feasible_y1", "skipped_reason",
    }
    ranked = summary["param"].tolist()
    skipped = raw[raw["skipped_reason"].fillna("").astype(str) != ""]
    checks = [
        check("seven-parameter Morris ranking", ranked == ["c_R_base", "gamma", "t_mid", "P_R", "N_max", "k_growth", "C_fixed"], str(ranked)),
        close("c_R_base mu_star", by_param.loc["c_R_base", "mu_star"], 6.101185e13, rel_tol=2e-5),
        close("gamma mu_star", by_param.loc["gamma", "mu_star"], 4.806236e13, rel_tol=2e-5),
        close("c_R_base sigma/mu_star", by_param.loc["c_R_base", "sigma_over_mu_star"], 0.123, abs_tol=0.002),
        close("gamma sigma/mu_star", by_param.loc["gamma", "sigma_over_mu_star"], 0.174, abs_tol=0.002),
        check("raw elementary-effect schema", required_raw.issubset(raw.columns)),
        check("all skipped effects carry a reason", len(skipped) == 18 and skipped["skipped_reason"].notna().all(), f"skipped={len(skipped)}"),
        check("all used effects use nonzero realized steps", (raw.loc[raw["skipped_reason"].fillna("") == "", "realized_step"].abs() > 0).all()),
        check("corrected Figure 10 generated", (ROOT / "figures" / "Figure10_Morris_screening_corrected.png").exists()),
    ]
    return sum(checks), len(checks)


def run_prcc_analyses():
    import run_prcc as prcc

    print("\nFIGURE 11: LHS-MILP-PRCC AT T=137 AND T=156")
    primary = prcc.run()
    supplementary = prcc.run(horizon=156)
    samples137 = primary["samples"]
    results137 = primary["results"]
    samples156 = supplementary["samples"]
    results156 = supplementary["results"]
    by137 = results137.set_index("parameter")
    by156 = results156.set_index("parameter")
    excluded = samples137[samples137["excluded"]]
    deposited156 = pd.read_csv(prcc.output_paths(156)["results"])
    checks = [
        check("T=137 Latin-hypercube sample count", len(samples137) == prcc.DEFAULT_SAMPLES),
        check("T=137 sample schema", list(samples137.columns) == prcc.SAMPLE_COLUMNS),
        check("T=137 feasible sample count", (~samples137["excluded"]).sum() == 251),
        check("T=137 excluded rows documented", excluded["exclusion_reason"].fillna("").str.len().gt(0).all()),
        close("T=137 c_R_base PRCC", by137.loc["c_R_base", "prcc"], 0.937491871967434, abs_tol=2e-12),
        close("T=137 gamma PRCC", by137.loc["gamma", "prcc"], -0.9030656874752124, abs_tol=2e-12),
        close("T=137 t_mid PRCC", by137.loc["t_mid", "prcc"], 0.8225360281155898, abs_tol=2e-12),
        close("T=137 c_R_base p-value", by137.loc["c_R_base", "p_value"], 2.1999313710542993e-113, rel_tol=2e-10),
        check("T=156 Latin-hypercube sample count", len(samples156) == prcc.DEFAULT_SAMPLES),
        check("T=156 all samples feasible", (~samples156["excluded"]).sum() == 256),
        close("T=156 c_R_base PRCC", by156.loc["c_R_base", "prcc"], 0.9789653302952617, abs_tol=2e-12),
        close("T=156 gamma PRCC", by156.loc["gamma", "prcc"], -0.9686140650250336, abs_tol=2e-12),
        close("T=156 t_mid PRCC", by156.loc["t_mid", "prcc"], 0.8313600960428515, abs_tol=2e-12),
        check(
            "T=156 deposited result schema",
            list(deposited156.columns) == ["parameter", "prcc", "p_value", "n_included", "T_fixed"],
        ),
        check("T=137 outputs retained", pd.read_csv(prcc.output_paths(137)["samples"])["T_fixed"].eq(137).all()),
        check("T=156 outputs separated", pd.read_csv(prcc.output_paths(156)["samples"])["T_fixed"].eq(156).all()),
        check("T=137 Figure 11 generated", prcc.output_paths(137)["figure"].exists()),
        check("T=156 supplementary figure generated", prcc.output_paths(156)["figure"].exists()),
    ]
    return sum(checks), len(checks)


def run_fig9():
    import fig9_frontiers

    print("\nFIGURE 9")
    output = Path(fig9_frontiers.main())
    ok = check("frontier comparison generated", output.exists())
    return int(ok), 1


def main(task_specs=None):
    _safe_console()
    os.chdir(ROOT)
    if task_specs is None:
        task_specs = [
            (run_deterministic, "Deterministic MILP"),
            (run_robust, "Robust MILP + scan + Figure 8"),
            (run_fig9, "Figure 9"),
            (run_morris, "Figure 10 Morris"),
            (run_prcc_analyses, "Figure 11 PRCC at T=137 and T=156"),
        ]
    print("V15 REPRODUCIBILITY VERIFICATION")
    total_pass = 0
    total_checks = 0
    results = {}
    started = time.time()
    for task, name in task_specs:
        try:
            passed, count = task()
            if count <= 0 or passed < 0 or passed > count:
                raise ValueError(f"invalid check counts: {passed}/{count}")
            total_pass += passed
            total_checks += count
            results[name] = (passed, count, "ok")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            total_checks += 1
            results[name] = (0, 1, f"error: {exc}")

    print("\nFINAL SUMMARY")
    for name, (passed, count, status) in results.items():
        ok = status == "ok" and passed == count
        print(f"  {PASS if ok else FAIL} {name}: {passed}/{count} ({status})")
    print(f"  Overall: {total_pass}/{total_checks}; elapsed={time.time() - started:.1f}s")
    if total_checks > 0 and total_pass == total_checks:
        print("  ALL V15 CHECKS PASSED")
        return 0
    print(f"  VERIFICATION FAILED: {total_checks - total_pass} check(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
