#!/usr/bin/env python3
"""Reproduce the manuscript LHS-MILP PRCC analyses.

The default run uses 256 Latin-hypercube samples, seed 2026, seven parameters
varied by +/-20%, and the canonical deterministic MILP at T=137.  Passing
``--horizon 156`` writes supplementary files with a ``_T156`` suffix, leaving
the T=137 deposition untouched.
"""

import argparse
import logging
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, qmc, rankdata, t as t_dist

import task1_q1_milp as model


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"
for directory in (DATA_DIR, FIGURE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_SEED = 2026
DEFAULT_SAMPLES = 256
DEFAULT_HORIZON = 137
RELATIVE_RANGE = 0.20

PARAMETERS = [
    ("c_R_base", 6.5e6, r"Nominal rocket cost $c_R^{\mathrm{base}}$"),
    ("gamma", 1.2, r"Cost elasticity $\gamma$"),
    ("t_mid", 50.0, r"Technology midpoint $t_{\mathrm{mid}}$"),
    ("P_R", 138.0, r"Rocket payload $P_R$"),
    ("N_max", 5000.0, r"Max launches $N_{\max}$"),
    ("k_growth", 0.12, r"Growth rate $k$"),
    ("C_fixed", 100e9, r"Elevator fixed cost $C_{\mathrm{fixed}}$"),
]
KEYS = [item[0] for item in PARAMETERS]
BASELINES = np.array([item[1] for item in PARAMETERS], dtype=float)
LOWER = BASELINES * (1.0 - RELATIVE_RANGE)
UPPER = BASELINES * (1.0 + RELATIVE_RANGE)

SAMPLE_COLUMNS = (
    ["sample_id", "seed", "T_fixed"]
    + [f"u_{key}" for key in KEYS]
    + KEYS
    + [
        "solver_status",
        "solver_code",
        "solver_message",
        "objective",
        "objective_TUSD",
        "infeasible",
        "excluded",
        "exclusion_reason",
    ]
)


def output_suffix(horizon):
    return "" if int(horizon) == DEFAULT_HORIZON else f"_T{int(horizon)}"


def output_paths(horizon):
    suffix = output_suffix(horizon)
    return {
        "samples": DATA_DIR / f"prcc_lhs_milp_samples{suffix}.csv",
        "results": DATA_DIR / f"prcc_results{suffix}.csv",
        "figure": FIGURE_DIR / f"Figure11_PRCC_corrected{suffix}.png",
        "log": LOG_DIR / f"prcc_run_log{suffix}.txt",
    }


def configure_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parameters_from_values(values):
    params = model.clone_params()
    params.C_R_BASE = float(values[0])
    params.GAMMA = float(values[1])
    params.T_MID_E = params.T_MID_R = params.T_MID_ISP = float(values[2])
    params.P_R = float(values[3])
    params.N_LAUNCH_MAX = float(values[4])
    params.GROWTH_K = float(values[5])
    params.FIXED_COST = float(values[6])
    return params


def _solver_fields(result, feasible):
    if not feasible:
        return {
            "solver_status": "infeasible_or_failed",
            "solver_code": "",
            "solver_message": "MILP returned no feasible solution",
        }
    code = int(result.get("solver_code", 0))
    return {
        "solver_status": result.get("solver_status", "optimal" if code == 0 else f"status_{code}"),
        "solver_code": code,
        "solver_message": result.get("solver_message", "Optimization terminated successfully."),
    }


def generate_samples(n_samples, seed, horizon):
    if n_samples <= len(PARAMETERS) + 2:
        raise ValueError("sample count must exceed number of parameters + 2")
    unit_samples = qmc.LatinHypercube(d=len(PARAMETERS), seed=seed).random(n_samples)
    sampled_values = qmc.scale(unit_samples, LOWER, UPPER)
    rows = []
    for sample_id, (unit_row, value_row) in enumerate(zip(unit_samples, sampled_values)):
        result = model.solve_milp(horizon, parameters_from_values(value_row))
        feasible = result is not None and np.isfinite(result["total_cost"])
        row = {"sample_id": sample_id, "seed": seed, "T_fixed": horizon}
        row.update({f"u_{key}": float(value) for key, value in zip(KEYS, unit_row)})
        row.update({key: float(value) for key, value in zip(KEYS, value_row)})
        row.update(_solver_fields(result, feasible))
        row.update(
            {
                "objective": float(result["total_cost"]) if feasible else np.nan,
                "objective_TUSD": float(result["total_cost"] / 1e12) if feasible else np.nan,
                "infeasible": not feasible,
                "excluded": not feasible,
                "exclusion_reason": "" if feasible else "MILP returned no feasible solution at fixed T",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def residualize(target, controls):
    design = np.column_stack([np.ones(len(target)), controls])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients


def calculate_prcc(samples):
    included = samples.loc[~samples["excluded"]].copy()
    if len(included) <= len(PARAMETERS) + 2:
        raise RuntimeError("too few feasible samples for seven-parameter PRCC")
    ranked_inputs = np.column_stack(
        [rankdata(included[key].to_numpy()) for key in KEYS]
    )
    ranked_output = rankdata(included["objective"].to_numpy())
    n = len(included)
    total_inputs = len(PARAMETERS)
    df = n - total_inputs - 1
    rows = []
    for index, (key, baseline, label) in enumerate(PARAMETERS):
        controls = np.delete(ranked_inputs, index, axis=1)
        residual_input = residualize(ranked_inputs[:, index], controls)
        residual_output = residualize(ranked_output, controls)
        coefficient, _ = pearsonr(residual_input, residual_output)
        t_stat = coefficient * np.sqrt(df / (1.0 - coefficient**2))
        p_value = 2.0 * t_dist.sf(np.abs(t_stat), df)
        rows.append(
            {
                "parameter": key,
                "label": label,
                "baseline": baseline,
                "lower": baseline * (1.0 - RELATIVE_RANGE),
                "upper": baseline * (1.0 + RELATIVE_RANGE),
                "prcc": float(coefficient),
                "p_value": float(p_value),
                "n_included": n,
                "n_excluded": int(len(samples) - n),
                "T_fixed": int(included["T_fixed"].iloc[0]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("prcc", key=lambda values: values.abs(), ascending=False)
        .reset_index(drop=True)
    )


def results_for_deposit(results, horizon):
    if int(horizon) == DEFAULT_HORIZON:
        return results
    return results[["parameter", "prcc", "p_value", "n_included", "T_fixed"]]


def make_figure(results, output):
    plot_data = results.sort_values("prcc").reset_index(drop=True)
    colors = ["#27AE60" if value > 0 else "#E94F37" for value in plot_data["prcc"]]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    bars = axis.barh(
        np.arange(len(plot_data)),
        plot_data["prcc"],
        color=colors,
        alpha=0.86,
        edgecolor="white",
    )
    axis.set_yticks(np.arange(len(plot_data)))
    axis.set_yticklabels(plot_data["label"])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlim(-1.05, 1.05)
    axis.set_xlabel("Partial rank correlation coefficient (PRCC)")
    included = int(results["n_included"].iloc[0])
    excluded = int(results["n_excluded"].iloc[0])
    horizon = int(results["T_fixed"].iloc[0])
    axis.set_title(
        f"LHS-MILP PRCC at fixed horizon T={horizon}\n"
        f"{included}/{included + excluded} feasible samples, seven parameters, +/-20% ranges"
    )
    for bar, coefficient, p_value in zip(bars, plot_data["prcc"], plot_data["p_value"]):
        text = f"{coefficient:+.3f}\np={p_value:.2g}"
        if abs(coefficient) >= 0.25:
            x = coefficient - 0.025 if coefficient > 0 else coefficient + 0.025
            horizontal = "right" if coefficient > 0 else "left"
            color = "white"
        else:
            x = coefficient + (0.025 if coefficient >= 0 else -0.025)
            horizontal = "left" if coefficient >= 0 else "right"
            color = "black"
        axis.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha=horizontal,
            color=color,
            fontsize=8.5,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def run(n_samples=DEFAULT_SAMPLES, seed=DEFAULT_SEED, horizon=DEFAULT_HORIZON):
    paths = output_paths(horizon)
    configure_logging(paths["log"])
    logging.info("LHS-MILP PRCC sensitivity analysis")
    logging.info("Python: %s", platform.python_version())
    logging.info(
        "Versions: numpy=%s scipy=%s pandas=%s matplotlib=%s",
        np.__version__,
        scipy.__version__,
        pd.__version__,
        matplotlib.__version__,
    )
    logging.info(
        "Design: samples=%d seed=%d parameters=%d range=+/-20%% fixed_T=%d",
        n_samples,
        seed,
        len(PARAMETERS),
        horizon,
    )
    samples = generate_samples(n_samples, seed, horizon)
    results = calculate_prcc(samples)
    samples.to_csv(paths["samples"], index=False, lineterminator="\n")
    results_for_deposit(results, horizon).to_csv(
        paths["results"], index=False, lineterminator="\n"
    )
    make_figure(results, paths["figure"])

    included = int((~samples["excluded"]).sum())
    excluded = int(samples["excluded"].sum())
    logging.info("Samples: included=%d excluded=%d", included, excluded)
    logging.info("Results:")
    for row in results.itertuples(index=False):
        logging.info("  %-10s PRCC=%+.6f p=%.6g", row.parameter, row.prcc, row.p_value)
    logging.info("Outputs:")
    for key in ("samples", "results", "figure", "log"):
        logging.info("  %s", paths[key].relative_to(ROOT))
    logging.info("Run complete")
    return {"samples": samples, "results": results, "paths": paths}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LHS-MILP PRCC analysis")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.samples, arguments.seed, arguments.horizon)
