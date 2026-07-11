#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig9_frontiers.py — Manuscript Figure 9: deterministic vs reliability-adjusted
cost-time frontiers, with both knees and the minimum feasible robust horizon.

Inputs  (repository root):
  data/pareto_nh1.csv        deterministic frontier (task1_q1_milp.py, manuscript §4.4)
  outputs/pareto_robust.csv  reliability-adjusted frontier (task2b_robust_milp.py, §6.1)

Output:
  figures/figure9_frontiers.png

Knees are re-derived from the CSVs with the same maximum-perpendicular-distance
criterion used throughout the manuscript (task1_q1_milp.find_knee), so the
figure is fully reproducible from the deposited datasets alone.

Run (from the repository root):
  python src/fig9_frontiers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_q1_milp as t1   # STYLE, CB palette, find_knee

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(t1.STYLE)
CB = t1.CB

ROOT = Path(__file__).resolve().parent.parent


def main():
    det = pd.read_csv(ROOT / "data" / "pareto_nh1.csv")
    rob = pd.read_csv(ROOT / "outputs" / "pareto_robust.csv")

    knee_d = t1.find_knee(det)
    knee_r = t1.find_knee(rob)
    T_min_robust = int(rob["T"].min())

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(det["T"], det["cost_T"], color=CB["blue"], lw=2.5,
            label="Deterministic frontier")
    ax.plot(rob["T"], rob["cost_T"], color=CB["red"], lw=2.5,
            label="Reliability-adjusted frontier")

    ax.scatter([knee_d["T"]], [knee_d["cost_T"]], s=120, color=CB["blue"], zorder=5)
    ax.scatter([knee_r["T"]], [knee_r["cost_T"]], s=140, color=CB["red"],
               marker="D", zorder=5)

    ax.axvline(T_min_robust, color=CB["gray"], ls="--", lw=1.5, zorder=1)

    ax.annotate(f"Deterministic knee\n{int(knee_d['T'])} yr, "
                f"${knee_d['cost_T']:.1f}T",
                xy=(knee_d["T"] + 1.5, knee_d["cost_T"] - 3),
                xytext=(knee_d["T"] + 30, knee_d["cost_T"] - 55),
                fontsize=11, color=CB["blue"],
                arrowprops=dict(arrowstyle="->", color=CB["blue"], lw=1.2))
    ax.annotate(f"Robust knee\n{int(knee_r['T'])} yr, ${knee_r['cost_T']:.1f}T",
                xy=(knee_r["T"] + 1.5, knee_r["cost_T"] + 1.5),
                xytext=(knee_r["T"] + 28, knee_r["cost_T"] + 22),
                fontsize=11, color=CB["red"],
                arrowprops=dict(arrowstyle="->", color=CB["red"], lw=1.2))
    ax.annotate(f"Minimum feasible robust horizon = {T_min_robust} yr\n"
                f"(= deterministic knee → zero schedule margin)",
                xy=(T_min_robust + 0.5, 271),
                xytext=(T_min_robust + 7, 284),
                fontsize=10.5, color=CB["navy"],
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.1))

    ax.set_xlabel("Programme duration $T$ [yr]")
    ax.set_ylabel("Total cost [\\$T]")
    ax.set_title("Deterministic vs reliability-adjusted cost–time frontiers")
    ax.legend(loc="lower left")

    out = ROOT / "figures" / "figure9_frontiers.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  figure -> {out}")
    print(f"  deterministic knee: T*={int(knee_d['T'])} yr, ${knee_d['cost_T']:.2f}T")
    print(f"  robust knee:        T*={int(knee_r['T'])} yr, ${knee_r['cost_T']:.2f}T")
    print(f"  minimum feasible robust horizon: {T_min_robust} yr")
    return out


if __name__ == "__main__":
    main()
