"""Regression coverage for Task-1 sensitivity parameter isolation."""

from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import task1_q1_milp as task1  # noqa: E402


class ParameterIsolationTests(unittest.TestCase):
    def test_sensitivity_helpers_do_not_mutate_global_parameters(self):
        keys = (
            "GAMMA",
            "C_R_BASE",
            "P_R",
            "C_E_OP",
            "CAP_E_MAX",
            "ISP_MAX",
            "GROWTH_K",
        )
        before = {key: getattr(task1.P, key) for key in keys}

        task1.morris_screening(n_trials=2, seed=0)
        task1.prcc_analysis(n_samples=20, seed=1)
        task1.monte_carlo_pareto(n_mc=1, seed=42)

        after = {key: getattr(task1.P, key) for key in keys}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
