"""Regression tests for the V15 Morris and PRCC workflows."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import fig10_morris_screening as morris  # noqa: E402
import run_prcc as prcc  # noqa: E402


class SubmissionSensitivityTests(unittest.TestCase):
    def test_morris_uses_realized_step_and_records_skips(self):
        summary, raw = morris.run_morris(seed=2026, n_traj=3, verbose=False)
        used = raw[raw["skipped_reason"].fillna("") == ""]
        recomputed = (used["y1"] - used["y0"]) / used["realized_step"]
        self.assertTrue(np.allclose(recomputed, used["elementary_effect"]))
        self.assertEqual(set(summary["param"]), set(morris.KEYS))

    def test_deposited_prcc_recomputes_with_df_243(self):
        samples = pd.read_csv(ROOT / "data" / "prcc_lhs_milp_samples.csv")
        results = prcc.calculate_prcc(samples).set_index("parameter")
        self.assertEqual(list(samples.columns), prcc.SAMPLE_COLUMNS)
        self.assertEqual(int((~samples["excluded"]).sum()), 251)
        self.assertAlmostEqual(results.loc["c_R_base", "prcc"], 0.937491871967434, places=13)
        self.assertAlmostEqual(results.loc["gamma", "prcc"], -0.9030656874752124, places=13)
        self.assertTrue(np.isclose(results.loc["c_R_base", "p_value"], 2.1999313710542993e-113, rtol=2e-10))

    def test_horizon_specific_outputs_do_not_collide(self):
        primary = prcc.output_paths(137)
        supplementary = prcc.output_paths(156)
        self.assertEqual(primary["results"].name, "prcc_results.csv")
        self.assertEqual(supplementary["results"].name, "prcc_results_T156.csv")
        self.assertEqual(supplementary["samples"].name, "prcc_lhs_milp_samples_T156.csv")
        self.assertEqual(supplementary["figure"].name, "Figure11_PRCC_corrected_T156.png")
        self.assertNotEqual(primary["results"], supplementary["results"])


if __name__ == "__main__":
    unittest.main()
