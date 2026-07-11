"""Regression tests for the master verification harness."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_all  # noqa: E402


class VerificationSummaryTests(unittest.TestCase):
    def test_missing_canonical_csv_is_not_a_byte_identity_pass(self):
        self.assertTrue(
            hasattr(run_all, "_canonical_bytes_match"),
            "run_all must expose the canonical-byte gate for regression tests",
        )
        self.assertFalse(run_all._canonical_bytes_match(None, "generated-sha"))
        self.assertTrue(run_all._canonical_bytes_match("same-sha", "same-sha"))

    def test_scan_gate_rejects_corrupted_deposited_evidence(self):
        self.assertTrue(
            hasattr(run_all, "_scan_frame_is_valid"),
            "run_all must expose a content validator for the deposited scan",
        )
        valid = pd.DataFrame(
            {
                "T": list(range(100, 138)),
                "feasible": [False] * 37 + [True],
                "total_cost_T": [float("nan")] * 37 + [289.918795],
            }
        )
        self.assertTrue(run_all._scan_frame_is_valid(valid))

        corrupted = valid.copy()
        corrupted.loc[0, "feasible"] = True
        self.assertFalse(run_all._scan_frame_is_valid(corrupted))

        wrong_cost = valid.copy()
        wrong_cost.loc[37, "total_cost_T"] = 0.0
        self.assertFalse(run_all._scan_frame_is_valid(wrong_cost))

    def test_task_exception_cannot_be_reported_as_success(self):
        """A crashed verification task must force a non-zero result."""
        self.assertTrue(
            hasattr(run_all, "main"),
            "run_all must expose a testable main(task_specs=...) entry point",
        )

        def passing_task():
            return 1, 1

        def crashing_task():
            raise RuntimeError("synthetic verification failure")

        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            exit_code = run_all.main(
                task_specs=[
                    (passing_task, "passing task"),
                    (crashing_task, "crashing task"),
                ]
            )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL]  crashing task", rendered)
        self.assertNotIn("ALL CHECKS PASSED", rendered)

    def test_zero_check_task_cannot_be_reported_as_success(self):
        """A task that performs no checks is an invalid verification result."""
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            exit_code = run_all.main(
                task_specs=[(lambda: (0, 0), "empty task")]
            )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("[FAIL]  empty task", rendered)
        self.assertNotIn("ALL CHECKS PASSED", rendered)


if __name__ == "__main__":
    unittest.main()
