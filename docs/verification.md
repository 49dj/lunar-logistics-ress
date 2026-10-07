# Portfolio verification, 7 October 2026

Executed on Windows with Python 3.12 and the existing pinned dependencies:
NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.2, Matplotlib 3.10.5.

| Command | Observed result |
|---|---|
| `python -m unittest discover -s tests -v` | 8 tests passed |
| `python src/run_all.py` | 48/48 checks; exit 0; about 122 seconds |

Full workflow: deterministic MILP 9/9; robust MILP, scan and Figure 8 11/11;
Figure 9 1/1; corrected Morris 9/9; T=137/T=156 PRCC 18/18.
Final line: `ALL V15 CHECKS PASSED`.

The regenerated PRCC coefficients differed from the deposited decimal text by
roughly 1e-16; corresponding p-values differed in their last floating-point
digits. No parameter, tolerance or equation was changed to obtain agreement.
Cross-platform numerical agreement is not byte identity. Previously deposited
generated CSVs, figures and logs are retained, rather than replacing them with
platform-dependent float formatting and image rendering. Fresh transcripts remain
local; they are not uploaded as machine-specific paths or environment logs.

These checks validate this V15 package's embedded regression criteria. They do
not establish real mission fidelity, independent empirical parameter validation,
accepted-manuscript status or reproduction of later V22.1 revisions.
