# Reliability-informed Earth–Moon logistics

A research code package for studying the cost, completion time and reliability
of Earth–Moon infrastructure logistics under launch and space-elevator disruptions.
The public model combines deterministic/robust MILP planning, reliability inputs,
Morris screening and Latin-hypercube MILP partial rank correlation analysis.

This is collaborative research. Yizhe Hu participates as a coauthor; the portfolio
does not claim sole authorship or infer a detailed individual contribution from
local filenames. Original Git history, upstream relationship and contributor
attribution are preserved.

## Public version and publication stage

This `portfolio-cleanup` branch builds on the already-public
`submission-readiness-fix` V15 package (commit `dcab18c`). The repository's `main`
remains the earlier v9 review package. Local V12 and V22.1 candidates were compared
at the code-package/version-documentation level. Later revisions change the
planning/costing scope and are not substituted for the public V15 model.

The associated manuscript is titled *Reliability-informed robust optimisation of
Earth-Moon infrastructure logistics under launch and space-elevator disruptions*.
This code package does not establish acceptance or publication; no verified journal
DOI is asserted. Manuscript files, author/reviewer material, private submission
packages and Paper 2 work are not added. See [version choice](docs/version_choice.md).

## Reproduce

Python 3.12 with the existing pinned scientific dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run_all.py
```

The fail-closed master command checks deterministic and robust MILPs, feasibility,
Figures 8–11, corrected Morris screening and both T=137/T=156 PRCC analyses.
Its documented success line is `ALL V15 CHECKS PASSED`. New verification evidence
and any limitations are recorded in [verification](docs/verification.md).

For individual commands, equations/assumptions, numerical reference values, full
artifact mapping and sensitivity settings, see [V15 reproduction](docs/v15_reproduction.md).

## Structure

`src/` contains the canonical model and verification modules; `tests/` guards the
verification harness and parameter isolation. `data/` and the retained robust CSV
are generated model outputs, not private empirical datasets. `figures/` retains
useful existing model figures. `logs/` contains public-safe model evidence; fresh
machine-specific execution transcripts stay outside the repository.

## Assumptions and limits

Rocket transport is costed per delivered tonne; unused launch capacity incurs no
per-attempt charge. Launch count constrains payload capacity and slot availability
but has no direct cost coefficient. This V15 assumption differs from later
campaign-accounting revisions. The legacy five-parameter sensitivity functions
are retained for API compatibility only and are not the publication figure path.

Regression agreement verifies the selected package's stated reference quantities,
not empirical reliability, real mission cost or every manuscript artifact.
Some conceptual/analytic figures have no standalone generating module. PRCC is
conditional on retained feasible samples and the specified sampling box. Cite the
repository branch and exact commit. Licensing was not broadened on behalf of
collaborators; see [rights](docs/rights.md).
