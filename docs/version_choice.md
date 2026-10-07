# Public branch choice

| Candidate | Scope | Portfolio decision |
|---|---|---|
| origin/main, d252c0d | v9 reviewer reproducibility package | Preserve unchanged |
| public tag v9-review | Earlier snapshot | Preserve unchanged |
| origin/submission-readiness-fix, dcab18c | V15 corrected Morris and canonical MILP PRCC | Base of portfolio-cleanup |
| local V12-final | Earlier correction package; includes private manuscript material | Do not copy private package |
| local submission-readiness | Matches public dcab18c V15 code lineage | No wholesale local overwrite |
| local V15 directory | Manuscript-associated artifacts, not a standalone repository | Exclude internal artifacts |
| local V22.1 code package | Later campaign-level launch-accounting model | Exclude; incompatible scientific/version scope |

The selected model was already publicly available. Cleanup adds documentation and
verified execution evidence without importing newer unpublished content. No merge
or force-push is performed. Main and original author history remain available.
The local source evidence does not resolve which later revision the research team
intends as the final manuscript deposit. Therefore this branch does not declare
itself authoritative for V22.1 or for an accepted journal article.
