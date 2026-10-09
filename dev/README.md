# Developer notes

`dev/` holds the maintainer records: plans, findings and the evidence they
cite, and the tooling that produced it. They are not user documentation.

- `TODO.md` lists the open work that no plan covers.
- `plans/` holds implementation plans, with their decisions and worklogs.
  A plan is kept current while its work is open, updated in place, and
  retained afterwards.
- `results/` holds findings and study writeups, named
  `YYYY-MM-DD-<slug>.md`, with no further subdirectories. Each states its
  question, its main results and a short interpretation, and links to its
  evidence.
- `experiments/` holds the machine-readable evidence that a result cites,
  one directory per study, `<slug>_<YYYYMMDD>/`. Its `README.md` gives the
  commands and seeds that produced the evidence; the PyIBS commit
  (`git rev-parse HEAD`) and whether the tree was clean (`git status
  --porcelain` empty); the Python, NumPy and SciPy versions and the
  platform; and the versions, or the commits of checkouts, of PyBADS,
  PyVBMC, gpyreg and MATLAB IBS when they ran. The scripts kept with the
  evidence are frozen with it.
- `scripts/` holds tooling that is not part of the package or its tests.
  Run it from the repository root with the venv, as
  `.venv/Scripts/python.exe dev/scripts/<name>.py` (`.venv/bin/python`
  elsewhere). Its raw output goes
  under `dev/scripts/runs/`, which git ignores and a fresh clone creates
  (`mkdir -p dev/scripts/runs`): a result that matters is summarized in a
  plan or a result, not committed raw. `scripts/octave/` holds the files
  that let MATLAB `ibslike.m` run under GNU Octave (`AGENTS.md`, "Sibling
  repositories").
- `private/` is gitignored: maintainer notes that are not published. A
  tracked record may point to one by its path, but never restates it.

A file or directory is created with its first entry.

## Index

- [plans/pyibs-1.5.md](plans/pyibs-1.5.md): PyIBS 1.5, from the 0.1.0 code
  to a release on the level of PyBADS 1.5 and PyVBMC 1.5.
- [results/2026-10-09-port-review.md](results/2026-10-09-port-review.md):
  the review of PyIBS against MATLAB `ibslike.m` and the IBS paper, with
  its ledger of findings; its evidence is
  [experiments/port-review_20261009/](experiments/port-review_20261009/).
- [results/2026-10-09-validation.md](results/2026-10-09-validation.md):
  the statistical validation of PyIBS 1.5's estimates against exact
  log-likelihoods, its runs with PyBADS and PyVBMC, and its timings
  against PyIBS 0.1.0; its evidence is
  [experiments/validation_20261009/](experiments/validation_20261009/).
