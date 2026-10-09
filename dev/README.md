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
  plan or a result, not committed raw.
- `private/` is gitignored: maintainer notes that are not published. A
  tracked record may point to one by its path, but never restates it.

A file or directory is created with its first entry.

## Index

- [plans/pyibs-1.5.md](plans/pyibs-1.5.md): PyIBS 1.5, from the 0.1.0 code
  to a release on the level of PyBADS 1.5 and PyVBMC 1.5.
