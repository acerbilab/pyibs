# Statistical validation, PyBADS and PyVBMC, and timing of PyIBS 1.5

The evidence that
[`dev/results/2026-10-09-validation.md`](../../results/2026-10-09-validation.md)
cites: the statistical validation of PyIBS's estimates against exact
log-likelihoods, the integration tests with PyBADS and PyVBMC, and the
timings of PyIBS 1.5 against PyIBS 0.1.0 (Phase 3 of
[`dev/plans/pyibs-1.5.md`](../../plans/pyibs-1.5.md)).

## Provenance

- Each output was produced at a commit of PyIBS with a clean tree (`git
  status --porcelain` empty, which the JSON outputs record in `meta` and
  which was checked before the other runs):
  - `validation.json` and `validation.txt`: the estimates were drawn at
    `c76b5ac9239b2f507f631291619c031fbad820b9` (`meta`), and their
    statistics and verdicts computed from the saved estimates at
    `55af9e2123794342bf963e2b5ee6363178821429` (`meta.restat`), whose
    `validate.py` gates the number of zero variance estimates by its exact
    binomial tail and the samples at `vectorized=False`, and centres the
    calibration of the threshold cells on the expected value of a
    thresholded estimate. The models, cells, seeds and drawing code are
    those of `c76b5ac`;
  - `timing.json` and `timing.txt` at `c76b5ac`;
  - `it_pybads.txt` and `it_pyvbmc.txt` at `55af9e2`;
  - `zero_share_replication.txt` at
    `4e4f0d9e2bd257b828d59a37b7792a72995391c6`, which adds the script that
    produced it, `zero_share_replication.py`, to the package and scripts of
    `c76b5ac`.

  The installed version string of PyIBS, `0.1.dev107+g2fbfe8d04`, is that
  of the editable install made at `2fbfe8d`, whatever the commit; the code
  of `pyibs/` outside `pyibs/testing/` is the same at the three commits.
- Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, on
  Linux-6.18.44-fc-v80-x86_64-with-glibc2.39, a cloud container with 4
  CPUs.
- PyBADS 1.5.1 from PyPI, with gpyreg 1.4.0. PyVBMC from the branch
  `feat-release-1.5-preparation` of `acerbilab/pyvbmc` at
  `89007a4eb616c2f96eee299065bf720a44969033`, whose installed version reads
  `1.0.5.dev1379+g89007a4eb` (the branch has no tag).
- PyIBS 0.1.0 from PyPI, in the venv `.venv-0.1` (Python 3.12.3, NumPy
  2.5.3, SciPy 1.18.1), for the timings.
- MATLAB IBS (`../ibs`) at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`:
  `ibs_example.m` gives the recipe of the example's data set, its bounds,
  starts and prior. No MATLAB code ran.

## Files

- `validation.json`: the validation's full run, 2,000 estimates in each of
  243 cells: the provenance, the models (trials, exact negative
  log-likelihood, exact variance of one repeat, chance-level threshold,
  range of the matching probabilities), the zero check, the exact
  references, and each cell's statistics and verdicts, with its drawing
  time (`seconds`). `validation.txt` is the table that the recomputation
  printed; its `smp_z` is given at `vectorized=False` without the
  threshold only, where the samples have no surplus and no ended repeats.
  Strings such as `'inf'` stand for non-finite numbers.
- `timing.json`: the timing's full run, with every estimate's wall time,
  simulator calls and simulated rows, from which the results compute the
  simulator's time for a cost per call or per response; `timing.txt` is
  its printed output.
  Its `median_seconds` is, for an even number of estimates, the upper of
  the two middle values; the results give the median, computed from
  `runs`.
- `it_pybads.txt` and `it_pyvbmc.txt`: the integration tests.
- `zero_share_replication.py`: the cells of `num_reps=10` without the
  threshold of the two models whose trials all match with probability
  0.999 (`bernoulli_p0.001` and `bernoulli_p0.999`), drawn again at ten
  other run seeds, 2,000 estimates each, with the number of estimates
  whose variance estimate is 0 against its exact expectation. Output:
  `zero_share_replication.txt`.

The smoke passes that projected the full runs' durations, and on which
the PI ruled, ran from the uncommitted scripts before `c76b5ac`; their
outputs are not kept, and the plan's worklog summarizes them.

## Commands

The environments: `.venv` as `AGENTS.md` ("Setup and commands") creates
it, with PyBADS and PyVBMC as `AGENTS.md` ("PyBADS and PyVBMC") installs
them, and `.venv-0.1` for the timings:

```console
uv venv --python 3.12 .venv-0.1
uv pip install --python .venv-0.1 pyibs==0.1.0
```

From the repository root, at the commit of each output:

```console
# At c76b5ac
.venv/bin/python -u dev/scripts/validate.py --estimates 2000 --jobs 3 --out dev/scripts/runs/validate_full_<t>.json > dev/scripts/runs/validate_full_<t>.log 2>&1
.venv/bin/python -u dev/scripts/timing.py --slow-jobs 12 --out dev/scripts/runs/timing_full_<t>.json > dev/scripts/runs/timing_full_<t>.log 2>&1
# At 4e4f0d9
.venv/bin/python -u dev/experiments/validation_20261009/zero_share_replication.py > dev/scripts/runs/zero_share_<t>.log 2>&1
# At 55af9e2
.venv/bin/python -u dev/scripts/validate.py --restat dev/scripts/runs/validate_full_<t>.json --out dev/scripts/runs/validate_restat_<t>.json > dev/scripts/runs/validate_restat_<t>.log 2>&1
.venv/bin/python -u -m pytest -m integration pyibs/testing/integration/test_pybads.py -s -v > dev/scripts/runs/it_pybads_<t>.log 2>&1
.venv/bin/python -u -m pytest -m integration pyibs/testing/integration/test_pyvbmc.py -s -v > dev/scripts/runs/it_pyvbmc_<t>.log 2>&1
```

`<t>` is `$(date +%s)` of the run; `--restat` reads the run's JSON and the
per-estimate arrays that `validate.py` saved beside it, in
`validate_full_<t>_raw/`, which stayed under `dev/scripts/runs/`. The
outputs were copied here from there: `validation.json` and
`validation.txt` from the recomputation's JSON and log.

The validation and the timing ran together, as the PI approved on
2026-10-09: the timing's fast cells ran alone, one version after the
other; its twelve cells named `slow`, whose simulator has a fixed cost
of 0.2 s per call, a sleep, then ran together, one process each, beside
the validation's three workers. Such a cell sleeps for nearly all of its
time, and only its computation competed for the CPUs. The integration
tests and the replication ran alone.

The seeds are constants of the scripts: in `validate.py`, `DATA_SEED`
20261009 (the data of every model) and `RUN_SEED` 20261010 (the estimates
and the exact references); in `timing.py`, `DATA_SEED` 20261009 and
`RUN_SEED` 20261011 (the simulator's draws); in the replication, the run
seeds 20261010 + 1000 r, r = 1 to 10; in the integration tests, `SEED`
20261009 (`pyibs/testing/integration/_psycho_fit.py`), from which the
data are drawn and whose children seed the starts, the IBS estimates and
PyBADS or PyVBMC.
