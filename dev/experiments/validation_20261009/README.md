# Statistical validation, PyBADS and PyVBMC, and timing of PyIBS 1.5

The evidence that
[`dev/results/2026-10-09-validation.md`](../../results/2026-10-09-validation.md)
cites: the statistical validation of PyIBS's estimates against exact
log-likelihoods, the integration tests with PyBADS and PyVBMC, and the
timings of PyIBS 1.5 against PyIBS 0.1.0 (Phase 3 of
[`dev/plans/pyibs-1.5.md`](../../plans/pyibs-1.5.md)).

## Provenance

- Each output was produced at a commit of PyIBS with a clean tree (`git
  status --porcelain` empty):
  - `validation.json`, `validation.txt`, `timing.json` and `timing.txt`
    at `c76b5ac9239b2f507f631291619c031fbad820b9`, which holds the
    scripts that produced them, `dev/scripts/validate.py` and
    `dev/scripts/timing.py`;
  - `it_pybads.txt` and `it_pyvbmc.txt` at
    `5e1876b01d33cc1d4390e39108366ef5964e0e7c`, whose package and tests
    are those of `c76b5ac` (it changes only the plan);
  - `zero_share_replication.txt` at
    `4e4f0d9e2bd257b828d59a37b7792a72995391c6`, which adds the script
    that produced it, `zero_share_replication.py`, to the package and
    scripts of `c76b5ac`.

  The installed version string of PyIBS, `0.1.dev107+g2fbfe8d04`, is that
  of the editable install made at `2fbfe8d`, whatever the commit.
- Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, on
  Linux-6.18.44-fc-v80-x86_64-with-glibc2.39, a cloud container with 4
  CPUs.
- PyBADS 1.5.1 from PyPI, with gpyreg 1.4.0. PyVBMC from the branch
  `feat-release-1.5-preparation` of `acerbilab/pyvbmc` at
  `89007a4eb616c2f96eee299065bf720a44969033`, whose installed version reads
  `1.0.5.dev1379+g89007a4eb` (the branch has no tag), as `AGENTS.md`
  ("PyBADS and PyVBMC") installs it.
- PyIBS 0.1.0 from PyPI, in the venv `.venv-0.1` (Python 3.12.3, NumPy
  2.5.3, SciPy 1.18.1), for the timings.
- MATLAB IBS (`../ibs`) at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`:
  `ibs_example.m` gives the example's data set, bounds, starts and prior.
  No MATLAB code ran.

## Files

- `validation.json`: the validation's full run, 2,000 estimates in each of
  243 cells: the provenance, the models (trials, exact negative
  log-likelihood, exact variance of one repeat, chance-level threshold,
  range of the matching probabilities), the zero check, the exact
  references, and each cell's statistics and verdicts. `validation.txt`
  is its printed table. Strings such as `'inf'` stand for non-finite
  numbers.
- `timing.json`: the timing's full run, with every estimate's wall time,
  simulator calls and simulated rows; `timing.txt` is its printed output.
- `it_pybads.txt` and `it_pyvbmc.txt`: the integration tests.
- `zero_share_replication.py`: the cells of `num_reps=10` without the
  threshold of the two models whose trials all match with probability
  0.999 (`bernoulli_p0.001` and `bernoulli_p0.999`), drawn again at ten
  other run seeds, 2,000 estimates each, with the number of estimates
  whose variance estimate is 0 against its exact expectation. Output:
  `zero_share_replication.txt`.

## Commands

From the repository root, at the commit of each output:

```console
.venv/bin/python -u dev/scripts/validate.py --estimates 2000 --jobs 3 --out dev/scripts/runs/validate_full_$(date +%s).json > dev/scripts/runs/validate_full_$(date +%s).log 2>&1
.venv/bin/python -u dev/scripts/timing.py --slow-jobs 12 --out dev/scripts/runs/timing_full_$(date +%s).json > dev/scripts/runs/timing_full_$(date +%s).log 2>&1
.venv/bin/python -u -m pytest -m integration pyibs/testing/integration/test_pybads.py -s -v > dev/scripts/runs/it_pybads_$(date +%s).log 2>&1
.venv/bin/python -u -m pytest -m integration pyibs/testing/integration/test_pyvbmc.py -s -v > dev/scripts/runs/it_pyvbmc_$(date +%s).log 2>&1
.venv/bin/python -u dev/experiments/validation_20261009/zero_share_replication.py > dev/scripts/runs/zero_share_$(date +%s).log 2>&1
```

The runs overlapped, as the PI approved on 2026-10-09: the timing's fast
cells ran alone, one version after the other; its twelve slow cells then
ran together, one process each, beside the validation's three workers and
the integration tests. A slow cell sleeps 0.2 s per simulator call for
nearly all of its time, and only its computation competed for the CPUs.
No clock decides the integration tests' results: each printed the same
point, values and number of evaluations as a run of the same tests made
alone, before the commit.

The outputs were copied here from `dev/scripts/runs/` under these names;
the validation's per-estimate arrays, which `validate.py` writes next to
its JSON, stayed there. The seeds are constants of the scripts:
`DATA_SEED` (the data of every model) and `RUN_SEED` (the estimates, the
exact references and, in the timing, the simulator's draws), and the
integration tests' `SEED` (`pyibs/testing/integration/_psycho_fit.py`).
