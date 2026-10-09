# Checks of the port review

The evidence that the port review,
[`dev/results/2026-10-09-port-review.md`](../../results/2026-10-09-port-review.md),
cites: runs of MATLAB `ibslike.m` under GNU Octave, which settle what it
does where its source alone leaves doubt, and a run of PyIBS that measures
the bias of finding F-1.

## Provenance

- PyIBS at `b6fe6fb361614995345580bc13acbdcf3ac08550`, a clean tree (`git
  status --porcelain` empty). The package code is that of `40e7d77`; the
  installed version string, `0.1.dev54+g40e7d7748`, is that of the editable
  install made there.
- Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, on
  Linux-6.18.44-fc-v80-x86_64-with-glibc2.39.
- MATLAB IBS (`../ibs`) at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`,
  `ibslike.m` 0.96, run under GNU Octave 8.4.0 with
  `dev/scripts/octave/compat` and `dev/scripts/octave/headless` on its path
  (`AGENTS.md`, "Sibling repositories"). Octave is not MATLAB: each check
  rests on a semantic that the two share, which the review names.

## Files

- `octave_checks.m`, with the simulator `scripted_fun.m`: `ibslike.m` with
  one trial on both paths, with `MaxIter = Inf`, with a per-trial `Nreps`
  vector, with a `MaxTime` passed before the first round, and
  `ibslike('test')`. Seeds: `rand('state', k)`, k = 1 to 4, one per group of
  checks. Output: `octave_checks.txt`, from which Octave's warnings ("FOR
  loop limit is infinite", for `MaxIter = Inf`) and blank lines are removed.
- `first_call_bias.py`: 3,000 first calls of new `IBS` objects with
  `vectorized=None` on a fake clock, generator seed 12345. Output:
  `first_call_bias.txt`.

## Commands

From the repository root:

```console
octave-cli --no-gui --norc -q dev/experiments/port-review_20261009/octave_checks.m 2>&1 | grep -v "^warning\|^    \|^$" > dev/experiments/port-review_20261009/octave_checks.txt
.venv/bin/python -u dev/experiments/port-review_20261009/first_call_bias.py 3000 > dev/experiments/port-review_20261009/first_call_bias.txt
```
