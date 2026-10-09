# Checks of the port review

The evidence that the port review,
[`dev/results/2026-10-09-port-review.md`](../../results/2026-10-09-port-review.md),
cites: runs of MATLAB `ibslike.m` under GNU Octave, which settle what it
does where its source alone leaves doubt, and a run of PyIBS that measures
the bias of finding F-1.

## Provenance

- Each output was produced at a commit of PyIBS with a clean tree (`git
  status --porcelain` empty):
  - `first_call_bias.txt` at `b6fe6fb361614995345580bc13acbdcf3ac08550`,
    whose package code is that of `40e7d77`, the code that the review read;
  - `first_call_bias_after_fix.txt` at
    `27e321c144a88a44902241313ce2e4f0a776c78d`, which fixes F-1;
  - `octave_checks.txt` at `14b0958e27a4886b392f2f66d2b074532851da6b`,
    which runs no code of PyIBS.

  The installed version string, `0.1.dev54+g40e7d7748`, is that of the
  editable install made at `40e7d77`, whatever the commit.
- Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, on
  Linux-6.18.44-fc-v80-x86_64-with-glibc2.39.
- MATLAB IBS (`../ibs`) at `2229c00c4a19eb9f236f9f257100dab9e87b6f92`,
  `ibslike.m` 0.96, run under GNU Octave 8.4.0 with
  `dev/scripts/octave/compat` and `dev/scripts/octave/headless` on its path
  (`AGENTS.md`, "Sibling repositories"). Octave is not MATLAB: the review
  names the semantic that each of its checks takes the two to share.

## Files

- `octave_checks.m`, with the simulator `scripted_fun.m`: `ibslike.m` with
  one trial on both paths, with `MaxIter = Inf`, with a per-trial `Nreps`
  vector, with a `MaxTime` passed before the first round, and
  `ibslike('test')`. The groups of checks that draw random numbers seed
  them with `rand('state', k)`, k = 1 to 4; the checks of the `Nreps`
  vector draw from fixed streams instead. Output: `octave_checks.txt`,
  from which Octave's warnings ("FOR loop limit is infinite", for
  `MaxIter = Inf`, and the lines that locate them) and blank lines are
  removed.
- `first_call_bias.py`: 3,000 first calls of new `IBS` objects with
  `vectorized=None` on a fake clock, generator seed 12345. Output:
  `first_call_bias.txt` before the fix of F-1, and
  `first_call_bias_after_fix.txt` after it.

## Commands

From the repository root, with the commit of each output checked out
(`git checkout <commit>`, then back to the branch):

```console
# At 14b0958
octave-cli --norc -q dev/experiments/port-review_20261009/octave_checks.m 2>&1 | grep -v "^warning\|^    \|^$" > dev/experiments/port-review_20261009/octave_checks.txt
# At b6fe6fb
.venv/bin/python -u dev/experiments/port-review_20261009/first_call_bias.py 3000 > dev/experiments/port-review_20261009/first_call_bias.txt
# At 27e321c
.venv/bin/python -u dev/experiments/port-review_20261009/first_call_bias.py 3000 > dev/experiments/port-review_20261009/first_call_bias_after_fix.txt
```
