# Port review: PyIBS against `ibslike.m` and the IBS paper

Date: 2026-10-09. Plan: [`dev/plans/pyibs-1.5.md`](../plans/pyibs-1.5.md),
Phase 2. Evidence:
[`dev/experiments/port-review_20261009/`](../experiments/port-review_20261009/).

## Question

Does PyIBS differ from MATLAB `ibslike.m` 0.96 and `ibs_basic.m` anywhere
that the catalogue of deliberate differences (`pyibs/README.md`) does not
list, and is it correct as an implementation of inverse binomial sampling
([1], van Opheusden, Acerbi & Ma, 2020)?

## Method

PyIBS at `40e7d77` (branch `dev-next`), MATLAB IBS at `2229c00`, and the
paper's transcription in `../pubs-llms` at `a25f58b`. Two reviewers read
the code without changing it:

- Reviewer A compared `ibslike.m` and `ibs_basic.m` with `pyibs/` line by
  line: options and defaults, validation, the sampling schedule, the
  likelihood threshold, the cap, the time limit, outputs, exit flags,
  errors and the self-tests. It checked every statement about `ibslike.m`
  in the docstrings, the catalogue, the plan and the changelog against the
  source.
- Reviewer B checked the package against [1] and for internal correctness:
  the estimator and its variance estimate, the weights, the independence of
  the repeats under the schedule, the threshold of Appendix C.1, the time
  limit, the cost counts and the edge cases.

Where a conclusion about `ibslike.m` rests on MATLAB semantics, it was
checked by running `ibslike.m` under GNU Octave 8.4.0 (`octave_checks.m`
and its output in the evidence directory); each such finding names the
semantic. Every finding below was checked again apart from the reviewer
who reported it: F-1 by the PyIBS run of the evidence directory, F-2 by the
test that its fix adds, F-3, F-4, F-6 and F-7 by its Octave runs, and the
others against the source or the argument.

## Main results

- Apart from the findings below, every behavioural difference between
  PyIBS and `ibslike.m` in the options, defaults, schedule, threshold, cap,
  time limit, outputs, exit flags and errors is catalogued, with its
  reason.
- One small statistical defect (F-1): an `IBS` object's first call with
  `vectorized=None` is biased when the simulator's running time depends on
  what it simulates.
- One defect of `ibs_basic` (F-2): it loops forever on simulated responses
  of a kind that NumPy never finds equal to the observed ones.
- Three differences from `ibslike.m` that the catalogue does not list, or
  lists incompletely (F-3, F-4, F-11).
- Three statements about `ibslike.m` that its source contradicts or leaves
  incomplete (F-6 to F-8), two statements about PyIBS's own estimates and
  settings that mislead (F-5, F-9), and two gaps in the guidance to users
  (F-10, F-12).
- `ibslike.m` runs unmodified under Octave and passes its self-tests there.

## Interpretation

PyIBS departs from `ibslike.m` only on purpose, and the catalogue gives
each departure with its reason, after the few entries that the review
added or corrected. The two defects lay outside a comparison of the two
codes: F-1, a bias that `ibslike.m` and PyIBS 0.1.0 share, appears only
when one asks whether the use of a sample can depend on its value, and
F-2 is a trap of comparisons between values of different kinds. Most other
findings were statements about `ibslike.m` that needed MATLAB's semantics
to settle, which running it under Octave did.

## Ledger

The verdicts and outcomes proposed, and the PI's rulings of 2026-10-09.
Every outcome is carried out, each fix with a test that fails before it:
F-1 in `27e321c`, with its changelog entry in `09bda5a`; F-2 in `cf0ed60`;
F-4 and F-10 in `27e321c` and `28a0891`; F-12 in `16790c6` and, at the
PI's request after the phase's review, by a check of object responses in
`2b2e299`; the others in `16790c6`. The phase's review, which the plan's
Worklog records, corrected the wording of several.

| ID | Finding | Reported by | Verdict | Outcome | Ruling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F-1 | The first call with `vectorized=None` uses the timing call's samples only when its first round requests one sample per trial | B | Defect | Fix: the timing call is always the first round; KD-8, KD-11, D19; changelog | Accepted, outcome 1 |
| F-2 | `ibs_basic` loops forever on responses of a kind that never matches | A, B | Defect | Fix: the kind check of `IBS`; KD-16, KD-18; changelog | Accepted, the fix |
| F-3 | With one trial, `ibslike.m`'s per-trial arrays have one entry per repeat | A | Deliberate difference, catalogue entry missing | KD-3; a test of the shapes | Accepted |
| F-4 | `MaxIter = Inf` disables `ibslike.m`'s cap; PyIBS refuses `max_iter=np.inf` | A | Deliberate difference, catalogue entry missing | KD-6, KD-11 | Accepted, the refusal kept |
| F-5 | A finite `max_time` biases the calls that complete in time too | B | Documentation error | Docstrings, KD-14; KD-9 and D4 on the cap | Accepted |
| F-6 | `ibslike.m`'s vectorized path returns NaN when the time limit passed before its first round | A | Documentation error | KD-14, parity table, D3 | Accepted |
| F-7 | KD-6: a per-trial `Nreps` vector runs on the vectorized path in some cases | A | Documentation error | KD-6 | Accepted |
| F-8 | D23: `ibslike.m`'s examples call their simulator, not `ibslike`, with a NaN design | A | Documentation error | D23 | Accepted |
| F-9 | Zero-weight trials are sampled; the chance-level threshold is written unweighted | B | Documentation error | Docstrings of `IBS` | Accepted |
| F-10 | A simulator that is slow only at its first call fixes `vectorized=None` at False | B | No issue in the code (D19) | A sentence in the `vectorized` docstring | Accepted |
| F-11 | KD-19 does not state the settings of the ported self-tests | A | Deliberate difference, catalogue entry incomplete | KD-19 | Accepted |
| F-12 | Responses mixing numbers and text become text in a NumPy array | B | Documentation gap | `response_matrix` docstring; then a check of object responses | Accepted; the check at the PI's request |
| N-1 | `IBS` checks settings before `_Settings` checks them again | A | No issue | None | Accepted |
| N-2 | `max_samples` (per trial and call) beside `max_samples_per_trial` | A | No issue | None | Accepted |
| N-3 | `ibs_basic` compares a scalar output with every column of a response | A, B | No issue | None | Accepted |
| N-4 | Smaller points: the plan's "for a positive level"; a NaN stimulus in `psycho_model` | A | No issue | None | Accepted |

## Findings

### F-1. The first call with `vectorized=None` uses the timing call's samples only when its first round requests one sample per trial

At an `IBS` object's first call with `num_reps > 1`, `vectorized=None`
times one simulation of every trial (`IBS.__call__`, `pyibs/ibs.py`; D19,
KD-8). At `40e7d77`, `sample` (`pyibs/_sampler.py`) took that call as its
first round when the round requested one sample per trial, and discarded
its samples otherwise. With the default `num_samples_per_call=0`, the
first round requests one sample per trial when the simulation lasts
`vectorized_threshold` or more and the decision is False, and `num_reps`
samples per trial (`max_mem` permitting) when it is True. `ibslike.m` does
the same at every call (lines 176-190, 287-289, 433-434).

Every other choice of the sampler is made before the samples it governs
are drawn, which keeps the repeats independent ([1], Eqs 14 and 17 assume
it). This one decides whether samples are used after they are drawn. When
the simulator's running time depends on its outcomes, the decision and the
outcomes are dependent, and the call is biased.

- Size: with h_i the timing call's outcome of trial i, the decision D
  (False) and X = Σ w_i g_i(h_i), where g_i(h) = (1 - h) log p_i / (1 -
  p_i) is the expected first count's term given h, the bias of the call's
  log-likelihood estimate is Cov(X, 1_D) / n. It is at most SD_1 / (2n),
  where SD_1² = Σ w_i² Li₂(1 - p_i) is the variance of one repeat: at most
  1/(2√n) of the returned SD, 0.16 SD at n = 10. It needs a simulation of
  all trials that lasts about `vectorized_threshold` and whose duration
  tracks the simulated responses, such as a model whose simulated response
  times or search depths set its running time.
- Reach: the object's first call with `num_reps > 1`. Code that creates an
  `IBS` object per evaluation meets it at every evaluation, as `ibslike.m`
  does.
- Evidence: `first_call_bias.txt`. One trial matched with probability 0.5,
  a simulator whose timing call lasts 0.2 s on a fake clock when it matches
  and 0 s otherwise, `num_reps=2`, 3,000 new objects: the mean estimate is
  0.5076 ± 0.0090, against the exact log 2 = 0.6931 (z = -20.5) and the
  0.75 log 2 = 0.5199 that keeping the timing call exactly when it matched
  predicts. No test at `40e7d77` could see it, since the fake clocks of
  the tests advance independently of the outcomes;
  `test_first_call_is_unbiased_when_the_timing_tracks_the_outcomes`
  repeats the case. After the fix,
  `first_call_bias_after_fix.txt` gives 0.6794 ± 0.0098 (z = -1.4).

Verdict: defect. The outcomes considered, of which the PI chose the first:

1. Recommended: the timing call is always the draw's first round. When the
   schedule's first round requests one sample per trial, it is that round,
   as at `40e7d77` and as in `ibslike.m`; otherwise it is an extra round of one
   sample per trial before the schedule's first, and the level does not
   grow after it, so that the rounds after it are those of `ibslike.m`'s
   schedule. The use of the samples no longer depends on their outcomes,
   the N samples of a discarded call are no longer wasted, and the code
   that counts a discarded call toward the cost and the cap goes. KD-8
   says that `ibslike.m` discards the timing call when its first round
   requests more than one sample per trial (lines 287-289), KD-11 loses
   its sentence on a discarded call, and D19 states the rule. A regression
   test repeats the evidence's case with 2,000 objects, against 4.5
   standard errors.
2. The timing call is always discarded: the same effect, at the cost of an
   extra simulation of every trial, which lasts `vectorized_threshold` or
   more when the decision is False, and a departure from `ibslike.m`'s loop
   path (lines 433-434).
3. No change, with the bias stated in KD-8, as `ibslike.m`'s.

### F-2. `ibs_basic` loops forever on responses of a kind that never matches

`ibs_basic` (`pyibs/ibs_basic.py`) samples while
`not np.all(np.asarray(simulate(s)) == R[i])`. NumPy finds text or bytes
unequal to numbers or booleans whatever their values, so a simulator that
returns `"1"` for numeric responses never matches, and
`ibs_basic(lambda theta, s: "1", None, np.ones(3))` never returns. `IBS`
raises `TypeError` after the first such simulator call (KD-16), and
`ibs_basic` raises on a NaN response (D23, KD-18) for the same reason, a
response that cannot match. `ibs_basic.m` compares characters with numbers
by their codes (line 33): it loops forever on this example too, since
`'1'` has the code 49, but it matches a character whose code equals the
response, where NumPy never finds text equal to a number.

Verdict: defect. Outcome: `ibs_basic` checks every simulated
response with the check of `IBS` (`_sampler._check_kinds`), at the cost of
one comparison of dtypes per sample; a test whose simulator returns `"1"`
for numeric responses and raises `RuntimeError` after 50 calls, which
fails with that error before the fix and gets `TypeError` after it. KD-16
names `ibs_basic` too; KD-18 adds the check and `ibs_basic`'s checks of `R`
and `S`, which `ibs_basic.m` does not make. The changelog's entry on
`ibs_basic` under Fixed adds the `TypeError`, since 0.1.0's `ibs_basic`
looped forever too. The alternative, keeping `ibs_basic` bare and
recording the endless loop in KD-18, was not chosen.

### F-3. With one trial, `ibslike.m`'s per-trial arrays have one entry per repeat

`ibslike.m` computes `nlogLvar_trials` as `sum(Ktab(max(K,1)), 2) ./
Nreps.^2` (lines 210-213), with `K` of shape `Ntrials`-by-`Nreps` (lines
399, 413). With one trial, `K` is a row, and MATLAB indexes the column
table `Ktab` with a row vector into a column, which the sum over the
second dimension leaves as it is. So `nlogLvar_trials` holds one entry per
repeat, each repeat's term over `Nreps²`, on both paths, and so does the
loop path's `nlogL_trials` (lines 484-485, 498), whenever a count exceeds 1;
when every count is 1, the tables are scalars and so are the arrays. The
totals are right (lines 225, 228). Semantics: indexing a vector with a
vector gives the orientation of the indexed vector, and `sum(A, dim)` over
a singleton dimension returns `A`. Evidence: `octave_checks.txt`, the lines
"N = 1".

PyIBS's arrays have shape (N,) at every N. Verdict: deliberate
difference, catalogue entry missing. Outcome: KD-3 states it, with the
lines; `test_scalar_responses_are_one_trial` (`test_ibs.py`) also asserts
the shape of `neg_logl_var_trials`.

### F-4. `MaxIter = Inf` disables `ibslike.m`'s cap; PyIBS refuses `max_iter=np.inf`

`ibslike.m` does not check `MaxIter`. With `Inf`, its loops `for iter =
1:MaxIter` (lines 260, 269, 427) run until the sampling ends, and
`ibslike:ConvergenceFail` does not occur. `NsamplesPerCall = Inf` likewise
starts the samples per call at their bounds (lines 264-266, 281-283).
Semantics: a `for` over a colon range iterates without building it; Octave
bounds an infinite one at 2^63 - 1 iterations, with a warning, which no run
reaches, and MATLAB, not run here, is taken to do likewise. Evidence:
`octave_checks.txt`, the lines "MaxIter", where `MaxIter = 2` is the
control. `IBS` raises `ValueError` for `max_iter=np.inf` and
`num_samples_per_call=np.inf` (`_check_count`), deliberately
(`test_settings_out_of_range_raise`, `test_ibs.py`), although the sampler
can run without a cap (`max_samples_per_trial=None`).

Verdict: deliberate difference, catalogue entry missing; D3 keeps
the cap as the guard against an observed response that the simulator
cannot produce, which would otherwise sample forever with nothing to name
the cause. Outcome: KD-11 states that `MaxIter = Inf` disables
`ibslike.m`'s cap and that `max_iter` is finite; KD-6 states that the
counts are finite, where `ibslike.m` takes `MaxIter` and `NsamplesPerCall`
of `Inf`. The alternative, mapping `max_iter=np.inf` to no cap, was not
chosen.

### F-5. A finite `max_time` biases the calls that complete in time too

A call ends with exit flag 0 when its sampling completes before
`max_time`. For a simulator whose running time grows with the samples it
draws, completing in time is more likely with small counts, which give
high log-likelihoods. Both the repeat values and the event of completing
in time decrease as the counts grow, and the counts are independent, so
the two are positively correlated (Harris's inequality): the estimates of
the calls with exit flag 0 are biased upward when the limit can bind. The
documents call a flag-0 estimate unbiased: `EstimateResult.exit_flag` and
the message of exit flag 0 (`pyibs/ibs.py`), 0.1.0's, which restates the
header of `ibslike.m` (lines 40-44), and the `max_time` docstring by
omission. With
`max_time=np.inf`, the default, nothing changes.

Wall-clock time reaches an estimate's value only here and in F-1. The
other uses of time, the rule of `acceleration_threshold` and the timing
call's duration, are decided before the samples they govern and change the
schedule only: the cost, the variance estimate of a repeat that the
threshold ends, and, since the cap counts the samples drawn after a
trial's last match, whether a call reaches the cap. KD-9 and D4 list the
first two effects of the schedule and not the third.

Verdict: documentation error. Outcome: the `max_time` docstring
of `IBS`, `EstimateResult.exit_flag` and KD-14 say that only an infinite
`max_time` gives unbiased estimates, since completing in time favours small
counts; the message of exit flag 0 stays as it is. KD-9 and D4 add the cap
to the effects of the schedule.

### F-6. `ibslike.m`'s vectorized path returns NaN when the time limit passed before its first round

`ibslike.m`'s vectorized path checks `MaxTime` before every round, the
first included (lines 272-275), counted from the start of the call (line
86). When the limit has passed by then, as when the timing call of
`'auto'` outlasts `MaxTime` but not `VectorizedThreshold`, no round is
sampled, every count is 0, and every value and variance is 0/0, NaN, with
exit flag 2 (lines 210-213, 396-398). Evidence: `octave_checks.txt`, the
lines "MaxTime = 1e-9", which show it on both paths. KD-14, the parity
table's row "Time limit" and D3 give NaN for the loop path only.

PyIBS checks the time only after a simulator call (`sample`), and raises
for a trial without a completed count, as KD-14 says. Verdict:
documentation error. Outcome: KD-14, the parity table and D3 state the
vectorized path's NaN, with lines 210-213 and 396-398.

### F-7. A per-trial `Nreps` vector runs on `ibslike.m`'s vectorized path in some cases

KD-6 said that `ibslike.m` reads `Nreps` as a vector of per-trial repeats
in places, "which no documentation offers and neither of its paths can
run". With `Vectorized` given, MATLAB stops a vector at line 193, whose
`&&` takes scalars; Octave reduces a vector there with `all` and goes on.
With `'auto'`, the `if` of line 177 takes the vector in both. The loop path
then fails at line 413, and the vectorized path at line 284 when
`NsamplesPerCall` is 0, since its level is then `Nreps` (line 264). With
`NsamplesPerCall` set, it reaches line 361, where implicit expansion
accepts the vector while every trial is open: with `Nreps = [2; 3]` and
two trials that complete together, it returns per-trial results for 2 and
3 repeats. Once some trials are done, line 361 fails, or, with one trial
open, applies another trial's limits and fails at line 372 or misplaces
the counts. Evidence: `octave_checks.txt`, the lines "Nreps = [2; 3]", run
with `Vectorized` given, which Octave lets through line 193; the last of
them fails at line 372, for a stream in which trial 1 completes first.
Semantics: `&&` on scalars only in
MATLAB, and implicit expansion, shared by MATLAB since R2016b and Octave.

Verdict: documentation error. Outcome: KD-6 says that its code does not
support a vector: line 193 stops it in MATLAB when `Vectorized` is given;
the loop path fails at line 413; and the vectorized path fails at line 284
unless `NsamplesPerCall` is set, and otherwise, once fewer than all trials
are open, at line 361 or 372, or misplaces the counts.

### F-8. `ibslike.m`'s examples call their simulator, not `ibslike`, with a NaN design

D23 says that `ibslike.m`'s own examples pass a NaN design (lines 57,
511). They call their simulator with a NaN array to generate the responses
(lines 57, 511, 587, 630), and call `ibslike` without a design or with
`[]`
(lines 58, 523, 588, 646). KD-17 states it correctly. Verdict:
documentation error in the plan. Outcome: D23 says that the examples call
their simulator with a design of NaN, so that a port passing such a design
to `IBS` keeps working.

### F-9. Zero-weight trials are sampled; the chance-level threshold is written unweighted

A trial of weight 0 adds nothing to the value, the variance or the
threshold's bound, but it is sampled to `num_reps` matches, as in
`ibslike.m` (line 271), at about `num_reps / p_i` samples. It can reach the
cap, and under the time limit it raises when it has no completed count.
The `trial_weights` docstring of `IBS.__call__` does not say so, and a
weight of 0 is a natural way to hold a trial out. Separately, the
`neg_logl_threshold` docstring gives the usual threshold as `N log 2` for
N binary choices, while the bound is weighted: the chance level is
`sum_i w_i log 2`, as the Notes of `sample` say.

Verdict: documentation error. Outcome: the `trial_weights`
docstring says that a trial of weight 0 is still sampled, and can reach
the cap or the time limit, so that a trial to leave out is better removed
from the data; the `neg_logl_threshold` docstring gives the weighted chance
level.

### F-10. A simulator that is slow only at its first call fixes `vectorized=None` at False

A JIT-compiled simulator, or one that fills caches at its first call, can
exceed `vectorized_threshold` at the timing call and never again, and D19
then keeps the one-sample schedule for every later call of the object.
This costs time, not accuracy, and follows from D19, where `ibslike.m`
decides at every call. Verdict: no issue in the code. Outcome: the
`vectorized` docstring says that the timing includes any warm-up, so that
such a simulator is given `vectorized=True` or called once before.

### F-11. KD-19 does not state the settings of the ported self-tests

`runtest1` to `runtest3` leave `Vectorized` at `'auto'` (lines 517-518,
578, 639-641), and `runtest1` and `runtest2` the acceleration under its
time rule; `runtest3` sets it to 1 (line 641), as its port does. The ports
(`test_ibslike_ports.py`) run with `vectorized=True`, and those of
`runtest1` and `runtest2` with the samples per call growing after every
call, so that a seed alone decides their draws. Their criteria are
`ibslike.m`'s, except that the ports pass on a strict `<` at the
tolerances of the RMSE and of the z-scores' mean and SD, where `ibslike.m`
passes on `<=` (lines 535, 610, 666). Verdict: deliberate difference,
catalogue entry incomplete. Outcome: KD-19 states the settings and their
reason.

### F-12. Responses mixing numbers and text become text in a NumPy array

Responses that mix numbers and text in a row keep their numbers in an
object array, but a simulator that builds its output with `np.array` turns
them into text: `np.array([[1, "a"], [2, "b"]])` is an array of text,
whose `"1"` never equals the response 1. At `40e7d77`, `_check_kinds` left
object responses unchecked, so such a call sampled until the cap, and
`ibs_basic` forever. Verdict: documentation gap. Outcome: the
`response_matrix` docstring says that responses mixing numbers and text
are given as an object array, and returned by the simulator as one. After
the phase's review, at the PI's request, `2b2e299` also checks the kinds of
the elements of object responses, so that such an output raises
`TypeError` at the first simulator call (KD-16).

### N-1 to N-4. No issue

- N-1. `IBS.__init__` checks the responses, the design, `max_iter` and
  `num_samples_per_call` before `_Settings` checks them again. Both use the
  same check functions, every value that `IBS` accepts passes `_Settings`,
  and every error names the argument that the user gave. `IBS` refuses
  `None` for `max_iter` and `num_samples_per_call`, which `_Settings` takes
  (F-4).
- N-2. `max_samples`, per trial and simulator call, appears only in the
  samples per call and the bound of the level; `max_samples_per_trial`,
  per trial and repeat, only in the cap and its message; `IBS` maps
  `max_samples` and `max_iter` onto them.
- N-3. `ibs_basic` compares a scalar output with every column of a
  response, which `IBS` refuses (D22); `ibs_basic.m` does the same (line
  33), so `ibs_basic` follows it.
- N-4. The plan's Phase 1, step 2, calls `math.floor(level + 0.5)`
  MATLAB's `round` "for a positive level", which fails just below 0.5; the
  level is at least 1, as the code's docstring says. A stimulus of NaN
  gives the response 0 in `psycho_gen.m` and -1 in `psycho_model`, an
  input with no meaning.

## Checked and found correct

- The count formulas: `ibs_loglik(K) = psi(1) - psi(K)` is [1], Eq 14, 0
  at K = 1, with mean log p; `ibs_var(K) = psi_1(1) - psi_1(K)` is Eq 16,
  whose mean is Li₂(1 - p), the variance of Eq 15, so the variance
  estimate is unbiased. The tables equal the formulas bitwise, and the
  bitwise properties of `repeat_estimates` that `AGENTS.md` states hold.
- The estimate, the mean of the repeat values, and its variance estimate,
  the sum of theirs over n²; the weights, w in the value and w² in the
  variance; the per-trial arrays of `"full"`, which add up, weighted, to
  the totals, and are NaN once the threshold ended a repeat (D21).
- Independence: every sample is a fresh draw, the samples per round are
  fixed before the round, and each sample goes to a repeat by its order
  and the repeats active at the round's start, so complete repeats are
  i.i.d. IBS repeats on either schedule and under `acceleration_threshold`;
  the n-th match is a stopping time, so the samples after it can be
  discarded. F-1 is the one exception.
- The threshold: the bound counts c + 1 for a trial sampling a repeat (the
  paper's counts c), is a lower bound on -Y_r, never decreases, and equals
  -Y_r at completion; every incomplete repeat is checked after every round,
  complete repeats on their exact values, so a repeat ends exactly when
  -Y_r > T and is worth exactly -T. `ibslike.m`'s bounds are c on its
  vectorized path and c + 1 on its loop path, with the partial counts kept
  (lines 347-387, 456-466), as KD-13 says.
- The time limit: D24's estimate and variance, D3's error, no check before
  the first call, and no flag for a sampling that completes.
- The cap, the cost counts, the samples-per-call formula with MATLAB's
  `round`, the output types (Python floats, a `tuple` for `"var"` and
  `"std"`), the warning on a zero variance, and the edge cases: one trial,
  `num_reps=1`, trials that always match, text, bytes and object
  responses, NaN responses, a design of None, and responses of shape (N,)
  and (N, 1).
- Every other statement about `ibslike.m` in the catalogue (each cited
  line), the docstrings, the tests, the plan's Context, parity table and
  decisions, and the changelog, and the example model against
  `psycho_gen.m` and `psycho_nll.m`.
- `ibslike('test')` passes its three self-tests under Octave
  (`octave_checks.txt`).

## References

[1] B. van Opheusden, L. Acerbi and W. J. Ma (2020), "Unbiased and
efficient log-likelihood estimation with inverse binomial sampling", PLOS
Computational Biology 16(12): e1008483,
https://doi.org/10.1371/journal.pcbi.1008483.
