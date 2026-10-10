# PyIBS and MATLAB ibslike.m: deliberate differences

PyIBS implements inverse binomial sampling as MATLAB `ibslike.m` does, the
reference implementation (`acerbilab/ibs`, Version 0.96 of Jan 21, 2021, at
commit `2229c00`). This file catalogues where PyIBS differs from
`ibslike.m`, and from the didactic `ibs_basic.m`, on purpose, each with its
reason. A difference that is not listed here is not known to be
deliberate: it is a defect until shown otherwise.

PyIBS is cited by module and function, `ibslike.m` and `ibs_basic.m` by
their lines at `2229c00`. Each entry names the decision of the plan
[`dev/plans/pyibs-1.5.md`](https://github.com/acerbilab/pyibs/blob/main/dev/plans/pyibs-1.5.md)
that settled it (D<n>), where there is one. A change that adds, removes or
alters a deliberate difference updates its entry here.

N is the number of trials, and n the number of repeats of a call
(`num_reps`, `ibslike.m`'s `Nreps`). T is the likelihood threshold
(`neg_logl_threshold`, `ibslike.m`'s `NegLogLikeThreshold`). A trial's
open count c is the number of samples it has drawn since its last match,
in the repeat it is sampling. Y_r is the log-likelihood estimate that the
complete sampling of repeat r gives, the weighted sum of its trials'
terms.

Kinds: *deliberate change* (PyIBS does it otherwise), *Python-only
feature* (`ibslike.m` has no counterpart), *removed feature* (`ibslike.m`
has it, PyIBS does not).

## The interface

**KD-1. An object holds the simulator, the data and the settings; a call
takes the parameter vector and the options of one estimate.**
`ibslike.m` is a function of the simulator, the parameters, the responses,
the design and an options structure (line 1). PyIBS keeps the calling
convention of PyIBS 0.1.0: `IBS(sample_from_model, response_matrix,
design_matrix, ...)` holds the simulator, the data, the settings and the
generator (KD-4) across the many calls of an optimization or of inference,
and `ibs(params, num_reps, trial_weights, additional_output,
return_positive)` estimates. `Nreps`, `TrialWeights` and `ReturnPositive`
are arguments of the call; `ReturnStd` is `additional_output="std"`;
`Vectorized` `'auto'` is `vectorized=None`; `NegLogLikeThreshold` `Inf`
is `neg_logl_threshold=np.inf`; the other options are settings of the
object with snake_case names. `ibslike.m`'s hard-coded `MaxSamples`,
`AccelerationThreshold`, `VectorizedThreshold` and `MaxMem` (lines
133-137) are settings with its values as defaults, but for
`acceleration_threshold` (KD-9). The simulator takes no extra arguments
(`varargin`, lines 31, 183, 293): a closure or `functools.partial` passes
them. There is no `ibslike('defaults')` (lines 101-108): the defaults are
those of the signature, and the settings are read-only attributes of the
object.
- PyIBS: `IBS.__init__`, `IBS.__call__` (`pyibs/ibs.py`).
- MATLAB: `ibslike.m:1`, `31`, `89-99`, `101-108`, `133-137`, `183`,
  `293`.
- Settled by: D2. Kind: deliberate change (interface); removed features
  (`varargin` and `ibslike('defaults')`).

**KD-2. The outputs are a float, a tuple of two floats, or an
`EstimateResult`.**
`ibslike.m` returns the negative log-likelihood, its variance (its SD with
`ReturnStd`), an exit flag and a structure with `funcCount`,
`NsamplesPerTrial`, `nlogL_trials` and `nlogLvar_trials` (lines 1,
209-232). PyIBS returns, by `additional_output`, the estimate alone, a
`tuple` of the estimate and its variance (`"var"`) or its SD (`"std"`),
the latter the form that PyBADS and PyVBMC take from a noisy target, or an
`EstimateResult` (`"full"`) with the estimate, its variance and SD, the exit
flag and its message, the elapsed time, the samples per trial, the
simulator calls and the per-trial arrays (KD-3). The numbers are Python
floats. A call that returns a variance estimate of exactly 0, as when
every trial matches at its first sample, issues a `UserWarning` that links
the FAQ, since PyBADS and PyVBMC refuse an SD of 0; the value is returned
as computed. `return_positive` changes the sign of the estimate only, as
`ReturnPositive` does (line 226).
- PyIBS: `IBS.__call__`, `EstimateResult` (`pyibs/ibs.py`).
- MATLAB: `ibslike.m:1`, `209-232`.
- Settled by: D2, D10. Kind: deliberate change (interface), with the
  Python-only fields `message` and `elapsed_time` and the warning.

**KD-3. The per-trial arrays are NaN when the likelihood threshold ended a
repeat.**
`ibslike.m`'s `nlogL_trials` and `nlogLvar_trials` (lines 220-221) average,
for each trial, its positive counts (lines 396-398, 483-485, 213), which
include the partial counts of the repeats that the threshold ended (KD-13).
In PyIBS a repeat that the threshold ended counts -T as a whole and has no
share in any trial, so `neg_logl_trials` and `neg_logl_var_trials` are NaN
when the threshold ended a repeat. Otherwise they are each trial's
average over its completed repeats, all of them unless `max_time` stopped
the sampling (KD-14), and its variance estimate, unweighted as in
`ibslike.m`; weighted, they add up to the returned estimate and its
variance. Even so, under a threshold they are finite only on draws whose
repeats all stayed above -T, which favours small values of
`neg_logl_trials`: a user who wants per-trial values sets no threshold.
With a single trial, `ibslike.m`'s arrays are not per trial whenever a
count exceeds 1. It indexes its column table of count terms with the
1-by-`Nreps` row of the trial's counts, which gives a column that the sum
over the second dimension leaves as it is (lines 210-213, 399, 413,
484-485, 498): `nlogLvar_trials`, and on the loop path `nlogL_trials`,
hold one entry per repeat, which add up to the totals (lines 225, 228).
PyIBS's arrays have shape (N,) at every N.
- PyIBS: `IBS.__call__` (`pyibs/ibs.py`); `trial_value_sums`,
  `trial_var_sums` and `trial_counts` of `_Draw` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:210-213`, `220-221`, `225`, `228`, `396-399`, `413`,
  `483-485`, `498`.
- Settled by: D21. Kind: deliberate change.

**KD-4. Every random draw comes from one generator, created from
`random_seed`, which the simulator receives when it takes `rng`.**
`ibslike.m` has no seed option: the simulator draws from MATLAB's global
stream (as its examples do, lines 56, 510). `IBS(..., random_seed=...)`
creates the object's `numpy.random.Generator`, `rng`, as PyBADS's
`random_seed` does: None derives it from NumPy's global random state, an
integer or a `SeedSequence` seeds a new one, a `Generator` is used as
given. The simulator is called as `sample_from_model(params, design_rows,
rng=rng)` when its signature has a parameter named `rng` that takes a
keyword, and as `sample_from_model(params, design_rows)` otherwise, so
that a simulator written for 0.1.0 keeps working. Two objects created with
the same seed give the same estimates when the simulator draws from `rng`
and no timing decides the sampling (KD-8, KD-9, KD-14).
- PyIBS: `_rng`, `_takes_rng`, `IBS.__init__` (`pyibs/ibs.py`).
- MATLAB: `ibslike.m:56`, `510`.
- Settled by: D8, D18. Kind: Python-only feature.

**KD-5. Without a design, the simulator receives 0-based trial indices.**
With an empty design, `ibslike.m` passes the trial numbers `1:Ntrials`
(lines 23-26, 170, 183, 293, 436). PyIBS passes 0-based indices, which
index Python arrays.
- PyIBS: `_simulate` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:23-26`, `170`, `183`, `293`, `436`.
- Kind: deliberate change (interface).

**KD-6. The settings and the arguments of a call are checked, by type and
by range.**
`ibslike.m` checks that `NsamplesPerCall` is a numeric scalar, that
`Acceleration` is at least 1, that `NegLogLikeThreshold` and `MaxTime` are
positive, and that `TrialWeights` has one element or one per trial (lines
139-168). PyIBS checks every setting when `IBS` is created, and `num_reps`,
`trial_weights`, `additional_output` and `return_positive` at each call.
It raises `ValueError` for an invalid value and `TypeError` for the wrong
type, naming the argument and its requirements. `return_positive` takes
only Python and NumPy booleans, where `ibslike.m` (line 226) and PyIBS 0.1.0
test the truth of any value, so that the string `"False"` returns the
log-likelihood. The counts (`num_reps`,
`num_samples_per_call`, `max_iter`, `max_samples`, `max_mem`) take integers and
whole-number floats, such as `1e5`, and refuse booleans, fractions and
infinity; `num_samples_per_call` is at least 0, the others at least 1.
`ibslike.m` takes `MaxIter = Inf`, which disables its cap (KD-11), and
`NsamplesPerCall = Inf`, which starts the samples per call at their bounds
(lines 264-266, 281-283). `acceleration` is finite. Trial weights are real
numbers, finite and at least 0, and refuse booleans and strings. `num_reps` is
one integer: `ibslike.m` reads `Nreps` as a vector of per-trial repeats in
places (lines 248, 259-260), which no documentation offers and its code does
not support: in MATLAB, whose `&&` takes scalars, a vector stops at line 193
when `Vectorized` is given; the loop path fails at line 413; and the vectorized
path fails at line 284 unless `NsamplesPerCall` is set, and otherwise, once
fewer than all trials are open, fails at line 361 or 372 or misplaces the
counts (lines 193, 264, 281-284, 361, 372, 413). The data are checked too: the
responses are a non-empty array of one or two dimensions; the design has one
row per trial, where `ibslike.m` ignores extra rows and fails at an index when
rows are missing (lines 185, 296, 439); and the trial weights are a scalar or
have shape (N,), where `ibslike.m` flattens any array of one or N elements
(line 163).
- PyIBS: `IBS.__init__`, `IBS.__call__` (`pyibs/ibs.py`); `_check_count`,
  `_check_real`, `_Settings` (`pyibs/_sampler.py`); `trial_weights`
  (`pyibs/_estimates.py`).
- MATLAB: `ibslike.m:139-168`, `185`, `193`, `226`, `248`, `259-266`,
  `269`, `281-284`, `296`, `361`, `372`, `413`, `427`, `439`.
- Settled by: D16, D25. Kind: deliberate change.

## Sampling

**KD-7. One sampler runs both schedules; `vectorized=False` samples all
repeats at once, one sample per open trial and call.**
`ibslike.m` has two code paths: the vectorized path (lines 237-401), which
samples all repeats of all trials rows-first, several samples per trial
and call, and the loop path (lines 406-487), which samples one repeat after
another, one sample of every trial that has not matched in that repeat per
call. PyIBS has one sampler, the vectorized path's design, whose schedule
`vectorized` selects: `True` requests `ibslike.m`'s number of samples per
trial and call (lines 281-283, with MATLAB's `round`), `False` requests one
sample of every trial that still needs a match, in whichever repeat it is
sampling. With `False`, as in the loop path, a simulator call requests at
most N rows and no sample is drawn after a trial's last match; but a trial
that completes a repeat goes on to its next repeat in the next simulator
call, so the sampling of n repeats takes as many simulator calls as the
trial with the most samples, where the loop path takes, summed over the
repeats, the most samples of a trial in each. `True` with `num_reps=1`
falls back to `False` with a warning, as in `ibslike.m` (lines 193-196).
- PyIBS: `sample`, `_samples_per_trial` (`pyibs/_sampler.py`);
  `IBS.__call__` (`pyibs/ibs.py`).
- MATLAB: `ibslike.m:193-196`, `199-205`, `237-401`, `281-283`,
  `406-487`.
- Settled by: D7. Kind: deliberate change.

**KD-8. `vectorized=None` is decided once per object.**
With `Vectorized` `'auto'`, `ibslike.m` times one simulation of all trials
at every call with `Nreps > 1`, and takes the loop path when it lasts
`VectorizedThreshold` (0.1 s) or more (lines 176-190). PyIBS makes the same
timing call, with the same rule, at the object's first call with
`num_reps > 1`, and keeps the decision for its later calls, readable as
`ibs.vectorized`; a call with `num_reps=1` takes the one-sample schedule
whatever the setting (lines 177-178). As in `ibslike.m`, the timing call
counts as a simulator call (line 189). Its samples are always the first
round of the sampling: the schedule's first round when that round requests
one sample of every trial, as in `ibslike.m` (lines 287-289, 433-434), and
otherwise an extra round before it, after which the samples per call do
not grow, so that the rounds after it are those of `ibslike.m`'s schedule.
`ibslike.m` discards them in that case (lines 287-289), so it uses them only
when the timing call was slow; when the simulator's running time depends on
what it simulates, whether the samples are used then depends on their
values, and the estimate is biased. The decision follows `ibslike.m`'s
rule, and the object's later calls stay on one schedule however the timing
of their simulations varies.
- PyIBS: `IBS.__call__` (`pyibs/ibs.py`); `first_round`, `sample`
  (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:135`, `176-190`, `287-289`, `433-434`.
- Settled by: D19. Kind: deliberate change.

**KD-9. The samples per call grow after every call by default; the time
rule of `ibslike.m` is opt-in.**
`ibslike.m` multiplies its level of samples per trial by `Acceleration`
only after a simulator call faster than `AccelerationThreshold`, 0.1 s
(lines 134, 310-313), so that the samples requested, and the assignment of
random numbers to trials, depend on the wall-clock time. PyIBS multiplies it
after every call by default (`acceleration_threshold=None`), so that a seed
reproduces a run; `acceleration_threshold=0.1` restores `ibslike.m`'s rule.
Either rule skips the extra first round of `vectorized=None` (KD-8). The
level is bounded by `max_samples`, which changes no request. The
values of complete repeats do not depend on the schedule; it changes the
cost, the variance estimate of a repeat that the threshold ends, whether a
call reaches the cap, which counts the samples drawn after a trial's last
match, and, under the time limit, which repeats complete.
- PyIBS: `sample` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:134`, `262-267`, `310-313`.
- Settled by: D4. Kind: deliberate change.

**KD-10. A call requests each trial's samples together.**
`ibslike.m` requests `Tmat(:)` with `Tmat = repmat(T,[1,Nsamples])` (lines
284, 293, 296): every open trial once, then every open trial again, and so
on. PyIBS requests the samples of each open trial together,
`np.repeat(open_trials, m)`, so that each trial's outcomes are contiguous.
Every requested row is an independent draw, so the order changes no
distribution; it changes which random numbers a seeded simulator gives
which trial.
- PyIBS: `_simulate` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:284`, `293`, `296`, `316-319`.
- Kind: deliberate change (implementation).

**KD-11. The cap counts the samples of one trial, surplus included, and
raises `IBSSamplingError`.**
`ibslike.m` caps the rounds of simulator calls: `MaxIter * Nreps` rounds in
the vectorized path and `MaxIter` per repeat in the loop path, and raises
`ibslike:ConvergenceFail` when trials were still open at the start of the
last round it allows, even if that round finished them (lines 260,
269-276, 390-393, 410, 427-430, 477-480). A round can hold many samples per
trial. PyIBS raises `IBSSamplingError`, a `RuntimeError`, once a trial has
drawn more than `max_iter * num_reps` samples in a call, the samples after
its last match included, checked after every simulator call, so that it
too raises after the simulator call that crosses the cap even if that call
completes the sampling. The cap thus means what `MaxIter`'s description
says, "per trial and estimate" (line 95), scaled by the repeats as the
vectorized path scales it. The timing call of `vectorized=None` is a
round (KD-8), so its sample of each trial counts toward the cap;
`ibslike.m`'s timing call is a round only when its first round requests one
sample of every trial. The error names the trials over the cap by their
0-based indices and the samples each drew. `MaxIter = Inf` disables
`ibslike.m`'s cap in effect: its loops run until the sampling ends (lines
260, 269, 427), since Octave, and presumably MATLAB, bound an infinite
range only at 2^63 - 1 iterations. `max_iter` is a finite integer, so that
a call has a cap unless the user chooses one so large, such as `10**18`,
that no call reaches it.
- PyIBS: `sample`, `_cap_error`, `IBSSamplingError` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:95`, `180-189`, `260`, `269-276`, `390-393`, `410`,
  `427-430`, `477-480`.
- Settled by: D3, D6. Kind: deliberate change.

**KD-12. The cost counts every simulated response.**
`ibslike.m`'s vectorized path adds to its sample count the number of open
trials per round, whatever the samples requested of each (line 308), and
leaves out the timing call of `'auto'` unless that call is the first round
(lines 287-289); its loop path counts every row (line 449);
`NsamplesPerTrial` divides the count by N (line 219). PyIBS's
`num_samples_per_trial` divides by N every response that the simulator
returned in the call, the timing call and the samples drawn after a
trial's last match included, the measure of the simulator's work.
`fun_count` counts the simulator calls, as `funcCount` does (lines 189,
206, 294, 297, 437, 440).
- PyIBS: `sample` (`pyibs/_sampler.py`); `IBS.__call__` (`pyibs/ibs.py`).
- MATLAB: `ibslike.m:189`, `206`, `219`, `287-289`, `294`, `297`, `308`,
  `437`, `440`, `449`.
- Kind: deliberate change.

## The likelihood threshold

**KD-13. Every repeat is checked against a weighted bound, and a repeat
that exceeds it counts exactly -T.**
`ibslike.m`'s vectorized path checks, after every round, only the lowest
repeat that a trial is still sampling (line 377). Its bound takes the
completed counts and, for the trials still sampling that repeat, their open
count c, which its count matrix holds at a trial's current repeat, 0 when
the trial's last sample matched (written at lines 347-373, read at line
379), and it compares the unweighted sum of the terms with T (line 380).
Its loop path checks the repeat it samples after every call, with c + 1
for the trials that have not matched (lines 456-466), also unweighted. In
both, the partial counts of an ended repeat stay among the trials' counts
and enter their averages (lines 382-384, 396-398, 483-485), so that the
estimate depends on the sampling schedule; its own comment calls the
threshold incompatible with vectorized sampling (line 91). PyIBS follows
the IBS paper, Appendix C.1, in checking every repeat and in valuing an
ended repeat at exactly -T: after every round, it checks every repeat that
is not ended against a running bound of its weighted negative
log-likelihood, and ends the repeats whose bound exceeds T; a complete
repeat whose value lies below -T is ended too. Its bound counts c + 1 for a
trial sampling the repeat, as the loop path does: one term more than the
paper's c, and still a lower bound on the trial's final term. Every
repeat's value is then `max(Y_r, -T)`, a function of its own draws,
whatever the schedule; an ended repeat's variance estimate is that of its
counts when it was ended. The weighted bound is on the scale of the
returned estimate. The exit flag is 1 when a repeat was ended, as in
`ibslike.m` (lines 385, 463).
- PyIBS: `_MatchCounts.bounds`, `end_above`, `end_complete_below`,
  `clipped_estimates`, `sample` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:91`, `347-387`, `396-398`, `456-466`, `483-485`.
- Settled by: D5. Kind: deliberate change.

## The time limit

**KD-14. When the time limit stops the sampling, each trial averages its
completed repeats, and a trial without one raises.**
`ibslike.m` counts `MaxTime` from the start of the call (line 86), checks
it before every round of its vectorized path (lines 272-275), and before
every repeat and after every call of its loop path (lines 422-425,
468-473), and stops with exit flag 2, also when the sampling happens to be
complete at that check. Its vectorized path then averages each trial's
positive counts, among them the partial count of the repeat it was
sampling (lines 347-373, 396-398), whose value depends on the schedule;
when the limit has passed before its first round, as when the timing call
of `'auto'` outlasts `MaxTime` but not `VectorizedThreshold`, no trial has
a count, and every value and variance is 0/0, NaN (lines 210-213,
396-398). Its loop path averages each trial's positive counts too: without a
threshold, these are its completed repeats, and a trial without one gets
0/0, NaN; under a threshold, they also include the c + 1 that line 458
sets for every trial that has not matched in the stopped repeat (lines
456-458, 483-485). Nothing warns. PyIBS counts `max_time` from the start
of the call too, and checks it after every simulator call of the call's
sampling while trials still need matches, so that a sampling that
completes is not flagged. Once `max_time` has passed, each trial's value
averages its completed counts in the repeats that the likelihood threshold
did not end, and the ended repeats count -T each: of n repeats with n_e
ended, the log-likelihood estimate is (n_e / n)(-T) + (1 - n_e / n) times
the weighted sum of the trials' averages. The call has exit flag 2 and
issues a `UserWarning`, since the exit flag is not seen in the `"std"`
output that PyBADS and PyVBMC take. A trial with no completed count
raises `IBSSamplingError`. In `ibslike.m` as in PyIBS, a finite time limit
also biases the calls that complete in time: completing in time is more
likely with small counts, which give high log-likelihoods, so only an
infinite `max_time` gives unbiased estimates. Such a call has exit flag 0,
or 1 when the threshold ended a repeat. PyIBS's message of flag 0 states
that every repeat completed, where `ibslike.m` describes flag 0 as "Correct
run of IBS; the estimate is unbiased" (line 44).
- PyIBS: `sample`, `_limited_estimates` (`pyibs/_sampler.py`);
  `IBS.__call__` (`pyibs/ibs.py`).
- MATLAB: `ibslike.m:44`, `86`, `210-213`, `272-275`, `347-373`,
  `396-398`, `422-425`, `456-458`, `468-473`, `483-485`.
- Settled by: D3, D24. Kind: deliberate change.

## Responses and matching

**KD-15. Responses of one column take an output of shape (r,) or (r, 1);
others the shape (r, C).**
`ibslike.m` checks only that the simulator returns one row per requested
trial (lines 303-306, 443-447), and compares with
`all(respMat(T,:) == simdata, 2)` (lines 316, 450), which MATLAB's implicit
expansion broadcasts: an output of one column is compared with every
column of the responses. PyIBS takes, for r requested rows, an output of
shape (r,) or (r, 1) when the responses have one column, (N,) or (N, 1),
so that a model ported from MATLAB runs unchanged, and of shape (r, C) when
they have C > 1 columns; any other shape raises `ValueError`. A simulated
row matches only when every column agrees, as in `ibslike.m`.
- PyIBS: `_simulate` (`pyibs/_sampler.py`).
- MATLAB: `ibslike.m:303-306`, `316`, `443-447`, `450`.
- Settled by: D22. Kind: deliberate change.

**KD-16. Responses that NumPy never finds equal to the simulated ones
raise `TypeError`.**
NumPy compares text, bytes, and numbers or booleans as unequal to one
another whatever their values, where MATLAB compares characters with
numbers by their codes. A simulator that returns responses of another of
these kinds than the observed ones would never match, and every trial
would sample until the cap, or forever in `ibs_basic`; `IBS` raises
`TypeError` after the first such simulator call, and `ibs_basic` after the
first such response. An object array of simulated rows is compared element
by element and not checked. An object array of responses is checked by the
kinds of its elements, so that rows that NumPy has made text, as it does of
rows that mix numbers and text, raise against responses holding numbers.
- PyIBS: `_check_kinds`, `_simulate` (`pyibs/_sampler.py`); `ibs_basic`
  (`pyibs/ibs_basic.py`).
- MATLAB: `ibslike.m:316`, `450`; `ibs_basic.m:33`.
- Kind: Python-only feature.

**KD-17. A NaN response raises `ValueError` when `IBS` is created.**
A NaN never equals a simulated response. In `ibslike.m` its trial samples
until the cap and fails there; under a likelihood threshold that the
trial's growing term reaches before the cap, every repeat ends instead, at
every parameter vector. Nothing names the cause. PyIBS refuses responses
that hold a NaN, an element not equal to itself (a float or complex NaN,
`NaT`, a NaN in an object array), naming their trials. The design is not
checked: a design of NaN serves a simulator that reads only its size, as
`ibslike.m`'s examples call theirs (lines 57, 511).
- PyIBS: `_check_responses` (`pyibs/_sampler.py`), called by
  `IBS.__init__` and `ibs_basic`.
- MATLAB: `ibslike.m:57`, `316`, `390-393`, `450`, `477-480`, `511`.
- Settled by: D23. Kind: deliberate change.

## ibs_basic

**KD-18. `ibs_basic` runs without a design, passes a generator, and
raises on its data and on responses that cannot match or have another
shape.**
`ibs_basic.m` indexes the design, `S(i,:)`, which it requires (line 33); it
returns 0 for empty responses (lines 28-29, 39), ignores extra rows of `S` and
fails at an index when rows are missing (line 33); and it loops forever on a
NaN response, since `NaN ~= NaN` (line 33). PyIBS's
`ibs_basic(sample_from_model, theta, R, S=None, *, random_seed=None)` passes
the 0-based trial index when `S` is None, as `IBS` does (KD-5), creates a
generator from `random_seed` and passes it to a simulator that takes `rng`, as
`IBS` does (KD-4), checks that `R` is a non-empty array of one or two
dimensions and that `S` has one row per trial, and refuses a NaN response
(KD-17) and a simulated response of a kind that NumPy never finds equal to `R`
(KD-16), as `IBS` does. It takes a simulated response of shape (C,) or
(1, C) for responses of C columns, or a scalar when C = 1, and raises
`ValueError` for any other shape. `ibs_basic.m` compares with
`any(fun(theta,S(i,:)) ~= R(i,:))` (line 33), which MATLAB's implicit
expansion broadcasts: a column of C values is compared as a C × C array,
and an empty output gives an empty comparison, for which `any` is false,
so that the loop ends as on a match. PyIBS returns the log-likelihood estimate, not its negative, as
`ibs_basic.m` does (line 39).
- PyIBS: `ibs_basic` (`pyibs/ibs_basic.py`).
- MATLAB: `ibs_basic.m:28-39`.
- Settled by: D22 (the shapes), D23 (the NaN response). Kind: deliberate
  change.

## Self-tests

**KD-19. `ibslike('test')` is the test suite.**
`ibslike('test')` runs `runtest1` to `runtest3` and plots their results
(lines 110-122, 503-698). PyIBS ports the three to
`pyibs/testing/test_ibslike_ports.py`, run by `pytest --pyargs pyibs`,
through the public `IBS`, without figures. `ibslike.m` checks `runtest1`
and `runtest3` at one random state; a correct sampler fails either by
chance at about one seed in a thousand, so each port checks three seeds and
passes at two. `runtest2` keeps `ibslike.m`'s criteria at one seed. The
ports run with `vectorized=True`, and those of `runtest1` and `runtest2`
with the samples per call growing after every call, where these self-tests
leave `Vectorized` at `'auto'` and the acceleration under its time rule
(lines 517-518, 578), so that a seed alone decides their draws; `runtest3`
and its port set the acceleration to 1 (line 641). The ports pass on a
strict `<` at the tolerances of the RMSE and of the z-scores' mean and SD,
where `ibslike.m` passes on `<=` (lines 535, 610, 666).
- PyIBS: `pyibs/testing/test_ibslike_ports.py`.
- MATLAB: `ibslike.m:110-122`, `503-698`.
- Kind: removed feature (`ibslike('test')`), substituted by the test
  suite.
