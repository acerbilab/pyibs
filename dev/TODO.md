# PyIBS: open work

Updated 2026-10-10. Other records name the items by their titles, so a
title stays as it is while its item is open.

## To investigate

- [ ] **A floor on the SD that a target gives PyBADS or PyVBMC.** An IBS
  call returns a variance estimate of exactly 0 when every trial of
  positive weight matches its response at its first sample in every
  repeat, and PyBADS and PyVBMC refuse an SD of 0 (`ValueError`). `IBS`
  returns the 0 as computed, with a warning that links the FAQ's answer
  "Why is the SD of the estimate zero, and why do PyBADS and PyVBMC refuse
  it?" (D10 of `plans/pyibs-1.5.md` rejects a floor inside PyIBS, which
  would misstate the precision). That answer offers more repeats and a
  lapse rate in the model. A further remedy, derived while it was written
  and kept out of it until it is studied (PI, 2026-10-09): the user's
  target returns `max(sd, 1 / num_reps)`.
  - The value: with unit trial weights, a trial's variance estimate in a
    repeat is `ψ₁(1) − ψ₁(K)`, 0 at `K = 1` and 1 at `K = 2`, and a call
    divides the sum over trials and repeats by `num_reps ** 2`, so
    `1 / num_reps` is the smallest positive SD that a call returns, that of
    one trial that needed a second sample in one repeat. The floor
    therefore changes no other SD. With trial weights, the smallest
    positive weight divided by `num_reps` plays its role.
  - Its size, with unit trial weights and `p_i` the probability that the
    simulator produces trial i's response: with `S = sum_i (1 - p_i)`, a
    call returns 0 with probability
    `prod_i p_i ** num_reps ≈ exp(-num_reps * S)`, and the true variance of
    its estimate is `sum_i Li₂(1 - p_i) / num_reps ≈ S / num_reps` (van
    Opheusden, Acerbi & Ma, 2020, PLOS Computational Biology 16(12):
    e1008483, Section 4.3). Since `Li₂(1 - p) ≤ -log p`, the probability
    of a 0 is at most `exp(-(num_reps * sd) ** 2)`, `sd` the true SD,
    whatever the number of trials: 0.37 at `sd = 1 / num_reps`, 0.018 at
    `2 / num_reps`. A 0 thus points to a true SD below about
    `2 / num_reps`.
  - Open: whether the advice holds up in practice. Fits with PyBADS and
    PyVBMC on a model whose trials match with probability near 1, with
    and without the floor, would show whether the floor changes their
    results; a much smaller floor, such as `1e-8`, would let PyBADS's
    precision-weighted final estimate rest on that one evaluation. The
    derivation has been checked numerically, but no fit has used the
    floor.
- [ ] **Trial weights large enough to overflow the variance estimate.**
  `trial_weights` takes any finite weight of at least 0, and the variance
  estimate sums `var * w**2` over the trials (`repeat_estimates`,
  `pyibs/_estimates.py`), which overflows once the weighted terms or their
  sum exceed the largest float, about `1.8e308`. On 20 Bernoulli
  trials with 10 repeats, `trial_weights=1e154` gives `neg_logl_var=inf`,
  and `1e160` gives NaN, from `0 * inf` in a trial whose variance term is 0,
  each with exit flag 0 and only NumPy's `RuntimeWarning`s; with
  `neg_logl_threshold=10` and `1e308`, the call returns 10 with a variance
  of NaN, an SD that PyBADS and PyVBMC refuse.
  - Open: a bound on the weights in their check (`trial_weights`,
    `pyibs/_estimates.py`), or a sentence in the docstring of
    `IBS.__call__`; no weighting in practical use comes near such values.
